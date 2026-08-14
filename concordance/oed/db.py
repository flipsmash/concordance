"""`oed` schema DDL + read/write helpers.

Lives in the same Postgres database as the `concordance` schema (its own
schema, so no name clashes — same pattern `vocab.wiktionary` already uses),
but is otherwise fully standalone: no FKs into `concordance.word`, no shared
tables. Connect with `concordance.db.connect()` / `concordance.db.database_url()`
directly; this module only owns the `oed.*` DDL and queries.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import psycopg

DEFAULT_SCHEMA = "oed"

_SCHEMA_DDL = """
CREATE SCHEMA IF NOT EXISTS {s};

-- One row per source PDF. file_hash drives idempotent re-ingestion: a
-- volume already marked 'done' with a matching hash is skipped; a changed
-- file (replaced/rescanned) re-ingests under --force.
CREATE TABLE IF NOT EXISTS {s}.volume (
    id          serial PRIMARY KEY,
    file_name   text NOT NULL,
    file_hash   text NOT NULL UNIQUE,
    volume_label text,
    page_count  integer,
    status      text NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'processing', 'done', 'error')),
    error_detail text,
    last_page_committed integer NOT NULL DEFAULT 0,
    ingested_at timestamptz,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- One row per headword — main entries AND bold run-on/compound sub-entries,
-- since those are typographically their own headwords in OED's layout.
CREATE TABLE IF NOT EXISTS {s}.entry (
    id                 serial PRIMARY KEY,
    volume_id          integer NOT NULL REFERENCES {s}.volume(id) ON DELETE CASCADE,
    headword           text NOT NULL,
    headword_norm       text GENERATED ALWAYS AS (lower(headword)) STORED,
    homograph_number   integer,
    part_of_speech     text,
    etymology          text,
    entry_type         text NOT NULL DEFAULT 'main'
                       CHECK (entry_type IN ('main', 'run_on', 'compound', 'variant_form')),
    parent_entry_id    integer REFERENCES {s}.entry(id) ON DELETE SET NULL,
    page_number        integer NOT NULL,
    raw_text           text NOT NULL,

    -- Pronunciation provenance (see pronunciation.py). The baked-in OCR text
    -- layer is never trusted for this field (see project notes) --
    -- pronunciation_raw is kept only for debugging, never surfaced as truth.
    pronunciation_raw       text,
    pronunciation_pass1     text,
    pronunciation_pass2     text,
    pronunciation_ipa       text,
    pronunciation_source    text CHECK (pronunciation_source IN ('vision_llm', 'manual')),
    pronunciation_needs_review boolean NOT NULL DEFAULT true,

    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS entry_headword_norm_idx ON {s}.entry (headword_norm);
CREATE INDEX IF NOT EXISTS entry_volume_id_idx ON {s}.entry (volume_id);
CREATE INDEX IF NOT EXISTS entry_needs_review_idx ON {s}.entry (pronunciation_needs_review)
    WHERE pronunciation_needs_review;

-- One row per numbered/lettered sense within an entry.
CREATE TABLE IF NOT EXISTS {s}.definition (
    id            serial PRIMARY KEY,
    entry_id      integer NOT NULL REFERENCES {s}.entry(id) ON DELETE CASCADE,
    sense_label   text,
    definition_text text NOT NULL,
    sort_order    integer NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS definition_entry_id_idx ON {s}.definition (entry_id);

-- One row per dated illustrative citation. author/source_title carry the
-- same baked-OCR noise as the rest of the prose text (unlike pronunciation,
-- this isn't symbolic and isn't given the double-pass treatment) — raw_text
-- on the parent entry is the recourse if a field looks garbled.
CREATE TABLE IF NOT EXISTS {s}.quotation (
    id            serial PRIMARY KEY,
    definition_id integer NOT NULL REFERENCES {s}.definition(id) ON DELETE CASCADE,
    year_raw      text,
    year          integer,
    year_approx   boolean NOT NULL DEFAULT false,
    author        text,
    source_title  text,
    quoted_text   text NOT NULL,
    sort_order    integer NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS quotation_definition_id_idx ON {s}.quotation (definition_id);

CREATE INDEX IF NOT EXISTS definition_text_fts_idx
    ON {s}.definition USING gin (to_tsvector('english', definition_text));
CREATE INDEX IF NOT EXISTS quotation_text_fts_idx
    ON {s}.quotation USING gin (to_tsvector('english', quoted_text));
"""


def apply_schema(conn: psycopg.Connection, schema: str = DEFAULT_SCHEMA) -> None:
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_DDL.format(s=schema))
        # Is this entry's headword the lemma (base/citation) form of the
        # word, as opposed to an inflected form OED gave its own headword
        # entry (e.g. "abandoned"/"ppl. a", a past-participial adjective
        # derived from "abandon")? See concordance/oed/lemma.py for the
        # computation. NULL lemma_computed_at means not yet computed --
        # distinct from a computed false, so an incremental run (new
        # volumes keep arriving) only touches entries not yet checked.
        cur.execute(f"ALTER TABLE {schema}.entry ADD COLUMN IF NOT EXISTS lemma boolean NOT NULL DEFAULT false")
        cur.execute(f"ALTER TABLE {schema}.entry ADD COLUMN IF NOT EXISTS lemma_computed_at timestamptz")
        cur.execute(f"CREATE INDEX IF NOT EXISTS entry_lemma_idx ON {schema}.entry (lemma) WHERE lemma")
        cur.execute(f"CREATE INDEX IF NOT EXISTS entry_lemma_uncomputed_idx ON {schema}.entry (id) "
                    f"WHERE lemma_computed_at IS NULL")
    conn.commit()


def _close_unbalanced_paren(ipa: str) -> str:
    """Fixes a known pronunciation.py extraction artifact: confirmed live on
    615/5007 resolved entries (e.g. "abandoner" -> "ˈbændənə(r", missing the
    closing paren on OED's linking-r marker). Only closes the specific case
    of exactly one more '(' than ')' -- anything else is left alone rather
    than guessed at."""
    if ipa.count("(") == ipa.count(")") + 1:
        return ipa + ")"
    return ipa


def pronunciation_lexicon(conn: psycopg.Connection, headwords: set[str],
                           schema: str = DEFAULT_SCHEMA) -> dict[str, list[str]]:
    """headword_norm -> every distinct resolved pronunciation_ipa across its
    entries -- a headword can have many oed.entry rows (homographs: "bay",
    "fleet", "back" each have up to 10 in this corpus), and most of those
    rows have no resolved pronunciation at all (pronunciation_ipa IS NULL --
    confirmed live: only 5007/24892). Callers decide what to do with >1
    distinct value; this just reports what's there. Mirrors
    localdict.build_lexicon's bulk-lookup shape (one query for every
    candidate headword, empty dict on an empty input rather than a
    `= ANY('{}')` query that would just scan nothing).

    Degrades to {} (not an error) if the oed schema/entry table doesn't
    exist yet -- this is now also called from compute_ipa's regular
    kaikki/Wordnik/local-Wiktionary/oed cascade (resolve_pronunciation.py),
    not just the standalone oed-ipa command someone would only run after
    already setting up oed-ingest, so a plain `concordance ipa`/`maintain`
    on a DB that's never touched OED must not start hard-erroring because
    of a tier it was never relying on before. Same to_regclass pattern
    import_defined_words uses for vocab.defined's optional presence."""
    if not headwords:
        return {}
    s = schema
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{s}.entry",))
        if cur.fetchone()[0] is None:
            return {}
        cur.execute(
            f"""SELECT headword_norm, pronunciation_ipa FROM {s}.entry
                WHERE headword_norm = ANY(%s) AND pronunciation_ipa IS NOT NULL""",
            (list(headwords),),
        )
        rows = cur.fetchall()
    lexicon: dict[str, list[str]] = {}
    for headword_norm, ipa in rows:
        cleaned = _close_unbalanced_paren(ipa)
        entries = lexicon.setdefault(headword_norm, [])
        if cleaned not in entries:
            entries.append(cleaned)
    return lexicon


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_volume(conn: psycopg.Connection, file_hash_: str, schema: str = DEFAULT_SCHEMA) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, file_name, status, last_page_committed "
            f"FROM {schema}.volume WHERE file_hash = %s",
            (file_hash_,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {"id": row[0], "file_name": row[1], "status": row[2], "last_page_committed": row[3]}


def upsert_volume(conn: psycopg.Connection, *, file_name: str, file_hash_: str,
                   volume_label: str | None, page_count: int, schema: str = DEFAULT_SCHEMA) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {schema}.volume (file_name, file_hash, volume_label, page_count, status)
            VALUES (%s, %s, %s, %s, 'pending')
            ON CONFLICT (file_hash) DO UPDATE SET file_name = EXCLUDED.file_name
            RETURNING id
            """,
            (file_name, file_hash_, volume_label, page_count),
        )
        volume_id = cur.fetchone()[0]
    conn.commit()
    return volume_id


def set_volume_status(conn: psycopg.Connection, volume_id: int, status: str, *,
                       last_page_committed: int | None = None, error_detail: str | None = None,
                       schema: str = DEFAULT_SCHEMA) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {schema}.volume
            SET status = %s,
                last_page_committed = COALESCE(%s, last_page_committed),
                error_detail = %s,
                ingested_at = CASE WHEN %s = 'done' THEN now() ELSE ingested_at END
            WHERE id = %s
            """,
            (status, last_page_committed, error_detail, status, volume_id),
        )
    conn.commit()


def insert_entry(conn: psycopg.Connection, *, volume_id: int, headword: str,
                  homograph_number: int | None, part_of_speech: str | None,
                  etymology: str | None, entry_type: str, parent_entry_id: int | None,
                  page_number: int, raw_text: str, schema: str = DEFAULT_SCHEMA) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {schema}.entry
                (volume_id, headword, homograph_number, part_of_speech, etymology,
                 entry_type, parent_entry_id, page_number, raw_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (volume_id, headword, homograph_number, part_of_speech, etymology,
             entry_type, parent_entry_id, page_number, raw_text),
        )
        return cur.fetchone()[0]


def insert_definitions(conn: psycopg.Connection, entry_id: int,
                        senses: list[dict], schema: str = DEFAULT_SCHEMA) -> list[int]:
    """`senses`: [{"sense_label": str|None, "definition_text": str, "quotations": [...]}]."""
    ids = []
    with conn.cursor() as cur:
        for order, sense in enumerate(senses):
            cur.execute(
                f"""
                INSERT INTO {schema}.definition (entry_id, sense_label, definition_text, sort_order)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (entry_id, sense.get("sense_label"), sense["definition_text"], order),
            )
            def_id = cur.fetchone()[0]
            ids.append(def_id)
            for q_order, q in enumerate(sense.get("quotations", [])):
                cur.execute(
                    f"""
                    INSERT INTO {schema}.quotation
                        (definition_id, year_raw, year, year_approx, author, source_title,
                         quoted_text, sort_order)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (def_id, q.get("year_raw"), q.get("year"), q.get("year_approx", False),
                     q.get("author"), q.get("source_title"), q["quoted_text"], q_order),
                )
    return ids


def update_pronunciation(conn: psycopg.Connection, entry_id: int, *, pronunciation_raw: str | None,
                          pass1: str | None, pass2: str | None, ipa: str | None,
                          source: str | None, needs_review: bool, schema: str = DEFAULT_SCHEMA) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {schema}.entry
            SET pronunciation_raw = %s, pronunciation_pass1 = %s, pronunciation_pass2 = %s,
                pronunciation_ipa = %s, pronunciation_source = %s, pronunciation_needs_review = %s
            WHERE id = %s
            """,
            (pronunciation_raw, pass1, pass2, ipa, source, needs_review, entry_id),
        )


def compute_lemma_flags(conn: psycopg.Connection, schema: str = DEFAULT_SCHEMA, *,
                        only_missing: bool = True, limit: int = 0, commit_every: int = 500) -> dict:
    """Backfill entry.lemma/lemma_computed_at -- see lemma.py's module
    docstring for the computation. only_missing=True (default) only touches
    entries not yet checked (lemma_computed_at IS NULL), so this is safe to
    re-run periodically as new volumes keep landing in oed-ingest without
    redoing the whole table each time."""
    from . import lemma as _lemma
    from .. import tokenize

    where = "WHERE lemma_computed_at IS NULL" if only_missing else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, headword_norm, part_of_speech FROM {schema}.entry "
                    f"{where} ORDER BY id" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    stats = {"entries": len(rows), "lemma": 0, "not_lemma": 0}
    if not rows:
        return stats

    nlp = tokenize.load_nlp()
    with conn.cursor() as cur:
        for i, (entry_id, headword_norm, pos) in enumerate(rows, 1):
            is_lemma = _lemma.is_lemma(nlp, headword_norm, pos)
            stats["lemma" if is_lemma else "not_lemma"] += 1
            cur.execute(
                f"UPDATE {schema}.entry SET lemma = %s, lemma_computed_at = now() WHERE id = %s",
                (is_lemma, entry_id))
            if i % commit_every == 0:
                conn.commit()
    conn.commit()
    return stats
