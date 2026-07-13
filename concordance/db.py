"""Sync the master vocabulary list into PostgreSQL (§07 follow-on).

The CSV stays the working format; this mirrors it into a database so a future web
app (and eventual integration with the related project) has a real, queryable
store. Tables live in their own schema (default ``concordance``) so they can share
a database with other projects without name clashes.

Normalisation vs the flat CSV: the ``source_book`` cell (a "BookA; BookB" list) is
split into a proper many-to-many via ``word_book``; ``synonyms`` becomes a text[].
Everything is upsert-based and idempotent — re-running ``sync-db`` reconciles the
DB with the current CSV.

Connection comes from ``DATABASE_URL`` (env or a git-ignored .env), e.g.
    DATABASE_URL=postgresql://user:pass@host:5432/dbname
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import csv

import psycopg
import requests

from .deepdef import _load_dotenv
from .model import normalize_pos

DEFAULT_SCHEMA = os.environ.get("CONCORDANCE_DB_SCHEMA", "concordance")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def database_url(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if "DATABASE_URL" not in os.environ:
        _load_dotenv(Path(".env"))
    return os.environ.get("DATABASE_URL", "").strip()


def _safe_schema(schema: str) -> str:
    if not _IDENT.match(schema):
        raise ValueError(f"unsafe schema name: {schema!r}")
    return schema


_SCHEMA_DDL = """
CREATE SCHEMA IF NOT EXISTS {s};

CREATE TABLE IF NOT EXISTS {s}.book (
    id          serial PRIMARY KEY,
    title       text NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.word (
    id                serial PRIMARY KEY,
    lemma             text NOT NULL,
    lemma_lc          text GENERATED ALWAYS AS (lower(lemma)) STORED UNIQUE,
    as_seen           text,
    definition        text,
    part_of_speech    text,
    ipa               text,
    sentence          text,
    chapter           text,
    synonyms          text[] NOT NULL DEFAULT '{{}}',
    etymology         text,
    definition_source text,
    first_added       date,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.word_book (
    word_id  integer NOT NULL REFERENCES {s}.word(id) ON DELETE CASCADE,
    book_id  integer NOT NULL REFERENCES {s}.book(id) ON DELETE CASCADE,
    PRIMARY KEY (word_id, book_id)
);

CREATE TABLE IF NOT EXISTS {s}.category (
    id          serial PRIMARY KEY,
    taxonomy    text NOT NULL DEFAULT 'usas',
    code        text NOT NULL,
    name        text NOT NULL,
    parent_id   integer REFERENCES {s}.category(id) ON DELETE CASCADE,
    level       integer NOT NULL DEFAULT 0,
    assignable  boolean NOT NULL DEFAULT true,
    UNIQUE (taxonomy, code)
);

CREATE TABLE IF NOT EXISTS {s}.word_category (
    word_id     integer NOT NULL REFERENCES {s}.word(id) ON DELETE CASCADE,
    category_id integer NOT NULL REFERENCES {s}.category(id) ON DELETE CASCADE,
    confidence  real,
    source      text,          -- 'usas-tagger' | 'wordnet' | 'llm' | 'dict-label'
    is_primary  boolean NOT NULL DEFAULT false,
    PRIMARY KEY (word_id, category_id)
);

CREATE TABLE IF NOT EXISTS {s}.word_difficulty (
    word_id           integer PRIMARY KEY REFERENCES {s}.word(id) ON DELETE CASCADE,
    archaic             text,          -- current | dated | archaic | obsolete
    archaic_evidence    text,
    archaic_confidence  double precision,
    updated_at          timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.word_ngram (
    word_id        integer PRIMARY KEY REFERENCES {s}.word(id) ON DELETE CASCADE,
    peak           double precision,
    recent         double precision,
    recency_ratio  double precision,
    peak_year      integer,
    fetched_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.word_audio (
    word_id      integer PRIMARY KEY REFERENCES {s}.word(id) ON DELETE CASCADE,
    source       text,          -- 'commons' | 'azure' | 'none' (looked up, nothing found)
    file_path    text,
    ipa_used     text,          -- the exact phoneme string sent to the synthesizer (azure only)
    voice        text,          -- azure voice name, or the Commons source URL
    license_note text,
    generated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {s}.word_commons_search (
    word_id      integer PRIMARY KEY REFERENCES {s}.word(id) ON DELETE CASCADE,
    found_title  text,          -- Commons "File:..." title of an exact English match, or NULL
    download_url text,
    checked_at   timestamptz NOT NULL DEFAULT now()
);

-- One row per (book, lemma) rejection, deliberately NOT deduped across books
-- like word/word_book is: the same lemma can be rejected for different
-- reasons in different books (e.g. the coinage/UNSURE call depends on
-- per-book recurrence count), so each book's ingestion run keeps its own
-- verdict rather than merging into a single global history.
CREATE TABLE IF NOT EXISTS {s}.rejected_word (
    id          serial PRIMARY KEY,
    book_id     integer NOT NULL REFERENCES {s}.book(id) ON DELETE CASCADE,
    lemma       text NOT NULL,
    lemma_lc    text GENERATED ALWAYS AS (lower(lemma)) STORED,
    reason      text,          -- frequency_floor | proper_noun | misspelling | not_a_word | not_interesting
    detail      text,
    count       integer,
    zipf        double precision,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (book_id, lemma_lc)
);

CREATE INDEX IF NOT EXISTS rejected_word_lemma_idx ON {s}.rejected_word (lemma_lc);
"""


# pg_trgm powers future fuzzy "did-you-mean" lookups; optional because CREATE
# EXTENSION needs privileges a managed role may lack.
_TRGM_DDL = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS word_lemma_trgm ON {s}.word USING gin (lemma gin_trgm_ops);
"""


def connect(url: str | None = None) -> psycopg.Connection:
    resolved = database_url(url)
    if not resolved:
        raise RuntimeError("no DATABASE_URL set (env or .env)")
    return psycopg.connect(resolved)


def apply_schema(conn: psycopg.Connection, schema: str = DEFAULT_SCHEMA) -> bool:
    """Create schema/tables if absent. Returns True if the pg_trgm index was
    created (False if privileges didn't allow it — the rest still works)."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_DDL.format(s=s))
        # idempotent column additions (CREATE TABLE IF NOT EXISTS won't alter an
        # existing table, so evolve columns explicitly)
        cur.execute(f"ALTER TABLE {s}.book ADD COLUMN IF NOT EXISTS author text")
        # so "un-rejecting" a word in the review webapp can produce a word row
        # with the same context a normally-kept word has, not a bare stub
        cur.execute(f"ALTER TABLE {s}.rejected_word ADD COLUMN IF NOT EXISTS pos text")
        cur.execute(f"ALTER TABLE {s}.rejected_word ADD COLUMN IF NOT EXISTS as_seen text")
        cur.execute(f"ALTER TABLE {s}.rejected_word ADD COLUMN IF NOT EXISTS sentence text")
        cur.execute(f"ALTER TABLE {s}.rejected_word ADD COLUMN IF NOT EXISTS chapter text")
        cur.execute(f"ALTER TABLE {s}.word_difficulty "
                    "ADD COLUMN IF NOT EXISTS archaic_confidence double precision")
        cur.execute(f"ALTER TABLE {s}.word_difficulty "
                    "ADD COLUMN IF NOT EXISTS difficulty double precision")
        cur.execute(f"ALTER TABLE {s}.word_difficulty "
                    "ADD COLUMN IF NOT EXISTS difficulty_factors jsonb")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS quiz_definition text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS quiz_def_source text")
        cur.execute(f"ALTER TABLE {s}.word_difficulty ADD COLUMN IF NOT EXISTS quizzable boolean")
        cur.execute(f"ALTER TABLE {s}.word_difficulty ADD COLUMN IF NOT EXISTS quizzable_reason text")
        # Raw Wordnik pronunciation, stored separately from ipa: fetching is a slow
        # rate-limited pass (~1 word/6s observed), converting to IPA is fast and
        # iterable — keeping them apart means a converter fix never costs a re-fetch.
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS wordnik_pron_raw text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS wordnik_pron_type text")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS wordnik_checked_at timestamptz")
        # soft-delete flag for the review-and-prune web UI: pruned words stay in
        # place (history/audio/etc. intact) but drop out of every downstream view
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS active boolean NOT NULL DEFAULT true")
        cur.execute(f"CREATE INDEX IF NOT EXISTS word_active_idx ON {s}.word (active)")
        # tracks words the pipeline itself rejected but a human rescued via the
        # review webapp's Rejected tab — distinct from words the pipeline kept
        # on its own, so this history survives even though rejected_word
        # (which had the original reason/detail) is deleted once promoted
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS rescued_from_reject boolean NOT NULL DEFAULT false")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS rescued_at timestamptz")
        cur.execute(f"ALTER TABLE {s}.word ADD COLUMN IF NOT EXISTS rescued_reason text")
    trgm = True
    try:
        with conn.cursor() as cur:
            cur.execute(_TRGM_DDL.format(s=s))
    except psycopg.Error:
        conn.rollback()
        trgm = False
    conn.commit()
    return trgm


def _synonyms(cell: str) -> list[str]:
    return [x.strip() for x in (cell or "").split(";") if x.strip()]


def _books(cell: str) -> list[str]:
    return [x.strip() for x in (cell or "").split(";") if x.strip()]


def _read_master_rows(path: Path) -> list[dict]:
    """master_vocab.csv is tool-written with a full MASTER_COLUMNS header (it is not
    hand-edited in Excel like the per-book files), so a plain DictReader keeps every
    column — crucially date_added and source_book, which the vocab-only reader drops."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if (r.get("word") or "").strip()]


def sync_master(csv_path: Path, conn: psycopg.Connection,
                schema: str = DEFAULT_SCHEMA) -> dict:
    """Upsert every row of master_vocab.csv into the DB. Idempotent."""
    s = _safe_schema(schema)
    rows = _read_master_rows(Path(csv_path))
    stats = {"words": 0, "books": 0, "links": 0, "rows": len(rows)}
    seen_books: dict[str, int] = {}

    with conn.cursor() as cur:
        for r in rows:
            word = (r.get("word") or "").strip()
            if not word:
                continue
            cur.execute(
                f"""INSERT INTO {s}.word
                    (lemma, as_seen, definition, part_of_speech, ipa, sentence,
                     chapter, synonyms, etymology, definition_source, first_added)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NULLIF(%s,'')::date)
                    ON CONFLICT (lemma_lc) DO UPDATE SET
                        as_seen=EXCLUDED.as_seen, definition=EXCLUDED.definition,
                        part_of_speech=EXCLUDED.part_of_speech, ipa=EXCLUDED.ipa,
                        sentence=EXCLUDED.sentence, chapter=EXCLUDED.chapter,
                        synonyms=EXCLUDED.synonyms, etymology=EXCLUDED.etymology,
                        definition_source=EXCLUDED.definition_source,
                        first_added=LEAST(
                            {s}.word.first_added,
                            COALESCE(EXCLUDED.first_added, {s}.word.first_added)),
                        updated_at=now()
                    RETURNING id""",
                (word, r.get("as_seen"), r.get("definition"), normalize_pos(r.get("part_of_speech")),
                 r.get("ipa"), r.get("sentence"), r.get("chapter"), _synonyms(r.get("synonyms", "")),
                 r.get("etymology"), r.get("source"), (r.get("date_added") or "")),
            )
            word_id = cur.fetchone()[0]
            stats["words"] += 1

            for title in _books(r.get("source_book", "")):
                if title not in seen_books:
                    cur.execute(
                        f"""INSERT INTO {s}.book (title) VALUES (%s)
                            ON CONFLICT (title) DO UPDATE SET title=EXCLUDED.title
                            RETURNING id""", (title,))
                    seen_books[title] = cur.fetchone()[0]
                    stats["books"] += 1
                cur.execute(
                    f"""INSERT INTO {s}.word_book (word_id, book_id) VALUES (%s,%s)
                        ON CONFLICT DO NOTHING""", (word_id, seen_books[title]))
                if cur.rowcount:
                    stats["links"] += 1
    conn.commit()
    return stats


def sync_book_results(conn, book_title: str, kept: list, rejected: list,
                       schema: str = DEFAULT_SCHEMA, author: str | None = None) -> dict:
    """Upsert one book's ingestion results straight into Postgres — no CSV, no
    hand-edit, no `finalize`. KEEP/UNSURE candidates go into word/word_book
    exactly like sync_master; DROPped ones go into rejected_word, one row per
    (book, lemma). Review/pruning happens afterward in the review webapp
    (word.active) rather than before promotion. Idempotent: re-running the
    same book updates both tables in place. `author` is COALESCEd on conflict
    so re-ingesting a book without a parsed author never blanks a known one."""
    s = _safe_schema(schema)
    stats = {"kept": 0, "rejected": 0}

    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {s}.book (title, author) VALUES (%s, %s)
                ON CONFLICT (title) DO UPDATE SET title=EXCLUDED.title,
                    author=COALESCE(EXCLUDED.author, {s}.book.author)
                RETURNING id""", (book_title, author))
        book_id = cur.fetchone()[0]

        for c in kept:
            rep = c.representative
            cur.execute(
                f"""INSERT INTO {s}.word
                    (lemma, as_seen, definition, part_of_speech, ipa, sentence,
                     chapter, synonyms, etymology, definition_source, first_added)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, CURRENT_DATE)
                    ON CONFLICT (lemma_lc) DO UPDATE SET
                        as_seen=EXCLUDED.as_seen, definition=EXCLUDED.definition,
                        part_of_speech=EXCLUDED.part_of_speech, ipa=EXCLUDED.ipa,
                        sentence=EXCLUDED.sentence, chapter=EXCLUDED.chapter,
                        synonyms=EXCLUDED.synonyms, etymology=EXCLUDED.etymology,
                        definition_source=EXCLUDED.definition_source,
                        updated_at=now()
                    RETURNING id""",
                (c.lemma, rep.surface if rep else "", c.definition,
                 normalize_pos(c.part_of_speech or c.pos), c.ipa,
                 rep.sentence if rep else "", rep.chapter if rep else "",
                 list(c.synonyms), c.etymology,
                 c.definition_source or ", ".join(c.validity_sources)))
            word_id = cur.fetchone()[0]
            stats["kept"] += 1
            cur.execute(
                f"""INSERT INTO {s}.word_book (word_id, book_id) VALUES (%s,%s)
                    ON CONFLICT DO NOTHING""", (word_id, book_id))

        for c in rejected:
            rep = c.representative
            cur.execute(
                f"""INSERT INTO {s}.rejected_word
                    (book_id, lemma, reason, detail, count, zipf, pos, as_seen, sentence, chapter)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (book_id, lemma_lc) DO UPDATE SET
                        reason=EXCLUDED.reason, detail=EXCLUDED.detail,
                        count=EXCLUDED.count, zipf=EXCLUDED.zipf,
                        pos=EXCLUDED.pos, as_seen=EXCLUDED.as_seen,
                        sentence=EXCLUDED.sentence, chapter=EXCLUDED.chapter""",
                (book_id, c.lemma, c.reject_reason.value if c.reject_reason else None,
                 c.interesting_reason or None, c.count, c.zipf,
                 c.pos, rep.surface if rep else None,
                 rep.sentence if rep else None, rep.chapter if rep else None))
            stats["rejected"] += 1

    conn.commit()
    return stats


def fetch_pruned_lemmas(conn, schema: str = DEFAULT_SCHEMA) -> set[str]:
    """Lemmas a human has already manually pruned via the review webapp
    (word.active = false) — checked as the very first thing in ingestion so
    a word a human has already explicitly judged not worth keeping doesn't
    get silently re-spent on the (expensive) LLM judge every time it shows
    up in a new book, and doesn't have its existing row's definition/POS/
    etc. overwritten by whatever the new book's context happened to produce."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT lemma_lc FROM {s}.word WHERE NOT active")
        return {r[0] for r in cur.fetchall()}


def normalize_word_pos(conn, schema: str = DEFAULT_SCHEMA) -> dict:
    """Clean up word.part_of_speech in place: folds abbreviations/case variants
    (adj, adv, pron, adp, sconj, num, Noun, Adjective, ...) accumulated from
    older write paths down to the canonical vocabulary via normalize_pos().
    Idempotent — safe to re-run any time a new inconsistency creeps in."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, part_of_speech FROM {s}.word")
        rows = cur.fetchall()
        changed = 0
        for wid, pos in rows:
            new_pos = normalize_pos(pos)
            if new_pos != (pos or ""):
                cur.execute(f"UPDATE {s}.word SET part_of_speech = %s WHERE id = %s", (new_pos, wid))
                changed += 1
    conn.commit()
    return {"words": len(rows), "changed": changed}


def load_taxonomy(conn: psycopg.Connection, schema: str = DEFAULT_SCHEMA,
                  taxonomy: str = "usas") -> dict:
    """Upsert the USAS category tree into {schema}.category. Idempotent."""
    from . import usas
    s = _safe_schema(schema)
    cats = usas.categories()
    code_to_id: dict[str, int] = {}
    with conn.cursor() as cur:
        # pass 1: upsert nodes (parent set in pass 2 once every id is known)
        for c in cats:
            cur.execute(
                f"""INSERT INTO {s}.category (taxonomy, code, name, level, assignable)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT (taxonomy, code) DO UPDATE SET
                        name=EXCLUDED.name, level=EXCLUDED.level, assignable=EXCLUDED.assignable
                    RETURNING id""",
                (taxonomy, c["code"], c["name"], c["level"], c["assignable"]))
            code_to_id[c["code"]] = cur.fetchone()[0]
        # pass 2: wire parents
        for c in cats:
            pid = code_to_id.get(c["parent_code"]) if c["parent_code"] else None
            cur.execute(f"UPDATE {s}.category SET parent_id=%s WHERE id=%s",
                        (pid, code_to_id[c["code"]]))
    conn.commit()
    return {"categories": len(cats), "top_level": sum(1 for c in cats if c["parent_code"] is None)}


def compute_archaic(conn, schema: str = DEFAULT_SCHEMA) -> dict:
    """Set the archaic-currency ordinal on word_difficulty for every word. Uses the
    definition register-label + (if present) vocab.wiktionary is_archaic/is_obsolete."""
    from collections import Counter
    from . import archaic as _archaic
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute("select to_regclass('vocab.wiktionary')")
        have_wik = cur.fetchone()[0] is not None
    join = ("LEFT JOIN (select lower(term) t, bool_or(is_archaic) arc, bool_or(is_obsolete) obs "
            "from vocab.wiktionary group by lower(term)) k on k.t = lower(w.lemma)") if have_wik else ""
    cols = "coalesce(k.arc,false), coalesce(k.obs,false)" if have_wik else "false, false"
    dist: Counter = Counter()
    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, w.definition, {cols}, g.peak, g.recency_ratio
                        FROM {s}.word w {join}
                        LEFT JOIN {s}.word_ngram g ON g.word_id = w.id""")
        rows = cur.fetchall()
        for wid, defn, arc, obs, peak, ratio in rows:
            flag, evid, conf = _archaic.classify(defn, arc, obs, peak, ratio)
            dist[flag] += 1
            cur.execute(
                f"""INSERT INTO {s}.word_difficulty (word_id, archaic, archaic_evidence, archaic_confidence, updated_at)
                    VALUES (%s,%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET
                        archaic=EXCLUDED.archaic, archaic_evidence=EXCLUDED.archaic_evidence,
                        archaic_confidence=EXCLUDED.archaic_confidence, updated_at=now()""",
                (wid, flag, evid, conf))
    conn.commit()
    return dict(dist)


def fetch_ngrams(conn, schema: str = DEFAULT_SCHEMA, only_missing: bool = True,
                 limit: int = 0, delay: float = 0.3) -> dict:
    """Fetch + cache Google Books Ngram features for words. Returns counts."""
    import time
    from . import ngram
    s = _safe_schema(schema)
    where = (f" WHERE NOT EXISTS (SELECT 1 FROM {s}.word_ngram g WHERE g.word_id=w.id)"
             if only_missing else "")
    with conn.cursor() as cur:
        cur.execute(f"SELECT w.id, w.lemma FROM {s}.word w{where}" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (concordance vocab tool)"})
    stats = {"words": len(rows), "fetched": 0, "in_corpus": 0, "failed": 0}
    with conn.cursor() as cur:
        for wid, lemma in rows:
            f = ngram.fetch(lemma, session)
            if f is None:
                stats["failed"] += 1
                time.sleep(delay); continue
            if f["peak"]:
                stats["in_corpus"] += 1
            cur.execute(
                f"""INSERT INTO {s}.word_ngram (word_id, peak, recent, recency_ratio, peak_year, fetched_at)
                    VALUES (%s,%s,%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET peak=EXCLUDED.peak, recent=EXCLUDED.recent,
                        recency_ratio=EXCLUDED.recency_ratio, peak_year=EXCLUDED.peak_year, fetched_at=now()""",
                (wid, f["peak"], f["recent"], f["recency_ratio"], f["peak_year"]))
            stats["fetched"] += 1
            if stats["fetched"] % 200 == 0:
                conn.commit()
            time.sleep(delay)
    conn.commit()
    return stats


def compute_difficulty(conn, schema: str = DEFAULT_SCHEMA) -> dict:
    """Compute the ex-ante difficulty scalar (+ factor breakdown) for every word."""
    import statistics
    from psycopg.types.json import Json
    from . import difficulty as _diff
    from .validity_score import _morph_root, effective_zipf
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT w.id, w.lemma, g.peak, d.archaic, d.archaic_confidence, coalesce(dom.fields,'')
            FROM {s}.word w
            LEFT JOIN {s}.word_ngram g ON g.word_id = w.id
            LEFT JOIN {s}.word_difficulty d ON d.word_id = w.id
            LEFT JOIN (SELECT wc.word_id, string_agg(DISTINCT left(c.code,1), '') fields
                       FROM {s}.word_category wc JOIN {s}.category c ON c.id = wc.category_id
                       GROUP BY wc.word_id) dom ON dom.word_id = w.id""")
        rows = cur.fetchall()
        scores = []
        for wid, lemma, peak, archaic, aconf, fields in rows:
            zipf = effective_zipf(lemma)
            has_domain = any(f in _diff.DOMAIN_FIELDS for f in fields)
            morph = _morph_root(lemma) is not None
            sc, factors = _diff.score(zipf, peak, archaic or "current", aconf, has_domain, morph)
            scores.append(sc)
            cur.execute(
                f"""INSERT INTO {s}.word_difficulty (word_id, difficulty, difficulty_factors, updated_at)
                    VALUES (%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET
                        difficulty=EXCLUDED.difficulty, difficulty_factors=EXCLUDED.difficulty_factors,
                        updated_at=now()""",
                (wid, sc, Json(factors)))
    conn.commit()
    return {"words": len(scores),
            "mean": round(statistics.mean(scores), 1) if scores else 0,
            "median": statistics.median(scores) if scores else 0}


def compute_quiz_definitions(conn, schema: str = DEFAULT_SCHEMA, cfg=None,
                             only_missing: bool = True, limit: int = 0) -> dict:
    """Set quiz_definition/quiz_def_source. Clean defs pass through free; leakers are
    LLM-rewritten (validated) or redacted. Resumable via only_missing (scale-ready)."""
    from collections import Counter
    from . import quizdef
    s = _safe_schema(schema)
    where = "quiz_definition IS NULL AND " if only_missing else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma, definition FROM {s}.word "
                    f"WHERE {where}coalesce(definition,'') <> ''" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    clean = [(i, l, d) for i, l, d in rows if not quizdef.has_leak(l, d)]
    leakers = [(i, l, d) for i, l, d in rows if quizdef.has_leak(l, d)]
    stats = Counter()

    with conn.cursor() as cur:
        for wid, lemma, defn in clean:                       # free — no model
            cur.execute(f"UPDATE {s}.word SET quiz_definition=%s, quiz_def_source='clean' WHERE id=%s",
                        (defn, wid))
            stats["clean"] += 1
        conn.commit()

    if leakers:
        rw = quizdef.Rewriter(cfg)
        res = rw.rewrite([{"word": l, "definition": d} for _, l, d in leakers])
        with conn.cursor() as cur:
            for wid, lemma, defn in leakers:
                qd, src = res.get(lemma.lower(), (quizdef.redact(lemma, defn), "redacted"))
                cur.execute(f"UPDATE {s}.word SET quiz_definition=%s, quiz_def_source=%s WHERE id=%s",
                            (qd, src, wid))
                stats[src] += 1
        conn.commit()
    return {"words": len(rows), "clean": stats["clean"],
            "rewritten": stats["rewritten"], "redacted": stats["redacted"]}


def compute_quizzable(conn, schema: str = DEFAULT_SCHEMA) -> dict:
    """Set the quizzable flag (+ reason) on word_difficulty for every word."""
    from collections import Counter
    from wordfreq import zipf_frequency
    from . import quizdef
    from .validity_score import _morph_root
    s = _safe_schema(schema)
    dist: Counter = Counter()
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma, definition FROM {s}.word WHERE coalesce(definition,'') <> ''")
        rows = cur.fetchall()
        for wid, lemma, defn in rows:
            root = _morph_root(lemma)
            rz = zipf_frequency(root, "en") if root else None
            ok, reason = quizdef.quizzable(defn, root, rz)
            dist["quizzable" if ok else "excluded"] += 1
            cur.execute(
                f"""INSERT INTO {s}.word_difficulty (word_id, quizzable, quizzable_reason, updated_at)
                    VALUES (%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET
                        quizzable=EXCLUDED.quizzable, quizzable_reason=EXCLUDED.quizzable_reason, updated_at=now()""",
                (wid, ok, reason or None))
    conn.commit()
    return dict(dist)


def fetch_wordnik_pronunciations(conn, schema: str = DEFAULT_SCHEMA, only_missing: bool = True,
                                  limit: int = 0, delay: float = 0.1) -> dict:
    """Fetch RAW pronunciation strings from Wordnik (ahd-5 diacritic respelling,
    arpabet, or gcide-diacritical — whichever it has) and store them as-is, with
    no IPA conversion here. Rate-limited (~1 word per several seconds observed on
    the free tier) but that cost is paid once: wordnik_checked_at gates re-fetch,
    so converting to IPA later is a separate, fast, freely-iterable pass that never
    re-triggers this fetch."""
    import time
    from collections import Counter
    from . import deepdef
    s = _safe_schema(schema)
    key = deepdef.wordnik_key()
    if not key:
        return {"error": "no WORDNIK_API_KEY in .env"}

    where = f" WHERE wordnik_checked_at IS NULL" if only_missing else ""
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma FROM {s}.word{where}" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    import requests
    from .dictionary import _get
    session = requests.Session()
    dist: Counter = Counter()
    with conn.cursor() as cur:
        for i, (wid, lemma) in enumerate(rows, start=1):
            r = _get(session, f"https://api.wordnik.com/v4/word.json/{lemma}/pronunciations",
                     {"api_key": key, "limit": 5})
            raw, rtype = None, None
            if r is not None and r.status_code == 200:
                data = r.json()
                if data:
                    raw, rtype = data[0].get("raw"), data[0].get("rawType")
                    dist[rtype or "unknown"] += 1
            if raw is None:
                dist["none"] += 1
            cur.execute(f"UPDATE {s}.word SET wordnik_pron_raw=%s, wordnik_pron_type=%s, "
                        "wordnik_checked_at=now() WHERE id=%s", (raw, rtype, wid))
            if i % 25 == 0:
                conn.commit()
                print(f"  ...{i}/{len(rows)} checked")
            time.sleep(delay)
    conn.commit()
    return {"words": len(rows), **dist}


def search_commons_direct(conn, schema: str = DEFAULT_SCHEMA, dump_path: str | None = None,
                           only_missing: bool = True, limit: int = 0, delay: float = 2.5) -> dict:
    """Second-pass Commons search for words kaikki's dump reported no audio for
    (confirmed empirically to under-count: kaikki missed real, exact-match English
    recordings for words like "unpeople"/"enkindle"). Stores only the search
    result (title + constructed URL) — actually downloading is a separate,
    fast, freely-retriable step. Deliberately slow (Commons rate-limits hard);
    meant to run for hours unattended."""
    import time
    from collections import Counter
    from . import commons_search, wiktextract
    s = _safe_schema(schema)

    where = (f" WHERE NOT EXISTS (SELECT 1 FROM {s}.word_commons_search c WHERE c.word_id=w.id)"
             if only_missing else "")
    with conn.cursor() as cur:
        cur.execute(f"SELECT w.id, w.lemma FROM {s}.word w{where}" +
                    (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
    if not rows:
        return {"candidates": 0}

    # skip words kaikki already solved — only worth the slow search for real gaps
    lemmas = {lemma.strip().lower() for _, lemma in rows}
    dump_path = dump_path or wiktextract.DEFAULT_DUMP_PATH
    lexicon = wiktextract.build_lexicon(
        dump_path, lemmas, progress_cb=lambda n: print(f"  ...{n} lines scanned"))
    candidates = [(wid, lemma) for wid, lemma in rows
                  if not lexicon.get(lemma.strip().lower(), {}).get("audio")]

    dist: Counter = Counter(total=len(rows), skipped_kaikki_has_audio=len(rows) - len(candidates))
    session = requests.Session()
    with conn.cursor() as cur:
        for i, (wid, lemma) in enumerate(candidates, start=1):
            titles = commons_search.search_word(lemma, session)
            match = commons_search.best_english_exact_match(titles, lemma)
            url = commons_search.download_url(match) if match else None
            cur.execute(
                f"""INSERT INTO {s}.word_commons_search (word_id, found_title, download_url, checked_at)
                    VALUES (%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET found_title=EXCLUDED.found_title,
                        download_url=EXCLUDED.download_url, checked_at=now()""",
                (wid, match, url))
            dist["found"] += 1 if match else 0
            dist["not_found"] += 0 if match else 1
            if i % 20 == 0:
                conn.commit()
                print(f"  ...{i}/{len(candidates)} searched")
            time.sleep(delay)
        # words skipped because kaikki already has audio still need a checked_at
        # row so a re-run doesn't re-parse the dump for them pointlessly
        for wid, lemma in rows:
            if (wid, lemma) not in candidates:
                cur.execute(
                    f"""INSERT INTO {s}.word_commons_search (word_id, found_title, download_url, checked_at)
                        VALUES (%s, NULL, NULL, now()) ON CONFLICT (word_id) DO NOTHING""", (wid,))
    conn.commit()
    return dict(dist)


def compute_ipa(conn, schema: str = DEFAULT_SCHEMA, dump_path: str | None = None,
                 only_missing: bool = True, limit: int = 0) -> dict:
    """Backfill + clean word.ipa. Sources tried in order per word: (1) kaikki's
    Wiktextract dump; (2) Wordnik's raw pronunciation (already fetched by
    `wordnik-pron`), converted via the matching notation converter — direct
    IPA as-is, ARPAbet or AHD respellings through their own deterministic
    converters (gcide-diacritical has no converter yet, lowest yield, skipped).
    Also NULLs out any existing transcription that fails the English-language
    sanity check (the pre-existing ad hoc scrape occasionally grabbed a
    cross-referenced foreign cognate's IPA instead of the word's own — e.g.
    "murmurer" had the French verb's transcription). Idempotent: with
    only_missing=True (default), only words with an empty or invalid ipa are
    candidates, so a re-run after everything's resolved does no dump parsing
    at all and is a no-op."""
    from collections import Counter
    from . import ahd, arpabet, audio, wiktextract
    s = _safe_schema(schema)

    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma, ipa, wordnik_pron_raw, wordnik_pron_type FROM {s}.word" +
                    (f" LIMIT {int(limit)}" if limit else ""))
        all_rows = cur.fetchall()

    def is_valid(ipa):
        return bool(ipa) and audio.looks_like_english_ipa(ipa)

    candidates = all_rows if not only_missing else [r for r in all_rows if not is_valid(r[2])]
    dist: Counter = Counter(total=len(all_rows), already_valid=len(all_rows) - len(candidates))
    if not candidates:
        return dict(dist)

    lemmas = {lemma.strip().lower() for _, lemma, _, _, _ in candidates}
    dump_path = dump_path or wiktextract.DEFAULT_DUMP_PATH
    lexicon = wiktextract.build_lexicon(
        dump_path, lemmas, progress_cb=lambda n: print(f"  ...{n} lines scanned"))

    def wordnik_ipa(raw, rtype):
        if not raw:
            return None
        if rtype == "IPA":
            converted = raw
        elif rtype == "arpabet":
            converted = arpabet.to_ipa(raw)
        elif rtype == "ahd-5":
            converted = ahd.to_ipa(raw)
        else:
            return None  # gcide-diacritical: no converter yet
        return converted if converted and audio.looks_like_english_ipa(converted) else None

    with conn.cursor() as cur:
        for wid, lemma, existing_ipa, wn_raw, wn_type in candidates:
            had_valid_existing = is_valid(existing_ipa)
            entry = lexicon.get(lemma.strip().lower(), {})
            kaikki_ipa = wiktextract.best_ipa(entry.get("ipa", []))
            if kaikki_ipa and not audio.looks_like_english_ipa(kaikki_ipa):
                kaikki_ipa = None
            replacement = kaikki_ipa or wordnik_ipa(wn_raw, wn_type)
            source = "kaikki" if kaikki_ipa else ("wordnik" if replacement else None)

            if had_valid_existing and not replacement:
                dist["already_valid"] += 1  # nothing to do, no change
                continue
            if not (existing_ipa or "").strip() and replacement:
                cur.execute(f"UPDATE {s}.word SET ipa=%s WHERE id=%s", (replacement, wid))
                dist[f"backfilled_{source}"] += 1
            elif (existing_ipa or "").strip() and not had_valid_existing and replacement:
                cur.execute(f"UPDATE {s}.word SET ipa=%s WHERE id=%s", (replacement, wid))
                dist[f"corrected_{source}"] += 1
            elif (existing_ipa or "").strip() and not had_valid_existing:
                cur.execute(f"UPDATE {s}.word SET ipa=NULL WHERE id=%s", (wid,))
                dist["cleared_no_replacement"] += 1
            else:
                dist["unresolved"] += 1
    conn.commit()
    return dict(dist)


def download_commons_direct_finds(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0,
                                   delay: float = 4.0) -> dict:
    """Download the real recordings `commons-search` confirmed exist, upgrading
    any word currently on 'azure' or 'none' to the real recording. Split out
    from `compute_audio` because interleaving Commons downloads with fast Azure
    calls exhausted Commons' upload-CDN rate limit mid-run (429s that the
    per-request backoff wasn't patient enough for — this earlier in the session
    took over a minute to clear even at near-zero request volume). Paced like
    `commons-search` itself: slow, meant to run unattended."""
    import time
    from collections import Counter
    from . import audio
    s = _safe_schema(schema)

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT w.id, w.lemma, cs.download_url, a.source
            FROM {s}.word w
            JOIN {s}.word_commons_search cs ON cs.word_id = w.id
            LEFT JOIN {s}.word_audio a ON a.word_id = w.id
            WHERE cs.found_title IS NOT NULL AND (a.source IS NULL OR a.source <> 'commons')
        """ + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    dist: Counter = Counter(candidates=len(rows))
    if not rows:
        return dict(dist)
    audio.AUDIO_DIR.mkdir(exist_ok=True)

    with conn.cursor() as cur:
        for i, (wid, lemma, url, prior_source) in enumerate(rows, start=1):
            lemma_lc = lemma.strip().lower()
            dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
            if audio.fetch_commons_audio(url, dest, tries=6):
                cur.execute(
                    f"""INSERT INTO {s}.word_audio (word_id, source, file_path, ipa_used, voice, license_note, generated_at)
                        VALUES (%s,'commons',%s,NULL,%s,%s, now())
                        ON CONFLICT (word_id) DO UPDATE SET source='commons', file_path=EXCLUDED.file_path,
                            ipa_used=NULL, voice=EXCLUDED.voice, license_note=EXCLUDED.license_note, generated_at=now()""",
                    (wid, str(dest), url,
                     "Wikimedia Commons recording (direct search — kaikki's dump missed it); "
                     "verify per-file license before public reuse"))
                dist["downloaded"] += 1
                dist[f"upgraded_from_{prior_source}"] += 1 if prior_source else 0
            else:
                dist["failed"] += 1
            if i % 20 == 0:
                conn.commit()
                print(f"  ...{i}/{len(rows)} downloaded")
            time.sleep(delay)
    conn.commit()
    return dict(dist)


def compute_audio(conn, schema: str = DEFAULT_SCHEMA, dump_path: str | None = None,
                   only_missing: bool = True, limit: int = 0, delay: float = 0.3) -> dict:
    """Fill in word_audio: real Commons recordings where kaikki/Wiktextract has
    one, else a real recording the direct Commons search found that kaikki
    missed, else Azure IPA-guided synthesis where a transcription is known
    (ours, kaikki's, or Wordnik's — backfilling word.ipa along the way), else
    a 'none' placeholder so re-runs don't keep re-parsing the dump for words
    with nothing to find."""
    import time
    from collections import Counter
    from . import audio, wiktextract
    s = _safe_schema(schema)

    where = (f" WHERE NOT EXISTS (SELECT 1 FROM {s}.word_audio a WHERE a.word_id=w.id)"
             if only_missing else "")
    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, w.lemma, w.ipa, cs.download_url
                        FROM {s}.word w
                        LEFT JOIN {s}.word_commons_search cs ON cs.word_id = w.id{where}""" +
                    (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    dist: Counter = Counter()
    if not rows:
        return {"candidates": 0, **dist}

    lemmas = {lemma.strip().lower() for _, lemma, _, _ in rows}
    dump_path = dump_path or wiktextract.DEFAULT_DUMP_PATH
    lexicon = wiktextract.build_lexicon(
        dump_path, lemmas, progress_cb=lambda n: print(f"  ...{n} lines scanned"))

    key, region = audio.azure_credentials()
    if not (key and region):
        print("  (no AZURE_SPEECH_KEY/AZURE_SPEECH_REGION in .env — skipping synthesis, Commons-only pass)")
    audio.AUDIO_DIR.mkdir(exist_ok=True)

    with conn.cursor() as cur:
        for i, (wid, lemma, existing_ipa, direct_search_url) in enumerate(rows, start=1):
            lemma_lc = lemma.strip().lower()
            entry = lexicon.get(lemma_lc, {})

            existing_ipa = existing_ipa if audio.looks_like_english_ipa(existing_ipa or "") else None

            kaikki_ipa = wiktextract.best_ipa(entry.get("ipa", []))
            if kaikki_ipa and not audio.looks_like_english_ipa(kaikki_ipa):
                kaikki_ipa = None
            if kaikki_ipa and not (existing_ipa or "").strip():
                cur.execute(f"UPDATE {s}.word SET ipa=%s WHERE id=%s", (kaikki_ipa, wid))
                existing_ipa = kaikki_ipa

            # tries=2 (not fetch_commons_audio's default 4-6): this loop needs to
            # move fast through many candidates and has Azure as a good fallback.
            # A sustained Commons rate-limit block turned a handful of slow
            # downloads into an hours-long stall here — `commons-download` is the
            # dedicated, patient (tries=6) pass for real recovery, run separately.
            best_recording = wiktextract.best_audio(entry.get("audio", []))
            row = None
            if best_recording:
                dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                if audio.fetch_commons_audio(best_recording["url"], dest, tries=1):
                    row = ("commons", str(dest), None, best_recording["url"],
                           "Wikimedia Commons recording; verify per-file license before public reuse")
                    dist["commons"] += 1
            if row is None and direct_search_url:
                dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                if audio.fetch_commons_audio(direct_search_url, dest, tries=1):
                    row = ("commons", str(dest), None, direct_search_url,
                           "Wikimedia Commons recording (direct search — kaikki's dump missed it); "
                           "verify per-file license before public reuse")
                    dist["commons_direct_search"] += 1
            if row is None and (existing_ipa or "").strip() and key and region:
                clip = audio.synthesize_azure(lemma, existing_ipa, key, region)
                if clip:
                    dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                    dest.write_bytes(clip)
                    row = ("azure", str(dest), audio.normalize_ipa(existing_ipa), audio.AZURE_VOICE, None)
                    dist["azure"] += 1
            if row is None:
                row = ("none", None, None, None, None)
                dist["none"] += 1

            cur.execute(
                f"""INSERT INTO {s}.word_audio (word_id, source, file_path, ipa_used, voice, license_note, generated_at)
                    VALUES (%s,%s,%s,%s,%s,%s, now())
                    ON CONFLICT (word_id) DO UPDATE SET source=EXCLUDED.source, file_path=EXCLUDED.file_path,
                        ipa_used=EXCLUDED.ipa_used, voice=EXCLUDED.voice, license_note=EXCLUDED.license_note,
                        generated_at=now()""",
                (wid, *row))
            if i % 50 == 0:
                conn.commit()
                print(f"  ...{i}/{len(rows)} words processed")
            time.sleep(delay)
    conn.commit()
    return {"candidates": len(rows), **dist}


def synthesize_unverified_guesses(conn, schema: str = DEFAULT_SCHEMA, limit: int = 0,
                                   delay: float = 0.3) -> dict:
    """Last resort for words with no real recording and no IPA anywhere: Azure
    guesses pronunciation from spelling alone, same as any TTS would. Recorded
    with source='azure_guess' — deliberately distinct from 'azure' (IPA-guided)
    so the quiz app can flag these as unverified rather than presenting a guess
    with the same confidence as a verified pronunciation."""
    import time
    from collections import Counter
    from . import audio
    s = _safe_schema(schema)

    key, region = audio.azure_credentials()
    if not (key and region):
        return {"error": "no AZURE_SPEECH_KEY/AZURE_SPEECH_REGION in .env"}

    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, w.lemma FROM {s}.word w
                        JOIN {s}.word_audio a ON a.word_id = w.id
                        WHERE a.source = 'none'""" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()

    dist: Counter = Counter(candidates=len(rows))
    if not rows:
        return dict(dist)
    audio.AUDIO_DIR.mkdir(exist_ok=True)

    with conn.cursor() as cur:
        for i, (wid, lemma) in enumerate(rows, start=1):
            lemma_lc = lemma.strip().lower()
            clip = audio.synthesize_azure_guess(lemma, key, region)
            if clip:
                dest = audio.AUDIO_DIR / f"{lemma_lc}.mp3"
                dest.write_bytes(clip)
                cur.execute(
                    f"""UPDATE {s}.word_audio SET source='azure_guess', file_path=%s, ipa_used=NULL,
                        voice=%s, license_note='unverified: no IPA available, Azure guessed from spelling',
                        generated_at=now() WHERE word_id=%s""",
                    (str(dest), audio.AZURE_VOICE, wid))
                dist["synthesized"] += 1
            else:
                dist["failed"] += 1
            if i % 50 == 0:
                conn.commit()
                print(f"  ...{i}/{len(rows)} words processed")
            time.sleep(delay)
    conn.commit()
    return dict(dist)
