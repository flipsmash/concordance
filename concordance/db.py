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
    with path.open(newline="", encoding="utf-8") as f:
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
                (word, r.get("as_seen"), r.get("definition"), r.get("part_of_speech"),
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
    from wordfreq import zipf_frequency
    from . import difficulty as _diff
    from .validity_score import _morph_root
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
            zipf = zipf_frequency(lemma, "en")
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
