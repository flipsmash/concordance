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
