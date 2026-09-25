"""Reference data loaded into Postgres: taxonomy, gazetteer, Google Ngram, and Wiktionary
language/English-reference lookups."""

from __future__ import annotations

import json
import os
import re

import psycopg
import requests

from .core import DEFAULT_SCHEMA, _safe_schema


def load_taxonomy(conn: psycopg.Connection, schema: str = DEFAULT_SCHEMA,
                  taxonomy: str = "usas") -> dict:
    """Upsert the USAS category tree into {schema}.category. Idempotent."""
    from .. import usas
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


def load_gazetteer(conn, schema: str = DEFAULT_SCHEMA, *,
                   census_path=None, geonames_path=None) -> dict:
    """Bulk-load the names/places gazetteer into {schema}.gazetteer_name --
    see concordance/gazetteer.py for sourcing. Idempotent (truncate + reload
    per kind, not an incremental upsert -- the source files are occasional,
    whole-file re-downloads, not something to merge row by row) and safe to
    re-run after a fresh download to pick up an updated source file. COPY,
    not executemany -- this is ~300k rows (162k surnames + 140k places + 8k
    given names), the same bulk-load-scale reasoning as every other
    multi-hundred-thousand-row load in this file."""
    from .. import gazetteer as _gaz
    s = _safe_schema(schema)

    sources: list[tuple[str, set[str]]] = [
        ("given_name", _gaz.load_given_names()),
        ("surname", _gaz.load_surnames(census_path) if census_path else _gaz.load_surnames()),
        ("place", _gaz.load_places(geonames_path) if geonames_path else _gaz.load_places()),
    ]

    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        for kind, names in sources:
            cur.execute(f"DELETE FROM {s}.gazetteer_name WHERE kind = %s", (kind,))
            with cur.copy(f"COPY {s}.gazetteer_name (name_lc, kind) FROM STDIN") as copy:
                for name in names:
                    copy.write_row((name, kind))
            counts[kind] = len(names)
    conn.commit()
    return counts


def fetch_gazetteer_names(conn, schema: str = DEFAULT_SCHEMA) -> frozenset[str]:
    """Every name/place in {schema}.gazetteer_name -- a plain set (the
    validity gate's own check only ever needs membership, not which kind).
    Degrades to empty (same convention as vocab.wiktionary elsewhere in
    this file) if `load-gazetteer` was never run -- a fresh checkout must
    still be able to ingest without this optional table existing yet."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute("select to_regclass(%s)", (f"{s}.gazetteer_name",))
        if cur.fetchone()[0] is None:
            return frozenset()
        cur.execute(f"SELECT name_lc FROM {s}.gazetteer_name")
        return frozenset(r[0] for r in cur.fetchall())


def load_ngram_bulk(conn, parts: list, totals: dict[int, int], ngram_schema: str = "ngram") -> dict:
    """Load ngram_bulk.build_tsv's parts into the standalone `ngram` schema
    (NOT part of apply_schema -- web restarts run that, and a multi-GB load
    must never hang behind one): staged COPY into ngram.unigram_new, primary
    key built after the load, then an atomic swap. Also refreshes
    ngram.decade_total (the per-decade denominators for decade_counts) and
    ngram.meta. Idempotent: re-running replaces the previous load."""
    from .. import ngram_bulk as nb
    g = _safe_schema(ngram_schema)
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {g}")
        cur.execute(f"DROP TABLE IF EXISTS {g}.unigram_new")
        cur.execute(f"""CREATE UNLOGGED TABLE {g}.unigram_new (
                           term text NOT NULL, peak double precision, recent double precision,
                           recency_ratio double precision, peak_year smallint, decade_counts bigint[])""")
        n = 0
        for part in parts:
            with open(part, "rb") as fh, cur.copy(
                    f"COPY {g}.unigram_new (term, peak, recent, recency_ratio, peak_year, decade_counts) "
                    "FROM STDIN WITH (FORMAT text, NULL '')") as copy:
                while chunk := fh.read(1 << 20):
                    copy.write(chunk)
            conn.commit()
        cur.execute(f"SELECT count(*), count(DISTINCT term) FROM {g}.unigram_new")
        n, distinct = cur.fetchone()
        if n != distinct:
            raise RuntimeError(f"ngram bulk load has duplicate terms ({n} rows, {distinct} distinct)")
        cur.execute(f"ALTER TABLE {g}.unigram_new SET LOGGED")
        cur.execute(f"ALTER TABLE {g}.unigram_new ADD PRIMARY KEY (term)")
        cur.execute(f"DROP TABLE IF EXISTS {g}.decade_total")
        cur.execute(f"CREATE TABLE {g}.decade_total (decade smallint PRIMARY KEY, match_count bigint NOT NULL)")
        for d in nb.DECADES:
            cur.execute(f"INSERT INTO {g}.decade_total VALUES (%s, %s)",
                        (d, sum(totals.get(y, 0) for y in range(d, d + 10))))
        cur.execute(f"""CREATE TABLE IF NOT EXISTS {g}.meta (
                           dataset text NOT NULL, terms bigint NOT NULL, loaded_at timestamptz NOT NULL)""")
        cur.execute(f"DELETE FROM {g}.meta")
        cur.execute(f"INSERT INTO {g}.meta VALUES (%s, %s, now())", (nb.BASE_URL, n))
        cur.execute(f"DROP TABLE IF EXISTS {g}.unigram")
        cur.execute(f"ALTER TABLE {g}.unigram_new RENAME TO unigram")
        cur.execute(f"ALTER INDEX {g}.unigram_new_pkey RENAME TO unigram_pkey")
    conn.commit()
    return {"terms": n}


def fetch_ngrams(conn, schema: str = DEFAULT_SCHEMA, only_missing: bool = True,
                 limit: int = 0, delay: float = 0.3, ngram_schema: str = "ngram") -> dict:
    """Fill word_ngram. Bulk first, then the live API only for what bulk can't
    answer:
      - bulk:        lemma found in ngram.unigram (the local v3 dataset --
                     see ngram_bulk.py / load_ngram_bulk) -- one SQL join;
      - bulk_absent: a plain lowercase lemma (^[a-z]+$) NOT in ngram.unigram
                     is genuinely absent from print -- the bulk table holds
                     every lowercase term Google counted -- so zeros, the
                     same "not in corpus" row the API returns;
      - api:         any other shape (hyphen, space, capital, accent) missing
                     from bulk, whose tokenization bulk 1-grams may not share.
    Falls back to API-only when the ngram schema hasn't been loaded."""
    import time
    from .. import ngram
    s = _safe_schema(schema)
    g = _safe_schema(ngram_schema)
    missing = (f" AND NOT EXISTS (SELECT 1 FROM {s}.word_ngram g WHERE g.word_id=w.id)"
               if only_missing else "")
    stats = {"words": 0, "bulk": 0, "bulk_absent": 0, "fetched": 0, "in_corpus": 0, "failed": 0}
    upsert = f"""INSERT INTO {s}.word_ngram (word_id, peak, recent, recency_ratio, peak_year, fetched_at)
                 {{select}}
                 ON CONFLICT (word_id) DO UPDATE SET peak=EXCLUDED.peak, recent=EXCLUDED.recent,
                     recency_ratio=EXCLUDED.recency_ratio, peak_year=EXCLUDED.peak_year,
                     fetched_at=EXCLUDED.fetched_at"""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{g}.unigram",))
        have_bulk = cur.fetchone()[0] is not None
        if have_bulk and not limit:
            cur.execute(upsert.format(select=f"""
                SELECT w.id, u.peak, u.recent, u.recency_ratio, u.peak_year, now()
                FROM {s}.word w JOIN {g}.unigram u ON u.term = w.lemma
                WHERE true{missing}"""))
            stats["bulk"] = cur.rowcount
            cur.execute(upsert.format(select=f"""
                SELECT w.id, 0, 0, NULL, NULL, now() FROM {s}.word w
                WHERE w.lemma ~ '^[a-z]+$'
                  AND NOT EXISTS (SELECT 1 FROM {g}.unigram u WHERE u.term = w.lemma){missing}"""))
            stats["bulk_absent"] = cur.rowcount
            conn.commit()
            stats["in_corpus"] = stats["bulk"]
        api_filter = ("" if not (have_bulk and not limit) else
                      " AND NOT (w.lemma ~ '^[a-z]+$') AND NOT EXISTS "
                      f"(SELECT 1 FROM {g}.unigram u WHERE u.term = w.lemma)")
        # With bulk loaded, the API tier only ever fills rows that don't exist
        # yet, even on a refetch: the dataset is the same frozen en-2019
        # corpus the API serves, so re-asking it (rate-limited) buys nothing.
        api_missing = (f" AND NOT EXISTS (SELECT 1 FROM {s}.word_ngram g WHERE g.word_id=w.id)"
                       if have_bulk else missing)
        cur.execute(f"SELECT w.id, w.lemma FROM {s}.word w WHERE true{api_missing}{api_filter}"
                    + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
    stats["words"] = stats["bulk"] + stats["bulk_absent"] + len(rows)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (concordance vocab tool)"})
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
                print(f"  ...{stats['fetched']}/{len(rows)} fetched ({stats['in_corpus']} in corpus, "
                      f"{stats['failed']} failed)")
            time.sleep(delay)
    conn.commit()
    return stats


# Wiktionary language names that count as English for foreign-word
# detection (historical/regional English and scientific Translingual entries),
# and the classical families Brian asked to leave alone.
_ENGLISH_FAMILY_LANGS = ("English", "Middle English", "Old English", "Early Modern English",
                         "Scots", "Yola", "Fingallian", "Translingual")


def load_wiktionary_langs(conn, dump_path: str | None = None, wikt_schema: str = "wikt") -> dict:
    """`concordance wiktionary-langs`: from the English Wiktionary dump (every
    language's entries, each tagged with its language), build
    <wikt_schema>.foreign_term -- terms (lowercased) that have entries ONLY in
    languages outside English-family/Translingual and outside Latin/Greek,
    with those languages -- and <wikt_schema>.historic_term, terms that are
    ONLY Middle/Old English (context_lang's Middle English markers). Standalone schema, not apply_schema (a large load).
    A byte-level grep pulls each entry's top-level "word"/"lang" pair (always
    adjacent, in that order) so the multi-GB JSON never needs parsing."""
    import subprocess
    from .. import wiktextract
    g = _safe_schema(wikt_schema)
    dump = dump_path or wiktextract.DEFAULT_DUMP_PATH
    pattern = r'"word": "(?:[^"\\]|\\.)*", "lang": "(?:[^"\\]|\\.)*", "lang_code": "[^"]*"'
    zcat = subprocess.Popen(["zcat", dump], stdout=subprocess.PIPE)
    grep = subprocess.Popen(["grep", "-aoP", pattern], stdin=zcat.stdout, stdout=subprocess.PIPE,
                            env={**os.environ, "LC_ALL": "C"})
    zcat.stdout.close()
    pair_re = re.compile(r'^"word": "(.*)", "lang": "(.*)", "lang_code": "[^"]*"$')
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {g}")
        cur.execute(f"DROP TABLE IF EXISTS {g}.term_lang_raw")
        cur.execute(f"CREATE UNLOGGED TABLE {g}.term_lang_raw (term text NOT NULL, lang text NOT NULL)")
        n = 0
        with cur.copy(f"COPY {g}.term_lang_raw (term, lang) FROM STDIN") as copy:
            for raw in grep.stdout:
                m = pair_re.match(raw.decode("utf-8", "replace").rstrip("\n"))
                if m:
                    copy.write_row((json.loads(f'"{m.group(1)}"').lower(), json.loads(f'"{m.group(2)}"')))
                    n += 1
        grep.wait(); zcat.wait()
        cur.execute(f"DROP TABLE IF EXISTS {g}.foreign_term")
        cur.execute(f"""CREATE TABLE {g}.foreign_term AS
                        SELECT term, array_agg(DISTINCT lang ORDER BY lang) AS langs
                        FROM {g}.term_lang_raw
                        GROUP BY term
                        HAVING NOT bool_or(lang = ANY(%s) OR lang ~ '(Latin|Greek)')""",
                    (list(_ENGLISH_FAMILY_LANGS),))
        cur.execute(f"ALTER TABLE {g}.foreign_term ADD PRIMARY KEY (term)")
        cur.execute(f"SELECT count(*) FROM {g}.foreign_term")
        foreign = cur.fetchone()[0]
        # Middle/Old English markers for context_lang: entries ONLY in those
        # languages -- anything also modern English, Scots or Translingual
        # would mark ordinary archaic or dialect prose as medieval.
        cur.execute(f"DROP TABLE IF EXISTS {g}.historic_term")
        cur.execute(f"""CREATE TABLE {g}.historic_term AS
                        SELECT term FROM {g}.term_lang_raw
                        GROUP BY term
                        HAVING bool_or(lang IN ('Middle English', 'Old English'))
                           AND NOT bool_or(lang IN ('English', 'Scots', 'Translingual',
                                                    'Early Modern English'))""")
        cur.execute(f"ALTER TABLE {g}.historic_term ADD PRIMARY KEY (term)")
        cur.execute(f"SELECT count(*) FROM {g}.historic_term")
        historic = cur.fetchone()[0]
        cur.execute(f"DROP TABLE {g}.term_lang_raw")
    conn.commit()
    return {"pairs": n, "foreign_only_terms": foreign, "historic_terms": historic}


def foreign_only_langs(conn, lemmas, wikt_schema: str = "wikt") -> dict[str, list[str]]:
    """lemma -> its Wiktionary languages, for the lemmas that are foreign-only
    per <wikt_schema>.foreign_term AND absent from the local English
    Wiktionary (vocab.wiktionary) and 0 Dict (oed.entry) headwords. Empty if
    the table hasn't been built. The per-word English-usage checks live in
    validity_score.foreign_cast_out_reason."""
    g = _safe_schema(wikt_schema)
    lemmas = sorted({l.lower() for l in lemmas})
    if not lemmas:
        return {}
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s), to_regclass('vocab.wiktionary'), to_regclass('oed.entry')",
                    (f"{g}.foreign_term",))
        have_ft, have_wik, have_oed = cur.fetchone()
        if not have_ft:
            return {}
        cur.execute(
            f"""SELECT f.term, f.langs FROM {g}.foreign_term f
                WHERE f.term = ANY(%s)"""
            + (" AND NOT EXISTS (SELECT 1 FROM vocab.wiktionary v WHERE v.term IN (f.term, initcap(f.term)))"
               if have_wik else "")
            + (" AND NOT EXISTS (SELECT 1 FROM oed.entry o WHERE o.headword_norm IN (f.term, initcap(f.term)))"
               if have_oed else ""),
            (lemmas,))
        return {term: langs for term, langs in cur.fetchall()}


def english_reference_terms(conn, lemmas, *, exclude_misspelling_glosses: bool = False) -> set[str]:
    """The lemmas (lowercased) with an entry in the local English Wiktionary
    (vocab.wiktionary) or 0 Dict (oed.entry) -- one query for the batch.
    exclude_misspelling_glosses: a Wiktionary entry that only says
    "misspelling of X" doesn't count (Wiktionary lists common typos)."""
    lemmas = sorted({l.lower() for l in lemmas})
    if not lemmas:
        return set()
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('vocab.wiktionary'), to_regclass('oed.entry')")
        have_wik, have_oed = cur.fetchone()
        found: set[str] = set()
        if have_wik:
            cur.execute("SELECT lower(term) FROM vocab.wiktionary WHERE (term = ANY(%s) OR term = ANY(%s))"
                        + (" AND definition !~* '(misspelling|misspelt|typo(graphical)? error) (of|for)'"
                           if exclude_misspelling_glosses else ""),
                        (lemmas, [l.capitalize() for l in lemmas]))
            found |= {r[0] for r in cur.fetchall()}
        if have_oed:
            cur.execute("SELECT lower(headword_norm) FROM oed.entry WHERE headword_norm = ANY(%s) "
                        "OR headword_norm = ANY(%s)", (lemmas, [l.capitalize() for l in lemmas]))
            found |= {r[0] for r in cur.fetchall()}
    return found & set(lemmas)
