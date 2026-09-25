"""kaikki/Wiktextract dump lookup (§ audio pronunciation).

The current ad hoc IPA scrape (dictionary.py, one Wiktionary REST call per word)
tops out around 50% coverage and has no access to real recorded pronunciation
audio. kaikki.org publishes a structured, offline JSONL dump of the entire English
Wiktionary (`wiktextract`) with per-word `sounds` entries carrying both IPA
(dialect-tagged) and Wikimedia Commons audio URLs when a human recording exists.
One local pass over the dump answers both questions at once, for free, with no
per-word API calls or rate limits -- and `concordance wiktextract-sounds` makes that pass
once, into Postgres (see load_sounds / sound_lexicon below).

Download once (~2.6GB compressed):
    curl -o data/wiktextract-en.jsonl.gz https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz

The dump is multilingual (English Wiktionary describes words from every language
it has entries for) — filtered here to lang_code == "en".
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

DEFAULT_DUMP_PATH = "data/wiktextract-en.jsonl.gz"


def build_lexicon(dump_path: str | Path, lemmas: set[str],
                   progress_cb=None) -> dict[str, dict]:
    """Stream the dump once, returning lemma_lc -> {"ipa": [...], "audio": [...]}
    for every requested lemma that has sound data. `lemmas` must already be
    lowercased. `progress_cb(lines_scanned)` is called periodically if given.
    """
    path = Path(dump_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Wiktextract dump not found at {path}. Download it with:\n"
            f"  curl -o {path} https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
        )

    found: dict[str, dict] = {}
    n_lines = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if progress_cb and n_lines % 2_000_000 == 0:
                progress_cb(n_lines)
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("lang_code") != "en":
                continue
            word = (obj.get("word") or "").strip().lower()
            if word not in lemmas:
                continue
            ipas, audios = _sounds_of(obj)
            if ipas or audios:
                entry = found.setdefault(word, {"ipa": [], "audio": []})
                entry["ipa"].extend(ipas)
                entry["audio"].extend(audios)
    return found


def best_ipa(entries: list[dict]) -> str | None:
    """Prefer a US-tagged transcription (matches the en-US synthesis voice);
    fall back to the first available."""
    if not entries:
        return None
    us = [e for e in entries if "US" in e.get("tags", [])]
    return (us[0] if us else entries[0])["ipa"]


def best_audio(entries: list[dict]) -> dict | None:
    if not entries:
        return None
    us = [e for e in entries if "US" in e.get("tags", [])]
    return us[0] if us else entries[0]


# --- the same data, loaded once into Postgres ---------------------------------
#
# Scanning the 2.7 GB dump costs ~10 minutes per call, and `ipa`, `audio` and
# `commons-direct` each made that call on every run. `wiktextract-sounds`
# loads every English entry's sounds into <wikt_schema>.sound once (rerun it
# after downloading a newer dump); sound_lexicon() reads that table when it
# exists and only falls back to scanning the dump when it doesn't.

def _sound_rows(dump_path: str | Path, progress_cb=None):
    """(term, ipas, audios) per English dump entry with sound data, in dump
    order -- the same per-entry extraction build_lexicon does."""
    path = Path(dump_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Wiktextract dump not found at {path}. Download it with:\n"
            f"  curl -o {path} https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
        )
    n_lines = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if progress_cb and n_lines % 2_000_000 == 0:
                progress_cb(n_lines)
            if '"sounds"' not in line:          # cheap superset filter before json.loads
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("lang_code") != "en":
                continue
            word = (obj.get("word") or "").strip().lower()
            ipas, audios = _sounds_of(obj)
            if word and (ipas or audios):
                yield word, ipas, audios


def _sounds_of(obj: dict) -> tuple[list[dict], list[dict]]:
    ipas, audios = [], []
    for s in obj.get("sounds") or []:
        if s.get("ipa"):
            ipas.append({"ipa": s["ipa"], "tags": s.get("tags", [])})
        url = s.get("ogg_url") or s.get("mp3_url")
        if url:
            audios.append({"url": url, "tags": s.get("tags", [])})
    return ipas, audios


def load_sounds(conn, dump_path: str | Path | None = None, wikt_schema: str = "wikt",
                progress_cb=None) -> dict:
    """`concordance wiktextract-sounds`: rebuild <wikt_schema>.sound (term ->
    ipa/audio lists, concatenated across the term's entries in dump order,
    exactly as build_lexicon returns them). Built under a new name and swapped
    in, so readers never see the table missing."""
    from psycopg.types.json import Jsonb

    from .db import _safe_schema
    g = _safe_schema(wikt_schema)
    n = 0
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {g}")
        cur.execute(f"DROP TABLE IF EXISTS {g}.sound_raw")
        cur.execute(f"""CREATE UNLOGGED TABLE {g}.sound_raw (
                            seq bigint NOT NULL, term text NOT NULL, ipa jsonb NOT NULL, audio jsonb NOT NULL)""")
        with cur.copy(f"COPY {g}.sound_raw (seq, term, ipa, audio) FROM STDIN") as copy:
            for n, (term, ipas, audios) in enumerate(
                    _sound_rows(dump_path or DEFAULT_DUMP_PATH, progress_cb), start=1):
                copy.write_row((n, term, Jsonb(ipas), Jsonb(audios)))
        cur.execute(f"DROP TABLE IF EXISTS {g}.sound_new")
        cur.execute(f"""CREATE TABLE {g}.sound_new AS
                        WITH i AS (SELECT term, jsonb_agg(e ORDER BY seq, o) AS ipa
                                   FROM {g}.sound_raw, jsonb_array_elements(ipa) WITH ORDINALITY AS x(e, o)
                                   GROUP BY term),
                             a AS (SELECT term, jsonb_agg(e ORDER BY seq, o) AS audio
                                   FROM {g}.sound_raw, jsonb_array_elements(audio) WITH ORDINALITY AS x(e, o)
                                   GROUP BY term)
                        SELECT term, coalesce(i.ipa, '[]') AS ipa, coalesce(a.audio, '[]') AS audio
                        FROM i FULL JOIN a USING (term)""")
        cur.execute(f"ALTER TABLE {g}.sound_new ADD PRIMARY KEY (term)")
        cur.execute(f"DROP TABLE IF EXISTS {g}.sound")
        cur.execute(f"ALTER TABLE {g}.sound_new RENAME TO sound")
        cur.execute(f"ALTER INDEX {g}.sound_new_pkey RENAME TO sound_pkey")
        cur.execute(f"DROP TABLE {g}.sound_raw")
        cur.execute(f"SELECT count(*) FROM {g}.sound")
        terms = cur.fetchone()[0]
    conn.commit()
    return {"entries": n, "terms": terms}


def sound_lexicon(conn, lemmas: set[str], dump_path: str | Path | None = None,
                  wikt_schema: str = "wikt", progress_cb=None) -> dict[str, dict]:
    """build_lexicon's result for `lemmas` (lowercased), from <wikt_schema>.sound
    when it has been loaded; an explicit `dump_path`, or no table, scans the
    dump instead."""
    from .db import _safe_schema
    g = _safe_schema(wikt_schema)
    if dump_path is None:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"{g}.sound",))
            if cur.fetchone()[0] is not None:
                cur.execute(f"SELECT term, ipa, audio FROM {g}.sound WHERE term = ANY(%s)",
                            (sorted(lemmas),))
                return {t: {"ipa": ipa, "audio": audio} for t, ipa, audio in cur.fetchall()}
    return build_lexicon(dump_path or DEFAULT_DUMP_PATH, lemmas, progress_cb=progress_cb)
