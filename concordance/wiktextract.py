"""kaikki/Wiktextract dump lookup (§ audio pronunciation).

The current ad hoc IPA scrape (dictionary.py, one Wiktionary REST call per word)
tops out around 50% coverage and has no access to real recorded pronunciation
audio. kaikki.org publishes a structured, offline JSONL dump of the entire English
Wiktionary (`wiktextract`) with per-word `sounds` entries carrying both IPA
(dialect-tagged) and Wikimedia Commons audio URLs when a human recording exists.
One local pass over the dump answers both questions at once, for free, with no
per-word API calls or rate limits.

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
            ipas, audios = [], []
            for s in obj.get("sounds") or []:
                if s.get("ipa"):
                    ipas.append({"ipa": s["ipa"], "tags": s.get("tags", [])})
                url = s.get("ogg_url") or s.get("mp3_url")
                if url:
                    audios.append({"url": url, "tags": s.get("tags", [])})
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
