"""Direct Wikimedia Commons search for pronunciation audio (§ audio pronunciation).

kaikki's Wiktextract dump only captures audio that was linked into a word's
Wiktionary pronunciation section at snapshot time (2026-06-01) — confirmed
empirically to miss real, current recordings (e.g. "unpeople", "enkindle" both
have exact-match Commons files kaikki's structured extraction never surfaced).
This searches Commons directly as a second, independent pass over the words
kaikki came up empty for.

Two real failure modes found during testing, both handled here:
  - fuzzy/stemmed search matches ("rhymer" search returning files titled
    "rhyme...") -> filtered to an EXACT filename match against the target word.
  - cross-language contamination ("minimo" exact-matching Italian recordings
    tagged LL-Q652) -> filtered to English-tagged files only (Lingua Libre
    "LL-Q1860 (eng)-..." or the older "En[-us|-uk|-au]-..." convention).

Commons' public API rate-limits hard (429s within seconds of light use) — pacing
here is deliberately conservative; this is meant to run for hours, unattended.
"""

from __future__ import annotations

import hashlib
import re
import time
import urllib.parse

import requests

_SEARCH_API = "https://commons.wikimedia.org/w/api.php"
_UA = "concordance-audio-research/1.0 (personal vocabulary tool, non-commercial)"
_RETRY_STATUS = {429, 500, 502, 503, 504}

# LL-Q1860 = Lingua Libre's Wikidata QID for English. The older convention
# (En-us-word.ogg, En-word.ogg, En-uk-/-au-) is inherently English by naming.
_ENGLISH_LL = re.compile(r"^LL-Q1860\b", re.IGNORECASE)
_ENGLISH_OLD = re.compile(r"^En(-us|-uk|-au)?-", re.IGNORECASE)


def _request(session: requests.Session, url: str, params: dict | None = None,
             tries: int = 5, base_delay: float = 3.0):
    delay = base_delay
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, headers={"User-Agent": _UA}, timeout=20)
        except requests.RequestException:
            if attempt == tries - 1:
                return None
            time.sleep(delay); delay *= 2; continue
        if r.status_code in _RETRY_STATUS and attempt < tries - 1:
            retry_after = r.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.strip().isdigit() else delay
            time.sleep(min(max(wait, delay), 60.0))
            delay *= 2
            continue
        return r
    return None


def search_word(word: str, session: requests.Session) -> list[str]:
    """Raw Commons file-title hits for `word` (namespace 6 = File)."""
    r = _request(session, _SEARCH_API, {
        "action": "query", "list": "search", "srsearch": f"intitle:{word} filetype:audio",
        "srnamespace": 6, "format": "json", "srlimit": 15,
    })
    if r is None or r.status_code != 200:
        return []
    try:
        data = r.json()
    except ValueError:
        return []
    return [h["title"] for h in data.get("query", {}).get("search", [])]


def best_english_exact_match(titles: list[str], word: str) -> str | None:
    """The first title that is BOTH an exact filename match for `word` AND
    tagged English, or None. (Order from Commons search is relevance-ranked,
    so the first qualifying hit is a reasonable pick among ties.)"""
    word_lc = word.strip().lower()
    for title in titles:
        name = title[5:] if title.startswith("File:") else title  # strip "File:"
        stem = re.sub(r"\.(ogg|wav|mp3|mid|flac)$", "", name, flags=re.IGNORECASE)
        tail = stem.rsplit("-", 1)[-1].strip().lower()
        if tail != word_lc:
            continue
        if _ENGLISH_LL.match(stem) or _ENGLISH_OLD.match(stem):
            return title
    return None


def download_url(file_title: str) -> str:
    """Construct the upload.wikimedia.org URL from a 'File:Name.ext' title via
    MediaWiki's standard MD5-hash path convention (no extra API call needed)."""
    name = file_title[5:] if file_title.startswith("File:") else file_title
    name = name.strip().replace(" ", "_")
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return f"https://upload.wikimedia.org/wikipedia/commons/{h[0]}/{h[0:2]}/{urllib.parse.quote(name)}"
