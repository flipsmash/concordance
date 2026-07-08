"""Google Books Ngram features (§ difficulty).

Fetched once and cached per word (network is the expensive part). Serves two
consumers: the archaic flag (a word that *was* common and faded is archaic — the
recency ratio) and the difficulty scalar (raw rarity in print).

Per word we keep: peak relative frequency (rarity), recent frequency (2000-2019),
the peak year, and recency_ratio = recent/peak (low => declined from its heyday).
"""

from __future__ import annotations

import time

import requests

_NGRAM = "https://books.google.com/ngrams/json"
_RECENT_YEARS = 20          # 2000-2019
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _get(word: str, session: requests.Session, tries: int = 4):
    delay = 0.5
    for attempt in range(tries):
        try:
            r = session.get(_NGRAM, params={
                "content": word, "year_start": 1500, "year_end": 2019,
                "corpus": "en-2019", "smoothing": 3}, timeout=15)
        except requests.RequestException:
            if attempt == tries - 1:
                return None
            time.sleep(delay); delay *= 2; continue
        if r.status_code in _RETRY_STATUS and attempt < tries - 1:
            time.sleep(delay); delay *= 2; continue
        return r
    return None


def fetch(word: str, session: requests.Session) -> dict | None:
    """Ngram features for `word`, or None on a hard network failure. A word absent
    from the corpus returns zeros (that is a real signal, not a failure)."""
    r = _get(word, session)
    if r is None or r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    if not data or not data[0].get("timeseries"):
        return {"peak": 0.0, "recent": 0.0, "recency_ratio": None, "peak_year": None}
    ts = data[0]["timeseries"]
    peak = max(ts)
    peak_year = 1500 + ts.index(peak)
    recent = sum(ts[-_RECENT_YEARS:]) / _RECENT_YEARS
    ratio = (recent / peak) if peak > 0 else None
    return {"peak": peak, "recent": recent, "recency_ratio": ratio, "peak_year": peak_year}
