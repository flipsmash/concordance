"""Deep definition lookup for the hard tail (§03.9 follow-on).

The everyday enrichment (Free Dictionary + Wiktionary, see dictionary.py) leaves a
tail of archaic / nonce / dialectal words undefined — exactly the interesting
ones. This module reaches further, in cascade, and is meant to run ONLY on words
still missing a definition:

  1. Wordnik      — the Century Dictionary + Webster's (gcide) + AHD, which carry
                    Shakespearean/archaic vocabulary the free APIs lack. Needs a
                    free API key in $WORDNIK_API_KEY (or a git-ignored .env file).
  2. yourdictionary.com — scraped meta-definition; a keyless aggregator that
                    sometimes has nonce words (e.g. "ungenitured") the APIs don't.
  3. web search + LLM extraction — a last resort that reads REAL search-result
                    text and has the local model *extract* a definition that is
                    actually present (never invent one). Opt-in (slow); see
                    websearch.py.

Every network call goes through dictionary._get, so Wordnik's aggressive rate
limiting (HTTP 429) is handled by the same exponential backoff.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests

from .dictionary import _get
from .model import Candidate

_WORDNIK = "https://api.wordnik.com/v4/word.json/{word}/definitions"
_YOURDICT = "https://www.yourdictionary.com/{word}"

# Wordnik source dictionaries in preference order; century/gcide carry the archaic
# glosses, ahd-5/wiktionary the modern ones. Others are fine too but ranked after.
_WORDNIK_PREF = ("century", "gcide", "ahd-5", "ahd", "wiktionary", "wordnet")


def wordnik_key() -> str:
    """Read the Wordnik key from the environment, loading a git-ignored .env once."""
    if "WORDNIK_API_KEY" not in os.environ:
        _load_dotenv(Path(".env"))
    return os.environ.get("WORDNIK_API_KEY", "").strip()


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def deep_enrich(cand: Candidate, session: requests.Session, key: str | None = None) -> bool:
    """Try the keyed/scraped sources in order. Returns True if a definition was set.
    Does NOT include the web+LLM tier — that is orchestrated separately since it
    needs the model and is opt-in."""
    key = wordnik_key() if key is None else key
    if key and _from_wordnik(cand, session, key):
        return True
    if _from_yourdictionary(cand, session):
        return True
    return False


def _from_wordnik(cand: Candidate, session: requests.Session, key: str) -> bool:
    resp = _get(session, _WORDNIK.format(word=cand.lemma), params={
        "limit": 8, "sourceDictionaries": "all", "includeRelated": "false",
        "useCanonical": "true", "api_key": key,
    })
    if resp is None or resp.status_code != 200:
        return False
    try:
        entries = resp.json()
    except ValueError:
        return False
    defs = [e for e in entries if isinstance(e, dict) and (e.get("text") or "").strip()]
    if not defs:
        return False
    # Same source dictionary can carry more than one entry (e.g. Century's
    # noun headword AND a secondary cross-reference gloss like "cangue" ->
    # "To sentence to the cangue.", which has no partOfSpeech at all) -- sort
    # is stable, so without a tiebreaker the API's own response order decides,
    # and a no-POS cross-reference sense can end up ahead of the real
    # definition purely by chance. Prefer an entry with a real partOfSpeech
    # within the same source-dictionary rank.
    defs.sort(key=lambda e: (
        _WORDNIK_PREF.index(e["sourceDictionary"]) if e.get("sourceDictionary") in _WORDNIK_PREF else len(_WORDNIK_PREF),
        0 if e.get("partOfSpeech") else 1,
    ))
    best = defs[0]
    cand.definition = _clean(best["text"])
    cand.part_of_speech = best.get("partOfSpeech", "") or cand.part_of_speech
    cand.definition_source = f"Wordnik ({best.get('sourceDictionary', '')})"
    return True


def _from_yourdictionary(cand: Candidate, session: requests.Session) -> bool:
    resp = _get(session, _YOURDICT.format(word=cand.lemma))
    if resp is None or resp.status_code != 200:
        return False
    m = re.search(r'<meta name="description" content="([^"]+)"', resp.text)
    if not m:
        return False
    desc = _clean(m.group(1))
    # Pages read "<Word> definition: <gloss>."; strip the lead-in, bail if it's the
    # generic not-found blurb (no "definition:" marker).
    lead = re.match(r"(?i)^.{0,40}?\bdefinition:\s*(.+)$", desc)
    if not lead:
        return False
    gloss = lead.group(1).strip().rstrip(".").strip()
    if len(gloss) < 3:
        return False
    cand.definition = gloss
    cand.definition_source = "yourdictionary.com"
    return True


def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)             # strip any HTML tags
    return re.sub(r"\s+", " ", s).strip()
