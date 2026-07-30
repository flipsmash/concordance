"""Merriam-Webster word lookup — official Collegiate Dictionary API (primary).

One key, one dictionary (Collegiate), capped at 1000 queries/day per MW's free
tier. To stay under that:

  - Every response (hit AND confirmed miss) is cached on disk keyed by the
    lowercased word, so re-looking-up a word already seen today (or last
    month) never costs a query.
  - A daily usage counter refuses new API calls once the count reaches
    _DAILY_QUOTA, so a batch run can't blow through the quota partway and
    take down other tools sharing the same key. Callers should fall back to
    concordance.mw_scrape (the Playwright site-scraper) when this returns no
    entries — that's the signal for "API can't help right now," whether the
    cause is a real miss or the daily cap.

See concordance.mw_scrape for the site-scraping fallback, used for words the
API doesn't have but the live site does.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import quote

import requests
from rich.console import Console

from .dictionary import _get

_API_URL = "https://www.dictionaryapi.com/api/v3/references/collegiate/json/{word}"
_AUDIO_URL = "https://media.merriam-webster.com/audio/prons/en/us/mp3/{subdir}/{file}.mp3"

_CACHE_DIR = Path(".cache")
_CACHE_FILE = _CACHE_DIR / "mw_api_cache.json"
_USAGE_FILE = _CACHE_DIR / "mw_api_usage.json"
_DAILY_QUOTA = 1000
_WARN_THRESHOLD = 900

# Guards the entire uncached branch of lookup_api (cache read-check, quota
# check, API call, cache write) now that ingest's parallel enrichment step
# calls this from multiple worker threads -- previously only mw_backfill's
# sequential loop called it, so the read-modify-write race on both the cache
# file and the usage counter never mattered. Coarse on purpose: locking only
# the usage-counter increment still leaves the cache save race (two threads
# resolving two different new words can each load the cache before either
# saves, and the second save clobbers the first thread's new entry -- that
# word then costs a second query next time). Cached hits stay lock-free;
# only genuinely new words serialize against each other, which is fine --
# MW is quota-scarce and doesn't need concurrency anyway.
#
# Thread-scoped only, not cross-process: this does not protect against a
# concurrent `ingest` and `mw-backfill` invocation racing the same on-disk
# cache/usage files. Don't run them at the same time.
_LOCK = threading.Lock()


@dataclass
class MWPronunciation:
    respelling: str = ""
    audio_url: str | None = None


@dataclass
class MWEntry:
    headword: str
    part_of_speech: str
    definitions: list[str] = field(default_factory=list)
    pronunciations: list[MWPronunciation] = field(default_factory=list)
    etymology: str = ""
    first_known_use: str = ""
    source: str = "Merriam-Webster API"


def mw_api_key() -> str:
    """Read the MW Collegiate Dictionary key from the environment, loading a
    git-ignored .env once (same convention as deepdef.wordnik_key)."""
    if "MW_DICTIONARY_API_KEY" not in os.environ:
        _load_dotenv(Path(".env"))
    return os.environ.get("MW_DICTIONARY_API_KEY", "").strip()


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# --- disk cache + quota tracking --------------------------------------------

def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json(path: Path, data: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _today() -> str:
    return date.today().isoformat()


def _usage_count() -> int:
    usage = _load_json(_USAGE_FILE)
    return usage.get("count", 0) if usage.get("date") == _today() else 0


def _record_usage() -> int:
    usage = _load_json(_USAGE_FILE)
    today = _today()
    count = usage.get("count", 0) + 1 if usage.get("date") == today else 1
    _save_json(_USAGE_FILE, {"date": today, "count": count})
    return count


def quota_exhausted() -> bool:
    """Whether today's usage has already hit the daily cap. `lookup_api` already
    checks this internally before spending a fresh (uncached) call, but a batch
    caller (e.g. db.mw_backfill) wants to check this BEFORE even picking its next
    candidate, so it can stop the whole run cleanly rather than silently getting
    an empty result back for every remaining word."""
    return _usage_count() >= _DAILY_QUOTA


# --- API tier ----------------------------------------------------------------

def lookup_api(word: str, api_key: str, session: requests.Session | None = None,
                console: Console | None = None) -> list[MWEntry]:
    """Cached, quota-respecting API lookup. Empty list means "try the scrape
    fallback" -- covers both a confirmed miss and a quota-exhausted skip; the
    caller doesn't need to tell those apart, just fall back either way."""
    session = session or requests.Session()
    key = word.lower()

    cache = _load_json(_CACHE_FILE)
    if key in cache:
        raw = cache[key]
    else:
        with _LOCK:
            cache = _load_json(_CACHE_FILE)   # re-read: another thread may have just cached this
            if key in cache:
                raw = cache[key]
            else:
                if _usage_count() >= _DAILY_QUOTA:
                    if console:
                        console.print(f"[yellow]MW API daily quota ({_DAILY_QUOTA}) reached — skipping API for {word!r}.[/yellow]")
                    return []
                resp = _get(session, _API_URL.format(word=quote(word)), params={"key": api_key})
                if resp is None:
                    if console:
                        console.print(f"[yellow]MW API unreachable for {word!r} — not caching, will retry next time.[/yellow]")
                    return []
                used = _record_usage()   # a request reached MW (good key or bad) -- counts against quota either way
                if console and used >= _WARN_THRESHOLD:
                    console.print(f"[yellow]MW API usage today: {used}/{_DAILY_QUOTA}[/yellow]")
                if resp.status_code != 200:
                    if console:
                        console.print(f"[yellow]MW API returned HTTP {resp.status_code} for {word!r} — not caching (check MW_DICTIONARY_API_KEY).[/yellow]")
                    return []
                try:
                    raw = resp.json()
                except ValueError:
                    if console:
                        console.print(f"[yellow]MW API returned non-JSON for {word!r} — not caching.[/yellow]")
                    return []
                cache[key] = raw
                _save_json(_CACHE_FILE, cache)

    if not raw or isinstance(raw[0], str):   # list[str] == MW's "did you mean" no-match signal
        return []
    return [e for e in (_parse_api_entry(r) for r in raw if isinstance(r, dict)) if e]


# --- MW's inline markup ({bc}, {it}...{/it}, {sx|word||}, ...) --------------

_TOKEN_RE = re.compile(r"\{(/?[a-z_]+)\|?([^{}]*)\}")
_STRIP_TAGS = {"it", "/it", "b", "/b", "inf", "/inf", "sup", "/sup", "gloss", "/gloss", "phrase", "/phrase"}
_LINK_TAGS = {"sx", "a_link", "d_link", "i_link", "et_link", "dx_def"}


def _clean_mw_markup(text: str) -> str:
    def repl(m: re.Match) -> str:
        tag, arg = m.group(1), m.group(2)
        if tag == "bc":
            return ":"
        if tag in _STRIP_TAGS:
            return ""
        if tag in _LINK_TAGS:
            return arg.split("|")[0]
        return ""
    return _TOKEN_RE.sub(repl, text).strip()


def _audio_subdir(file: str) -> str:
    """MW's documented CDN bucket rule for pronunciation audio filenames."""
    if file.startswith("bix"):
        return "bix"
    if file.startswith("gg"):
        return "gg"
    if file[:1].isdigit() or not file[:1].isalpha():
        return "number"
    return file[0]


def _parse_api_entry(raw: dict) -> MWEntry | None:
    hwi = raw.get("hwi", {})
    headword = hwi.get("hw", raw.get("meta", {}).get("id", "")).replace("*", "")
    pos = raw.get("fl", "")

    pronunciations = []
    for p in hwi.get("prs", []):
        respelling = p.get("mw", "")
        audio_file = p.get("sound", {}).get("audio")
        audio_url = _AUDIO_URL.format(subdir=_audio_subdir(audio_file), file=audio_file) if audio_file else None
        if respelling or audio_url:
            pronunciations.append(MWPronunciation(respelling=respelling, audio_url=audio_url))

    definitions = [_clean_mw_markup(d) for d in raw.get("shortdef", [])]
    if not definitions:
        return None

    etymology = " ".join(_clean_mw_markup(part[1]) for part in raw.get("et", []) if part[0] == "text").strip()
    first_known_use = _clean_mw_markup(raw.get("date", ""))

    return MWEntry(
        headword=headword, part_of_speech=pos, definitions=definitions,
        pronunciations=pronunciations, etymology=etymology,
        first_known_use=first_known_use, source="Merriam-Webster API",
    )


# Coarse tagger tag -> the lowercase prefix MW's own `fl`/part_of_speech text
# would start with -- same mapping dictionary.py's _pick_sense uses for the
# identical tie-break, against a different source's response shape.
_COARSE_POS = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb"}


def pick_entry(entries: list[MWEntry], tagger_pos: str = "") -> MWEntry:
    """Choose the homograph entry (MW splits "run" into separate verb/noun/
    adjective entries, each its own dict) matching the word's own
    tagger-derived POS, else the first entry MW itself returned -- MW's own
    ordering is most-common-sense-first, the same fallback dictionary.py's
    _pick_sense uses when no POS hint is available or matches."""
    if len(entries) == 1 or not tagger_pos:
        return entries[0]
    target = _COARSE_POS.get(tagger_pos)
    if not target:
        return entries[0]
    for e in entries:
        if e.part_of_speech.lower().startswith(target):
            return e
    return entries[0]


# MW tags a not-yet-naturalized loanword's POS as "<Language> noun/verb/..."
# -- e.g. "Swahili noun" for "hatari" ("danger") -- capitalized, distinct from
# its lowercase grammatical modifiers ("plural noun", "combining form").
# Checked against the RAW `fl` string, before model.normalize_pos lowercases
# everything and destroys this exact signal. Confirmed live: "hatari" ->
# "Swahili noun" -- a real foreign word this project's other sources would
# never have surfaced, caught only because MW's own category name for it
# says so.
_FOREIGN_POS_RE = re.compile(r"^[A-Z][a-z]+ (noun|verb|adjective|adverb|combining form)s?$")


def is_foreign_pos(raw_pos: str) -> bool:
    return bool(_FOREIGN_POS_RE.match(raw_pos.strip()))


def _normalize_headword(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def exact_matches(entries: list[MWEntry], word: str) -> list[MWEntry]:
    """Entries whose own headword literally IS `word`, ignoring case/spaces/
    hyphens/asterisked-syllable-marks. MW's API does fuzzy full-text search
    and will happily return a same-ballpark idiom for a query that isn't a
    real headword at all -- confirmed on live data: querying "atune" matched
    the idiom "sing a different tune", "atrail" matched "blaze a trail",
    "aglance" matched "at a glance". Fine for a human browsing
    lookup_mw.py's fuller interactive results (that's genuinely useful
    context there), but an automated writer (db.mw_backfill) needs this
    filter first so it never records an unrelated idiom's definition as if
    it were the literal queried word's own."""
    target = _normalize_headword(word)
    return [e for e in entries if _normalize_headword(e.headword) == target]
