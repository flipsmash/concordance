"""Stage 9 — enrichment (§03.9).

Look up each shortlisted word in FREE, keyless sources and attach the fields the
user asked for: definition, part of speech, IPA, synonyms, etymology. When more
than one sense is returned, the local model picks the one matching the book's
sentence (sense disambiguation); with no model, the first sense is used.

Primary:  Free Dictionary API  (https://api.dictionaryapi.dev) — def/POS/IPA/synonyms
Fallback: Wiktionary REST      (https://en.wiktionary.org)     — coverage + etymology

Network use here is fine — the "no API cost" rule is about paid LLM tokens, not
free dictionary lookups. Failures degrade gracefully: a word keeps its slot with
whatever fields were found.

Robustness: the rare/archaic words this tool exists to surface live almost
entirely in Wiktionary, not Free Dictionary. Running hundreds of lookups back to
back gets throttled (HTTP 429) by both hosts, which silently emptied ~84% of an
early full run. Every request now goes through ``_get``, which retries with
exponential backoff and honours ``Retry-After`` so bulk runs actually land their
definitions.
"""

from __future__ import annotations

import re
import time

import requests

from .model import Candidate

_FREEDICT = "https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
_WIKTIONARY = "https://en.wiktionary.org/api/rest_v1/page/definition/{word}"
_WIKT_RAW = "https://en.wiktionary.org/w/api.php"
_TIMEOUT = 8

# Wikimedia's REST API returns 403 without a descriptive User-Agent.
_USER_AGENT = "Concordance/0.1 (local vocabulary tool; https://github.com/)"

# Transient statuses worth retrying — rate limiting and gateway hiccups.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_TRIES = 4
_BACKOFF_BASE = 0.5   # seconds; doubles each retry


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    return session


def _get(session: requests.Session, url: str, params: dict | None = None):
    """GET with exponential backoff on throttling/transient errors.

    Returns the final ``Response`` (which may still be an error status the caller
    must check) or ``None`` if every attempt raised a network exception.
    """
    delay = _BACKOFF_BASE
    for attempt in range(_MAX_TRIES):
        try:
            resp = session.get(url, params=params, timeout=_TIMEOUT)
        except requests.RequestException:
            if attempt == _MAX_TRIES - 1:
                return None
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code in _RETRY_STATUS and attempt < _MAX_TRIES - 1:
            retry_after = resp.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.strip().isdigit() else delay
            time.sleep(min(wait, 30.0))   # never sleep absurdly long on a bad header
            delay *= 2
            continue
        return resp
    return None


def enrich(cand: Candidate, session: requests.Session | None = None) -> None:
    session = session or make_session()
    if _from_freedict(cand, session):
        cand.definition_source = "Free Dictionary API"
    elif _from_wiktionary(cand, session):
        cand.definition_source = "Wiktionary"
    else:
        return
    if not cand.etymology or not cand.ipa:
        _augment_from_raw(cand, session)


def _from_freedict(cand: Candidate, session: requests.Session) -> bool:
    resp = _get(session, _FREEDICT.format(word=cand.lemma))
    if resp is None or resp.status_code != 200:
        return False
    try:
        entry = resp.json()[0]
    except (ValueError, IndexError, KeyError):
        return False

    phonetics = [p.get("text", "") for p in entry.get("phonetics", []) if p.get("text")]
    cand.ipa = phonetics[0] if phonetics else ""
    if entry.get("origin"):          # Free Dictionary occasionally carries etymology here
        cand.etymology = entry["origin"].strip()

    senses = []
    for meaning in entry.get("meanings", []):
        pos = meaning.get("partOfSpeech", "")
        for d in meaning.get("definitions", []):
            if d.get("definition"):
                senses.append((pos, d["definition"], d.get("synonyms", [])))
        cand.synonyms = list({*cand.synonyms, *meaning.get("synonyms", [])})[:8]

    if not senses:
        return False
    pos, definition, syns = _pick_sense(cand, senses)
    cand.part_of_speech = pos
    cand.definition = definition
    if syns:
        cand.synonyms = list({*cand.synonyms, *syns})[:8]
    return True


def _from_wiktionary(cand: Candidate, session: requests.Session) -> bool:
    resp = _get(session, _WIKTIONARY.format(word=cand.lemma))
    if resp is None or resp.status_code != 200:
        return False
    try:
        data = resp.json().get("en", [])
    except ValueError:
        return False
    senses = []
    for block in data:
        pos = block.get("partOfSpeech", "")
        for d in block.get("definitions", []):
            gloss = _strip_html(d.get("definition", ""))
            if gloss:
                senses.append((pos, gloss, []))
    if not senses:
        return False
    pos, definition, _ = _pick_sense(cand, senses)
    cand.part_of_speech = pos
    cand.definition = definition
    return True


def _augment_from_raw(cand: Candidate, session: requests.Session) -> None:
    """Best-effort etymology + IPA from Wiktionary's plaintext extract. Silent on
    miss — the REST definition endpoint drops both sections, so this fills them in
    from a single extra call we make only for the shortlisted words."""
    resp = _get(session, _WIKT_RAW, params={
        "action": "query", "format": "json", "titles": cand.lemma,
        "prop": "extracts", "explaintext": 1, "redirects": 1,
    })
    if resp is None or resp.status_code != 200:
        return
    try:
        pages = resp.json().get("query", {}).get("pages", {})
    except ValueError:
        return
    text = next((p.get("extract", "") for p in pages.values()), "")
    if not cand.etymology:
        ety = _parse_etymology(text)
        if ety:
            cand.etymology = ety
    if not cand.ipa:
        ipa = _parse_ipa(text)
        if ipa:
            cand.ipa = ipa


# First /slashed/ or [bracketed] IPA transcription after an "IPA" cue.
_IPA_RE = re.compile(r"IPA[^/\[\n]{0,20}?(/[^/\n]+/|\[[^\]\n]+\])")


def _parse_ipa(text: str) -> str:
    m = _IPA_RE.search(text)
    return m.group(1).strip() if m else ""


# Section headers that end the Etymology block in a plaintext Wiktionary extract.
_ETY_STOP = re.compile(
    r"^(Pronunciation|Noun|Verb|Adjective|Adverb|Alternative forms|Derived terms"
    r"|Related terms|Descendants|Anagrams|References|See also|Usage notes"
    r"|Conjugation|Declension|Translations|Etymology \d)\b",
    re.IGNORECASE,
)


def _parse_etymology(text: str) -> str:
    """Pull the first Etymology section out of a plaintext dump. Headers arrive
    either bare ("Etymology") or wrapped ("=== Etymology ===")."""
    lines = text.splitlines()
    out: list[str] = []
    capturing = False
    for raw in lines:
        line = raw.strip().strip("= ").strip()
        if not capturing:
            if re.match(r"^Etymology(\s+1)?$", line, re.IGNORECASE):
                capturing = True
            continue
        if not line:
            if out:
                break            # blank line after collecting = end of section
            continue
        if _ETY_STOP.match(line):
            break
        out.append(line)
    ety = " ".join(out).strip()
    return ety if len(ety) > 3 else ""


def _pick_sense(cand: Candidate, senses: list[tuple[str, str, list]]) -> tuple[str, str, list]:
    """Choose the sense matching the book's sentence. The LLM sense-picker wires
    in here; until then, prefer a sense whose POS matches the tagger, else first."""
    if len(senses) == 1:
        return senses[0]
    tagged = _COARSE_POS.get(cand.pos)
    if tagged:
        for s in senses:
            if s[0].lower().startswith(tagged):
                return s
    return senses[0]


_COARSE_POS = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb"}


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()
