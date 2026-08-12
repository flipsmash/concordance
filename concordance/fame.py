"""Book/author fame (historical & cultural importance) scoring.

An ABSOLUTE 1-10 score, LLM-judged against a fixed external rubric anchored
on real reference figures (Shakespeare = 10) -- explicitly NOT corpus-
relative percentile normalization. A percentile scale would guarantee a
uniform spread across whatever's currently ingested, but would also mean a
score drifts every time the corpus grows; the absolute scale is stable and
meaningful in isolation, at the cost of the resulting spread reflecting
however skewed real fame actually is (most of this corpus is public-domain/
pulp-heavy, so real score mass sits low, not spread evenly 1-10). This
trade was put to the user directly and this is the chosen side.

Signal sources, each proven live against real corpus authors before this
was written (Shakespeare / Sara Teasdale / Kris Neville -- a believable
~1000x spread at each step):
  - Google Books Ngram phrase frequency of the name/title (`ngram.fetch`,
    reused as-is -- already proven to handle multi-word phrases).
  - Wikidata sitelinks count (`wikidata_lookup`) -- the standard academic
    proxy for historical fame (see MIT Media Lab's "Pantheon"/Historical
    Popularity Index project: Wikipedia presence across many language
    editions correlates with lasting cultural importance far better than
    raw pageviews, which are recency-biased). Confirmed live: Shakespeare
    (Q692) = 335 sitelinks, Sara Teasdale (Q276013) = 42, Kris Neville
    (Q4315776) = 4.
  - Web search snippets (`websearch.search_snippets`, reused as-is).

Deliberately NOT included in v1:
  - Wikidata lookups for book TITLES -- confirmed live that `wbsearchentities`'s
    fuzzy name search can resolve to the wrong entity even for a real
    person's name ("Rex Beach" matched a spurious 0-sitelink disambiguation
    stub); title search collides with common phrases/film adaptations even
    more, with no equivalent cheap corroboration check available for authors
    (their Wikidata description can be checked for "writer"/"author"/etc.,
    a title has no such property to sanity-check against).
  - Wikipedia pageviews -- needs the same fragile article-resolution step
    as Wikidata (inheriting its false-match risk), costs another HTTP
    round-trip, and only covers July 2015 onward: a recency-biased "current
    relevance" signal on a corpus that's mostly older public-domain work.
    The one real insight behind it (multi-language persistence) is already
    captured more cheaply by the sitelinks count itself.

Every gathering function here follows ngram.fetch's/websearch.search_snippets's
own contract: never raise, return partial data with EXPLICIT failure/skip
markers rather than silently omitting a key -- a silent gap would look
identical to "the LLM considered this evidence and it was weak" instead of
"we never actually got the evidence," which is the one failure mode this
module is built to avoid (a whole run silently scoring blind is far worse
than a run that visibly records what it couldn't check).
"""

from __future__ import annotations

import json
import re

import requests

from .dictionary import _get
from .ngram import fetch as ngram_fetch
from .websearch import search_snippets

_WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Substrings checked against Wikidata's own short English description before
# its sitelinks count is trusted -- catches wbsearchentities resolving a
# common/ambiguous name to the wrong entity (confirmed live: "Rex Beach"
# matched an unrelated 0-sitelink stub). Absence of a corroborating
# description is itself recorded as a factor, not silently dropped.
_WRITER_DESCRIPTION_HINTS = (
    "writer", "author", "poet", "novelist", "playwright", "dramatist",
    "essayist", "journalist", "poetess", "storyteller", "screenwriter",
    "biographer", "literary",
)

# Book titles past this length rarely have meaningful Google Books Ngram
# phrase coverage regardless of the book's actual fame -- ngram.fetch's own
# "absent from corpus" return is all-zeros, and past a few tokens that stops
# meaning "obscure" and starts meaning "the query itself was hopeless."
# Skipping and recording that explicitly beats showing the LLM a misleading
# real zero.
_NGRAM_TITLE_TOKEN_LIMIT = 5

# book.author is stored in Project-Gutenberg/library-catalog order --
# "Last, First M. (Full Given Name)" -- confirmed live against the real
# corpus to cover 3,883/3,992 (97%) of distinct authors. Querying Wikidata/
# Ngram/DDGS with that raw string measurably degrades results (confirmed
# live: a 3-author smoke test came back "wikidata failed" for ALL THREE
# real, findable authors purely because "Aanrud, Hans" isn't how Wikidata
# indexes "Hans Aanrud") -- every external query in this module goes
# through _natural_name first.
_PAREN_RE = re.compile(r"\(([^)]+)\)")


def _natural_name(author: str) -> str:
    """'Last, First M. (Full Given Name)' -> 'Full Given Name Last' (prefers
    the parenthetical expansion when present, since it spells out
    abbreviated initials in full -- 'H. De Vere' vs. 'Henry De Vere').
    'First Last' (no comma) passes through unchanged, and so does any
    string this doesn't parse cleanly (never raises, worst case is the
    original string)."""
    if "," not in author:
        return author
    surname, rest = author.split(",", 1)
    surname = surname.strip()
    rest = rest.strip()
    paren = _PAREN_RE.search(rest)
    given = paren.group(1).strip() if paren else _PAREN_RE.sub("", rest).strip()
    if not given or not surname:
        return author
    return f"{given} {surname}"


def wikidata_lookup(name: str, session: requests.Session) -> dict | None:
    """Resolve `name` to a Wikidata entity and return
    {"qid", "sitelinks", "description", "corroborated"}, or None if no
    entity was found at all. `corroborated` is False when the entity's own
    description doesn't plausibly read as a writer -- callers should treat
    `sitelinks` as unreliable (but still record it) in that case."""
    resp = _get(session, _WIKIDATA_API, params={
        "action": "wbsearchentities", "search": name, "language": "en",
        "format": "json", "limit": 1, "type": "item",
    })
    if resp is None or resp.status_code != 200:
        return None
    try:
        hits = resp.json().get("search", [])
    except ValueError:
        return None
    if not hits:
        return None
    qid = hits[0].get("id")
    if not qid:
        return None

    resp = _get(session, _WIKIDATA_API, params={
        "action": "wbgetentities", "ids": qid, "props": "sitelinks|descriptions",
        "languages": "en", "format": "json",
    })
    if resp is None or resp.status_code != 200:
        return {"qid": qid, "sitelinks": None, "description": "", "corroborated": False}
    try:
        entity = resp.json().get("entities", {}).get(qid, {})
    except ValueError:
        return {"qid": qid, "sitelinks": None, "description": "", "corroborated": False}

    sitelinks = len(entity.get("sitelinks", {}))
    description = entity.get("descriptions", {}).get("en", {}).get("value", "")
    corroborated = any(hint in description.lower() for hint in _WRITER_DESCRIPTION_HINTS)
    return {"qid": qid, "sitelinks": sitelinks, "description": description, "corroborated": corroborated}


def gather_author_evidence(author: str, session: requests.Session) -> dict:
    """Every signal gathered for one author, explicitly marked where a
    source found nothing or failed outright. Never raises."""
    factors: dict = {}
    name = _natural_name(author)   # book.author is catalog-order ("Last, First") -- see _natural_name

    ngram = ngram_fetch(name, session)
    factors["ngram"] = (
        {"queried": name, "peak": ngram["peak"], "recent": ngram["recent"], "failed": False}
        if ngram is not None else {"queried": name, "failed": True}
    )

    wd = wikidata_lookup(name, session)
    factors["wikidata"] = wd if wd is not None else {"failed": True}

    snippets = search_snippets(f"{name} author writer", max_results=6)
    factors["snippets"] = snippets
    factors["snippets_failed"] = not snippets

    return factors


def gather_book_evidence(title: str, author: str, author_fame: dict | None,
                          session: requests.Session) -> dict:
    """Every signal gathered for one book. `author_fame`, when given, is
    {"fame_score", "fame_reasoning", "computed_at"} from an already-scored
    author row -- recorded verbatim as context, never used to skip or
    shortcut this book's own evidence gathering."""
    factors: dict = {}

    if len(title.split()) <= _NGRAM_TITLE_TOKEN_LIMIT:
        ngram = ngram_fetch(title, session)
        factors["ngram"] = (
            {"queried": title, "peak": ngram["peak"], "recent": ngram["recent"], "failed": False}
            if ngram is not None else {"queried": title, "failed": True}
        )
    else:
        factors["ngram"] = {"skipped": True, "reason": "title too long for a meaningful phrase query"}

    name = _natural_name(author) if author else ""   # catalog-order -- see _natural_name
    query = f"{title} {name} book" if name else f"{title} book"
    snippets = search_snippets(query, max_results=6)
    factors["snippets"] = snippets
    factors["snippets_failed"] = not snippets

    factors["author_fame_seen"] = author_fame   # None if not yet scored -- recorded as-is

    return factors


# --- LLM rubrics + scoring ---------------------------------------------------

AUTHOR_RUBRIC = (
    "Score this author's HISTORICAL/CULTURAL IMPORTANCE on an ABSOLUTE 1-10 "
    "scale, judged against the fixed reference points below -- NOT relative "
    "to any particular collection of books. Most authors who ever published "
    "a book, including ones still read today, score LOW (1-3): genuine "
    "lasting fame is rare. Anchors: "
    "10 = William Shakespeare, Jane Austen (foundational, universally known "
    "centuries later, taught worldwide). "
    "9 = Charles Dickens, Mark Twain, Fyodor Dostoyevsky (immensely famous, "
    "arguably as widely recognized as the very top tier, but not quite "
    "foundational/universal the way 10 is). "
    "8 = Edgar Allan Poe, Herman Melville, Charlotte Brontë (major canonical "
    "figures, widely read and taught, a clear notch below the 9 tier -- do "
    "not default here just because a figure is clearly major; if the "
    "evidence supports it, use 9). "
    "5-6 = a genre-defining figure known within their field but not a "
    "household name (e.g. a foundational pulp/genre writer still cited by "
    "historians of that form). "
    "2-3 = a competently, professionally published author of their era with "
    "no lasting individual reputation -- forgotten outside specialist "
    "bibliographies. This is the MOST COMMON case for an obscure public-"
    "domain author; do not inflate this to 5 out of politeness. "
    "1 = no discoverable trace of individual notability at all. "
    "Use your own literary knowledge PLUS the evidence provided (web "
    "snippets, Wikidata sitelink count, Google Ngram frequency of the name). "
    "Evidence marked failed/skipped/unavailable means 'we don't know', NOT "
    "'this person is obscure' -- never treat a missing signal as evidence of "
    "low fame."
)

BOOK_RUBRIC = (
    "Score this SPECIFIC BOOK's importance, 1-10, absolute scale, same "
    "anchor logic as author fame: "
    "10 = Hamlet/Pride and Prejudice-tier (a work virtually every literate "
    "person has heard of). "
    "9 = an extremely famous, near-universally recognized work, a rung "
    "below the very top (e.g. Great Expectations, Crime and Punishment, "
    "Frankenstein). "
    "8 = a major canonical novel widely taught in schools, clearly "
    "celebrated but not as universally recognized as the 9 tier (e.g. The "
    "Scarlet Letter, Vanity Fair) -- do not default here just because a "
    "work is clearly canonical; if the evidence supports it, use 9. "
    "5-6 = well-regarded within a genre or period but not broadly famous. "
    "2-3 = an unremarkable published work with no individual reputation -- "
    "the typical case; do not inflate this out of politeness. "
    "1 = no discoverable trace of this specific title. "
    "The author's own fame score, if given, is CONTEXT ONLY -- a weak "
    "prior, never a floor or cap: a famous author's minor or forgotten book "
    "must still score LOW on its own merits (e.g. a lesser potboiler by a "
    "famous author is a 2-3, NOT inflated toward the author's score) -- "
    "score the WORK, not the person. Evidence marked failed/skipped/"
    "unavailable means 'we don't know', NOT evidence of obscurity."
)

_RESPONSE_INSTRUCTIONS = (
    'Output ONLY a JSON object, no prose and no code fences: '
    '{"score": <integer 1-10>, "why": "<one-sentence justification, <=200 chars>"}.'
)

_MAX_REASONING_TOKENS = 200


def _format_factors(factors: dict) -> str:
    return json.dumps(factors, ensure_ascii=False, default=str)


def _parse_fame_verdict(text: str) -> dict | None:
    """Tolerantly pull {"score", "why"} out of a model reply -- handles code
    fences or prose wrapping, same approach as judge.py's _parse_verdicts."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
    start = text.find("{")
    if start == -1:
        return None
    snippet = text[start:]
    for end in range(len(snippet), 0, -1):
        try:
            data = json.loads(snippet[:end])
            break
        except json.JSONDecodeError:
            continue
    else:
        return None
    return data if isinstance(data, dict) else None


def _score_from_verdict(verdict: dict | None) -> tuple[float | None, str]:
    """(score, reasoning) from a parsed verdict, or (None, "") if the verdict
    is missing/malformed/out-of-range. A score outside [1,10] or non-numeric
    is a parse failure, never silently clamped -- a clamped nonsense value
    would look like a real, considered judgment."""
    if not verdict:
        return None, ""
    raw_score = verdict.get("score")
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None, ""
    if not (1.0 <= score <= 10.0):
        return None, ""
    why = str(verdict.get("why", ""))[:200]
    return score, why


def _query_llm(llm, system: str, user: str) -> str:
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=_MAX_REASONING_TOKENS,
    )
    return out["choices"][0]["message"]["content"] or ""


def score_author(llm, author: str, factors: dict, tries: int = 2) -> tuple[float | None, str]:
    """One (or, on an unparseable reply, up to `tries`) LLM call(s) ->
    (score, reasoning), or (None, "") on repeated parse failure."""
    system = f"{AUTHOR_RUBRIC}\n\n{_RESPONSE_INSTRUCTIONS}"
    # Natural name order in the prompt too, not just for external queries --
    # "AUTHOR: Jacobs, W. W. (William Wymark)" reads far worse to the model
    # than "AUTHOR: William Wymark Jacobs" (book.author is catalog-order;
    # see _natural_name).
    user = f"AUTHOR: {_natural_name(author)}\nEVIDENCE: {_format_factors(factors)}"
    for _ in range(tries):
        score, why = _score_from_verdict(_parse_fame_verdict(_query_llm(llm, system, user)))
        if score is not None:
            return score, why
    return None, ""


def score_book(llm, title: str, author: str, factors: dict, tries: int = 2) -> tuple[float | None, str]:
    system = f"{BOOK_RUBRIC}\n\n{_RESPONSE_INSTRUCTIONS}"
    name = _natural_name(author) if author else "an unknown author"
    user = f"BOOK: {title!r} by {name}\nEVIDENCE: {_format_factors(factors)}"
    for _ in range(tries):
        score, why = _score_from_verdict(_parse_fame_verdict(_query_llm(llm, system, user)))
        if score is not None:
            return score, why
    return None, ""
