"""Stage 3 follow-on — collapse a derived surface form to its underlying root
when the root is independently a legitimate word, so the derived form and
its root aren't counted, floored, and judged as two separate vocabulary
items (quickly -> quick, unhappy -> happy).

Two productive, near-total English derivations get this treatment:

  - a regular "-ly" adverb -> its adjective root. An adverb's difficulty is
    almost always inherited straight from the adjective, and wordfreq
    undercounts the "-ly" form even when the underlying concept is ordinary
    (see validity_score.effective_zipf's docstring for the same observation
    applied to scoring only) — so going forward the adjective is what gets
    ingested, not the adverb.
  - "un-" + adjective/participle -> the un-less root, same reasoning. This
    mirrors the frequency-floor peel validity_score._morph_root already does
    for SCORING (its "un" entry in _PREFIXES), but this module actually
    changes candidate identity rather than just the Zipf used to score it.

The two chain once: an "un-" root that is itself a regular "-ly" adverb
(unfrowardly -> frowardly -> froward) reduces the rest of the way, since
that pattern (un + adjective + ly) is common enough among this project's
real archaic vocabulary to be worth the extra step. Nothing chains further
than that — see derived_root's docstring.

Deliberately conservative: the "un-" side only fires when the stripped form
is attested in BOTH WordNet and NLTK's 234k-word corpus (stricter than
validity_score._morph_root's own OR-gate — see _known_root's docstring for
why a false positive here is worse than in _morph_root's scoring-only use),
so "under"/"uncle"/"union" (not real un+root derivations) pass through
untouched, as do adverbs with no attested adjective root (inobtrusively,
scrofulously — see validity_score's own module docstring for why an LLM is
never the authority here either).
"""

from __future__ import annotations

_words: set[str] | None = None


def _wordset() -> set[str]:
    global _words
    if _words is None:
        try:
            from nltk.corpus import words as nltk_words
            _words = set(w.lower() for w in nltk_words.words())
        except Exception:
            _words = set()
    return _words


def _in_wordnet(word: str) -> bool:
    try:
        from nltk.corpus import wordnet
        return bool(wordnet.synsets(word))
    except Exception:
        return False


def _known_root(word: str) -> bool:
    """Deliberately STRICTER than validity_score._morph_root's known() gate
    (which accepts wordset OR wordnet OR a bare zipf>=2.0, and a 3-letter
    floor): that gate is fine for _morph_root's use, a MAX() nudge to a
    score that only matters when it moves a decision, but here a false
    positive doesn't nudge a score — it silently becomes the word's new
    identity. The permissive gate lets short coincidental substrings of a
    non-decomposable "un-" word pass as if they were real roots: "der"
    (under) clears wordnet on its own, "cle" (uncle) clears zipf>=2.0 (web
    noise — URLs, abbreviations), "ion"/"til" (union/until) are real short
    words that just aren't what those are built from. Requiring BOTH
    authorities to agree, plus a 4-letter floor, is empirically the
    smallest gate that still passes every unhappy/unsafe/unwise-style
    derivation tried while rejecting every un+coincidence found (the one
    casualty: "unfit" -> "fit", a 3-letter root, no longer reduces)."""
    return len(word) >= 4 and word in _wordset() and _in_wordnet(word)


def _is_adjective(word: str) -> bool:
    try:
        from nltk.corpus import wordnet
        return any(s.pos() in ("a", "s") for s in wordnet.synsets(word))
    except Exception:
        return False


# (suffix, replacement) tried in order; the first that both matches the
# surface form and lands on an attested adjective wins. Covers the regular
# spelling-change patterns (-ically/-ibly/-ably/-ily) before the plain -ly
# strip, so "basically" tries "basic" (via -ically) rather than "basical"
# (via the too-eager plain -ly rule) first.
_LY_STRIP = (
    ("ically", "ic"),
    ("ibly", "ible"),
    ("ably", "able"),
    ("ily", "y"),
    ("ly", ""),
)


def adverb_to_adjective(word: str) -> str | None:
    """The adjective root of a regular "-ly" adverb, or None if `word`
    isn't one / has no attested root (jauntily -> jaunty, culpably ->
    culpable; inobtrusively -> None — no WordNet entry for "inobtrusive")."""
    word = word.strip().lower()
    for suf, repl in _LY_STRIP:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            cand = word[: -len(suf)] + repl
            if cand != word and _is_adjective(cand):
                return cand
    return None


def _un_strip(word: str) -> str | None:
    """Bare structural "un-" strip (length floor only, no attestation check)
    — used only as an INTERMEDIATE hand-off to adverb_to_adjective inside
    derived_root's "unfrowardly" chain below. un_root() is the gated,
    standalone-identity version; this one is safe to leave ungated because
    adverb_to_adjective's own WordNet-adjective requirement is what actually
    decides whether the chain is real, same as un_root gates the other
    chaining direction ("unhappily" -> "unhappy" -> "happy")."""
    word = word.strip().lower()
    if not word.startswith("un") or len(word) - 2 < 4:
        return None
    return word[2:]


def un_root(word: str) -> str | None:
    """The un-less root of an "un-" word, or None if the stripped form isn't
    independently attested (unhappy -> happy; under/uncle/union -> None,
    since "der"/"cle"/"ion" fail the _known_root gate)."""
    stem = _un_strip(word)
    return stem if stem and _known_root(stem) else None


def derived_root(word: str, pos: str) -> tuple[str, str] | None:
    """(root_lemma, root_pos) if `word`/`pos` (a spaCy coarse POS tag) should
    be substituted with its root for ingestion, else None.

    `pos` only gates the adverb checks (un_root applies regardless of pos,
    since "un-" attaches across categories); the returned pos is "ADJ"
    once either rule has fired, otherwise `pos` unchanged.

    Chains once in whichever order the surface form needs, since either
    derivation can sit on the outside:
      - "unhappily" is un+(happy+ly) — the -ily strip lands on "unhappy"
        first, which then needs its own "un-" peeled (the gated un_root) to
        reach "happy".
      - "unfrowardly" is (un+froward)+ly, where the -ly strip on the whole
        word finds no adjective ("unfroward" isn't a WordNet entry) and
        it's the "un-" peel that must go first, landing on "frowardly",
        which then reduces to "froward". The un- peel here uses the
        ungated _un_strip, not un_root — "frowardly" itself doesn't clear
        _known_root's WordNet bar (WordNet's adverb coverage is thin), but
        adverb_to_adjective's own strict gate on the FINAL result ("froward"
        must be a real WordNet adjective) makes the intermediate step safe
        to leave unchecked.
    Trying the natural single-step order first and falling back to the
    other keeps both covered without open-ended recursion."""
    word = word.strip().lower()
    if pos == "ADV":
        adj = adverb_to_adjective(word)
        if adj:
            deeper = un_root(adj)
            return (deeper, "ADJ") if deeper else (adj, "ADJ")
        stripped = _un_strip(word)
        if stripped:
            adj = adverb_to_adjective(stripped)
            if adj:
                return adj, "ADJ"
        return None
    root = un_root(word)
    if root:
        adj = adverb_to_adjective(root)
        if adj:
            return adj, "ADJ"
        return root, pos
    return None
