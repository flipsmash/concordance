"""Definitions for undefined words whose meaning follows from a known root.

Some active words have no dictionary entry anywhere because nobody lists
them, not because they aren't words:
  - archaic verb inflections: inveighest = "(archaic) second-person singular
    present of inveigh";
  - superlatives of rare adjectives: puissantest, untowardest;
  - transparent derivatives: visageless, propensness, affiancement.

Each rule fires only when the root is in the local Wiktionary with the part
of speech the affix needs (-ness/un- need an adjective, -less/-ful a noun,
-ment and -est/-eth inflections a verb). That keeps coincidental splits out
(irrased is not ir- + "rased"). The definition quotes the root's own first
sense and is marked `derived (...)` in definition_source, so it never passes
for a dictionary entry. Words scored likely-artifact are left for pruning,
not defined.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .crossref import classification_gloss, is_abbreviation_stub, mentions_pointer


@dataclass
class Proposal:
    word_id: int
    lemma: str
    kind: str            # "inflection" | "derivative"
    root: str
    definition: str
    source: str          # definition_source to store
    part_of_speech: str


def _gerund(verb: str) -> str:
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and not verb.endswith(("ee", "ye", "oe")):
        return verb[:-1] + "ing"
    return verb + "ing"


def _stems(stem: str) -> list[str]:
    """Candidate roots for a stem left after removing a suffix: as is,
    with a silent e restored, with y restored from i, and undoubled
    (bid-dest -> bid)."""
    out = [stem, stem + "e"]
    if stem.endswith("i"):
        out.append(stem[:-1] + "y")
    if len(stem) > 2 and stem[-1] == stem[-2]:
        out.append(stem[:-1])
    return out


# suffix -> (root part of speech, kind, template); {r} root, {g} root's gloss
_SUFFIXES = [
    ("est", "VERB", "inflection", "(archaic) second-person singular present of {r} — {g}"),
    ("eth", "VERB", "inflection", "(archaic) third-person singular present of {r} — {g}"),
    ("est", "ADJECTIVE", "inflection", "Superlative form of {r} — {g}"),
    ("less", "NOUN", "derivative", "Having no {r} ({r}: {g})"),
    ("ness", "ADJECTIVE", "derivative", "The quality or state of being {r} ({r}: {g})"),
    ("ment", "VERB", "derivative", "The act or result of {ing} ({r}: {g})"),
    ("ful", "NOUN", "derivative", "Full of {r} ({r}: {g})"),
    ("ly", "ADJECTIVE", "derivative", "In {a} {r} manner ({r}: {g})"),
    ("ish", "ADJECTIVE", "derivative", "Somewhat {r} ({r}: {g})"),
]
_PREFIXES = [
    ("un", "ADJECTIVE", "derivative", "Not {r} ({r}: {g})"),
    ("over", "ADJECTIVE", "derivative", "Excessively {r} ({r}: {g})"),
    ("over", "VERB", "derivative", "To {r} excessively ({r}: {g})"),
]
_POS_OUT = {"NOUN": "noun", "VERB": "verb", "ADJECTIVE": "adjective"}
# the part of speech the DERIVED word has
_DERIVED_POS = {"less": "adjective", "ness": "noun", "ment": "noun", "ful": "adjective",
                "ly": "adverb", "ish": "adjective", "un": None, "over": None}


def propose(conn, schema: str = "concordance", *, min_common_zipf: float = 3.5) -> list[Proposal]:
    """One proposal per undefined, active, not-likely-artifact word a rule
    covers (the first matching rule wins). Inflections of COMMON verbs are
    skipped: those are cast out by clean-archaic-spellings (design rule 3),
    not defined."""
    from wordfreq import zipf_frequency

    from .db import _safe_schema

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT id, lemma FROM {s}.word
                        WHERE active AND coalesce(definition, '') = ''
                          AND coalesce(validity_label, '') <> 'likely-artifact'""")
        words = cur.fetchall()

    def candidates(w: str):
        for suf, pos, kind, tmpl in _SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                for root in _stems(w[: -len(suf)]):
                    yield root, pos, kind, tmpl, suf
        for pre, pos, kind, tmpl in _PREFIXES:
            if w.startswith(pre) and len(w) - len(pre) >= 3:
                yield w[len(pre):], pos, kind, tmpl, pre

    roots = {r for _, lemma in words for r, *_ in candidates(lemma.lower())}
    with conn.cursor() as cur:
        cur.execute("""SELECT DISTINCT ON (lower(term), part_of_speech) lower(term), part_of_speech, definition
                       FROM vocab.wiktionary WHERE lower(term) = ANY(%s)
                       ORDER BY lower(term), part_of_speech, id""", (sorted(roots),))
        senses: dict[tuple[str, str], str] = {}
        for term, pos, d in cur.fetchall():
            gloss = classification_gloss((d or "").split(";")[0]).rstrip(" .")
            # a root whose own entry is a stub or a pointer, or wiki markup,
            # says nothing usable about it
            if (not gloss or is_abbreviation_stub(gloss) or mentions_pointer(gloss)
                    or re.search(r"[{}\[\]|]|,,|^\W|^(?:or|and)\b", gloss)):
                continue
            senses[(term, (pos or "").upper())] = gloss[:140]

    out = []
    for wid, lemma in words:
        w = lemma.lower()
        for root, pos, kind, tmpl, affix in candidates(w):
            gloss = senses.get((root, pos))
            if not gloss or root == w:
                continue
            if kind == "inflection" and pos == "VERB" and zipf_frequency(root, "en") >= min_common_zipf:
                break                       # an archaic form of a common verb: cast out, not defined
            if tmpl.startswith("Superlative") and root.endswith("ght"):
                continue                    # bethoughtest is "thou bethoughtest", not a superlative
            text = tmpl.format(r=root, g=gloss, ing=_gerund(root), a="an" if root[0] in "aeiou" else "a")
            out_pos = ("verb" if pos == "VERB" else "adjective") if kind == "inflection" else (
                _DERIVED_POS.get(affix) or _POS_OUT[pos])
            label = f"inflection of {root}" if kind == "inflection" else (
                f"{affix}- + {root}" if affix in ("un", "over") else f"{root} + -{affix}")
            out.append(Proposal(wid, lemma, kind, root, text[0].upper() + text[1:] if not text.startswith("(") else text,
                                f"derived ({label})", out_pos))
            break
    return out


def apply(conn, proposals: list[Proposal], schema: str = "concordance") -> int:
    from .db import _safe_schema
    from .db.sync import _invalidate_definition_dependents

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        for p in proposals:
            cur.execute(f"""UPDATE {s}.word SET definition=%s, definition_source=%s,
                                part_of_speech=COALESCE(NULLIF(part_of_speech, ''), %s), updated_at=now()
                            WHERE id=%s AND coalesce(definition, '') = ''""",
                        (p.definition, p.source, p.part_of_speech, p.word_id))
            _invalidate_definition_dependents(cur, s, p.word_id)
    conn.commit()
    return len(proposals)
