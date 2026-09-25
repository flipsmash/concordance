"""Definitions that point at another word instead of defining this one.

"Obsolete spelling of bread.", "Pronunciation spelling of better.", "Third-
person singular simple present indicative of think.", "Plural of goblin.",
"abbreviation of chimney pot." -- every parser of that cross-reference shape
lives here, built from the same pieces (leading "(label)" qualifiers, an
optional article), so a fix to one is visible next to all the others.

The parsers deliberately differ in how much they accept -- each consumer
makes a different decision from a match -- so they stay separate, named
patterns rather than one catch-all:

  mentions_pointer              anywhere in the text; quiz exclusion (broad)
  dialect_respelling_target     whole gloss is a dialect/eye-dialect respelling
  archaic_inflection_target     -eth/-est inflection gloss + word shape
  obsolete_spelling_target      whole gloss is an obsolete/archaic spelling
  classification_gloss          strip pointer clauses before judging meaning
  abbreviation_target           the dump's bare "abbreviation of X" stub
  plural_target                 a bare "plural of X"
"""

from __future__ import annotations

import re

# Leading "(archaic)" / "(Scotland, obsolete)" labels, then an optional article.
_LABELS = r"^\s*(?:\([^)]*\)\s*)*"
_ARTICLE = r"(?:(?:an?|the)\s+)?"


# --- anywhere in the text: "isn't real vocabulary content" ------------------

# Definitions that just point at another word (the "answer" isn't real
# vocabulary). Unanchored on purpose: a quiz can't use a word whose gloss
# contains a cross-reference anywhere, so this is far broader than the
# whole-gloss parsers below.
MENTIONS_POINTER_RE = re.compile(
    r"\b(form|spelling|inflection|tense|participle|plural|abbreviation|initialism) of\b"
    r"|\b(first|second|third)-person (singular|plural)\b", re.IGNORECASE)


def mentions_pointer(definition: str | None) -> bool:
    return bool(MENTIONS_POINTER_RE.search(definition or ""))


# --- the whole gloss is a respelling of another word -------------------------

# A definition that is ONLY a dialect/eye-dialect respelling cross-reference
# ("Pronunciation spelling of better.", "A dialectal form of folk."),
# optionally behind leading "(label)" qualifiers. "form of" only counts after
# dialect(al)/nonstandard -- "An informal form of address" (guvnor) is a real
# word's gloss, not a respelling. A definition that opens with a real sense
# and mentions a respelling later (nucular) deliberately doesn't match.
_DIALECT_DEF_RE = re.compile(
    _LABELS + _ARTICLE + r"(?:(?:obsolete|archaic|rare)\s+(?:or|and)\s+)?"
    r"(?:(?:eye[- ]dialect|pronunciation|non-?standard|informal|colloquial)\s+spelling"
    r"|(?:dialect(?:al)?|non-?standard)\s+(?:spelling|form))"
    r"\s+of\s+([^,.;:(\[]+)", re.IGNORECASE)


def dialect_respelling_target(definition: str | None) -> str | None:
    """The standard word a dialect/eye-dialect respelling points at (bettah ->
    "better", bimeby -> "by and by"), or None if `definition` isn't purely
    such a cross-reference. Detection is by the definition, never by the
    word's shape: a dropped-g pattern (-in for -ing) matched only real words
    live (survivin/securin are proteins, likin/gamin/matin real nouns)."""
    m = _DIALECT_DEF_RE.match(definition or "")
    if not m:
        return None
    target = m.group(1).strip().strip("'\"").lower()
    return target or None


# "(archaic) third-person singular simple present indicative of think" --
# Wiktionary's gloss for an archaic verb inflection (thinketh, findest).
_ARCHAIC_INFLECTION_DEF_RE = re.compile(
    _LABELS + r"(?:archaic\s+)?(?:second|third)-person singular\b[^;.]*?\bof\s+([a-z][a-z'-]*)",
    re.IGNORECASE)


def archaic_inflection_target(word: str, definition: str | None) -> str | None:
    """The base verb of an archaic -eth/-est inflection (thinketh -> "think",
    risest -> "rise"), by definition AND shape together: the definition must
    be Wiktionary's person/number inflection gloss and the word must carry the
    archaic ending. Shape alone is useless (hest, gest, queth, prest, teth are
    real words; worldliest/stickiest are ordinary superlatives)."""
    if not re.search(r"(?:eth|est|th|st)$", word.strip().lower()):
        return None
    m = _ARCHAIC_INFLECTION_DEF_RE.match(definition or "")
    return m.group(1).lower() if m else None


# "Obsolete spelling of bread.", "An obsolete form of spill.", "Archaic
# spelling of boulder." -- a word whose ENTIRE gloss is a pointer to its
# modern spelling (design rule 3). Leading "(label)" qualifiers allowed; a
# definition opening with a real sense (entiendo, remigate) doesn't match.
_OBSOLETE_SPELLING_DEF_RE = re.compile(
    _LABELS + _ARTICLE + r"(?:(?:obsolete|archaic|rare|dated)\s+(?:or|and)\s+)?"
    r"(?:obsolete|archaic)(?:\s+(?:or|and)\s+[a-z]+)?\s+(?:spelling|form)\s+of\s+([^,.;:(\[\u2014]+)",
    re.IGNORECASE)


def obsolete_spelling_target(definition: str | None) -> str | None:
    """The modern word an obsolete/archaic spelling points at (breade ->
    "bread", bowpot -> "bough pot"), or None."""
    m = _OBSOLETE_SPELLING_DEF_RE.match(definition or "")
    if not m:
        return None
    return m.group(1).strip().strip("'\"").lower() or None


# --- the gloss a classifier / embedder may see ------------------------------
#
# A definition that points at another spelling ("Variant spelling of faggot —
# A bundle of sticks...", "Obsolete form of intend.", "Plural of goblin.")
# must never let the TARGET word into semantic judgments about THIS word: the
# target can carry senses the pointer doesn't (faggot's slur sense got fagot
# tagged S3.2 "Relationship: Intimate/sexual" despite a bundle-of-sticks gloss
# and a sentence about fagots lit for Latimer and Ridley). Only the pointer's
# own gloss (after its dash) survives; a bare pointer contributes nothing.

_POINTER_QUALIFIER = (r"(?:alternative|variant|obsolete|archaic|dated|rare|non-?standard|informal|dialect(?:al)?"
                      r"|eye[- ]dialect|pronunciation|mis-?spelling|standard|british|american|us|uk|scottish"
                      r"|scots|older|early|later|contracted|clipped|irregular|poetic|historical|rare)")
_POINTER_RE = re.compile(
    _LABELS + _ARTICLE + r"(?:"
    rf"(?:{_POINTER_QUALIFIER}\s+(?:(?:or|and|,)\s*)?)+(?:spelling|form|variant)s?\s+of\b"
    r"|(?:variant|plural|misspelling|diminutive)\s+of\b"
    r"|(?:first|second|third)-person\b[^.;\u2014]{0,80}?\bof\b"
    r"|(?:simple\s+)?(?:past|present)\s+(?:tense|participle)\b[^.;\u2014]{0,40}?\bof\b"
    r"|(?:comparative|superlative)\s+(?:form\s+)?of\b"
    r")", re.IGNORECASE)
_GLOSS_DASH_RE = re.compile(r"\s(?:\u2014|\u2013|--|-)\s|:\s")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.;])\s+")


def classification_gloss(definition: str | None) -> str:
    """`definition` with every cross-reference to another spelling removed --
    the text semantic classification (and anything else judging THIS word's
    meaning) should see. A pointer clause with its own gloss keeps only the
    gloss; a bare pointer clause is dropped; other clauses pass through."""
    kept = []
    for clause in _CLAUSE_SPLIT_RE.split((definition or "").strip()):
        if not _POINTER_RE.match(clause):
            kept.append(clause)
            continue
        parts = _GLOSS_DASH_RE.split(clause, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            kept.append(parts[1].strip())
    return " ".join(kept).strip()


# --- the local Wiktionary dump's "abbreviation of X" stub --------------------

# The dump often glosses a term as a bare "abbreviation of X." -- true for
# real clippings (contemp -> contemporary) but just as often mislabels a
# plain spelling/case/compound-spacing variant (vizor -> visor, waterpower
# -> water power). localdict resolves X's own definition instead of trusting
# the label.
_ABBREVIATION_STUB_RE = re.compile(r"^abbreviation of\s+(.+?)\.?$", re.IGNORECASE)


def is_abbreviation_stub(definition: str | None) -> bool:
    return bool(_ABBREVIATION_STUB_RE.match((definition or "").strip()))


def abbreviation_target(definition: str | None) -> str | None:
    """X out of an "abbreviation of X[.]" stub, minus a trailing "(label)";
    None if `definition` isn't a stub ("" if it is one with nothing left)."""
    m = _ABBREVIATION_STUB_RE.match((definition or "").strip())
    if not m:
        return None
    return re.sub(r"\s*\([^)]*\)\s*$", "", m.group(1).strip())


# --- a bare "plural of X" ----------------------------------------------------

_PLURAL_OF_RE = re.compile(
    r"^(?:alternative |archaic |dialectal |obsolete )?plural (?:form )?of (\S+?)\.?$",
    re.IGNORECASE,
)


def plural_target(definition: str | None) -> str | None:
    """The singular X of a definition that is only "plural of X", else None."""
    m = _PLURAL_OF_RE.match((definition or "").strip())
    return m.group(1).strip(".,;").lower() if m else None
