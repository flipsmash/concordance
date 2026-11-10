"""Readable names for word.definition_source.

That column holds whatever each pipeline wrote over the years -- "Local
Wiktionary (DB)", "OED" and "oed", "mw", "Merriam-Webster API" and
"merriam_webster", "dm", "Wordnik (century)", and composites like
"Local Wiktionary (DB) (synonym of 'odoriferous')" or "derived (stringless +
-ly)". describe_definition_source turns one into a source name a reader
recognizes plus, when the definition was borrowed or built, a short note
saying how. The raw value stays in the column (and the API) unchanged.
"""

from __future__ import annotations

import re

# lowercased raw value -> display name
_NAMES = {
    "local wiktionary (db)": "Wiktionary",
    "local wiktionary": "Wiktionary",
    "wiktionary": "Wiktionary",
    "wordnik (wiktionary)": "Wiktionary",
    "oed": "0 Dict",
    "mw": "Merriam-Webster",
    "merriam-webster api": "Merriam-Webster",
    "merriam_webster": "Merriam-Webster",
    "merriam-webster (scraped)": "Merriam-Webster",
    "free dictionary api": "Free Dictionary",
    "datamuse": "Datamuse",
    "dm": "Datamuse",
    "wordnik (century)": "Century Dictionary",
    "century": "Century Dictionary",
    "wordnik (gcide)": "GCIDE",
    "wordnik (ahd-5)": "American Heritage Dictionary",
    "wordnik (wordnet)": "WordNet",
    "wordnet": "WordNet",
    "phrontistery": "The Phrontistery",
    "collins": "Collins",
    "apininja": "API Ninjas",
    "yourdictionary.com": "YourDictionary",
    "web (llm-extracted)": "Web search",
    "vocab.defined import": "Earlier vocabulary project",
    "admin (hand-written)": "Written by an admin",
}
# placeholders on words that never got a definition
_NO_SOURCE = {"", "corpus", "dictionary"}

_SYNONYM = re.compile(r"^(.*?)\s*\(synonym of '(.+)'\)\s*$")
_DERIVED = re.compile(r"^derived \((.+)\)\s*$")


def describe_definition_source(raw: str | None) -> tuple[str | None, str | None]:
    """(display name, note) for a raw definition_source; (None, None) when
    there's nothing to credit. The note says how a borrowed or built
    definition came about ("from its synonym “odoriferous”")."""
    text = (raw or "").strip()
    if text.lower() in _NO_SOURCE:
        return None, None
    if m := _SYNONYM.match(text):
        name, _ = describe_definition_source(m.group(1))
        return name, f"the definition of its synonym “{m.group(2)}”"
    if m := _DERIVED.match(text):
        inner = m.group(1).strip()
        if inner.startswith("inflection of "):
            return "Derived", f"a form of “{inner[len('inflection of '):]}”"
        return "Derived", f"built from {inner}"
    if text.lower().startswith("web (manual)"):
        rest = text.split(":", 1)[1].strip() if ":" in text else ""
        return "Web (added by hand)", rest or None
    return _NAMES.get(text.lower(), text), None
