"""Word origins: the language each word ultimately comes from, and its route
into English, for the Origins visualization (`concordance word-origins`).

Sources, tried in order for each active word (measured 2026-11-07 against a
blind hand-labelled sample: ~90% agree, most of the rest defensible):

  1. Wiktionary's structured etymology links (wikt.etym, every language,
     loaded by `wiktextract-morphology`), followed back from entry to entry.
     The ROOT is the earliest attested language: the walk stops at the first
     reconstructed (proto-) language, because past one the links wander
     sideways into its daughter languages (dunkel came out Old Saxon). A
     chain that dead-ends in an English stage ("greasy, from Middle English
     gresi") defers to the entry's own parts ("grease + -y").
  2. Pointer definitions ("Alternative form of X", "See X.") take X's origin.
  3. The 0 Dict's etymology, in Oxford shorthand ("ad. med.L. ...", "f. Gr.
     ..."): text after "cf." is cognates, not sources; "of uncertain origin"
     wins when it is stated up front (caviare is "of uncertain origin, found
     in Turkish ..."); the deepest language is picked by RANK, not position,
     because its scanned Greek is OCR noise that can contain a stray "F.".
  4. Derivatives the dictionaries skip take their base word's origin
     (heavenliness -> heavenly): affix roots, then a suffix/prefix peel with
     English spelling rules (y->i, doubled consonants, e restored).
  5. The 0 Dict's "f. prec. + -ly" formations (after the peel, which is surer).
  6. Compounds -- ONLY splits the compound judge verified (compound_verdict);
     a naive letter-boundary split was ~50% wrong.
  7. Local Wiktionary free text ("From Sanskrit ...").

What nothing knows stays unknown: no guessing (no model inference -- declined).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

ENGLISH_STAGES = {"en", "enm", "ang", "sco"}

# Root language families, in display order. Languages are matched by exact
# code, then Wiktionary's etymology-only codes, then by code prefix
# ("la-med" -> Latin, "grc-koi" -> Greek).
FAMILY_ORDER = [
    "Latin", "Greek", "Old & Middle English", "French", "Dutch, German & Frankish", "Norse & Scandinavian",
    "Celtic", "Italian", "Spanish & Portuguese", "Slavic", "Hebrew & Semitic", "Arabic, Persian & Turkic",
    "Indian languages", "East Asian", "Other", "Mixed", "Obscure origin", "Imitative",
]
_FAMILY_CODES = {
    "Latin": {"la", "itc-ola"},
    "Greek": {"grc", "el", "gkm", "qsb-grc"},
    "Old & Middle English": {"ang", "enm", "en", "sco"},
    "French": {"fr", "frm", "fro", "xno", "pro", "oc"},
    "Dutch, German & Frankish": {"nl", "dum", "odt", "de", "gmh", "goh", "nds", "gml", "yi", "osx", "frk", "gem", "fy", "ofs"},
    "Norse & Scandinavian": {"non", "is", "da", "sv", "no", "nb", "nn", "gmq", "fo"},
    "Celtic": {"ga", "sga", "mga", "gd", "cy", "wlm", "owl", "br", "kw", "gv", "cel", "cel-gau", "xcb"},
    "Italian": {"it", "roa-oit", "vec", "nap", "scn"},
    "Spanish & Portuguese": {"es", "osp", "pt", "roa-opt", "ca", "lad", "gl"},
    "Slavic": {"ru", "orv", "pl", "cs", "sk", "uk", "be", "sh", "bg", "cu", "sla", "sl", "mk"},
    "Hebrew & Semitic": {"he", "hbo", "arc", "syc", "sem", "akk", "phn", "gez", "egy", "cop", "mt"},
    "Arabic, Persian & Turkic": {"ar", "fa", "fa-cls", "pal", "peo", "ae", "xpr", "ota", "tr", "az", "uz", "tt", "ira", "ur", "ps"},
    "Indian languages": {"sa", "pi", "hi", "bn", "ta", "te", "mr", "ml", "kn", "gu", "pa", "inc", "pra", "si", "dra"},
    "East Asian": {"zh", "cmn", "yue", "ltc", "och", "ja", "ojp", "ko", "okm", "vi", "th", "ms", "id", "jv", "tl", "mn", "nan"},
}
_CODE_FAMILY = {c: f for f, cs in _FAMILY_CODES.items() for c in cs}
_ETY_ONLY = {"ML.": "Latin", "LL.": "Latin", "NL.": "Latin", "VL.": "Latin", "CL.": "Latin", "EL.": "Latin",
             "MGr.": "Greek", "LGr.": "Greek"}
_ETY_ONLY_NAMES = {"ML.": "Medieval Latin", "LL.": "Late Latin", "NL.": "New Latin", "VL.": "Vulgar Latin",
                   "CL.": "Classical Latin", "EL.": "Ecclesiastical Latin", "MGr.": "Medieval Greek",
                   "LGr.": "Late Greek"}
SPECIAL = {"*mixed*": "Mixed", "obscure": "Obscure origin", "imitative": "Imitative"}


def family(code: str) -> str:
    """The display family of a language code (or a special root)."""
    if code in SPECIAL:
        return SPECIAL[code]
    if code in _CODE_FAMILY:
        return _CODE_FAMILY[code]
    if code in _ETY_ONLY:
        return _ETY_ONLY[code]
    head = code.split("-")[0]
    return _CODE_FAMILY.get(head, "Other")


def is_proto(code: str) -> bool:
    return code.endswith("-pro")


# --- the 0 Dict's Oxford shorthand ---------------------------------------------

# Abbreviation -> representative code, and how DEEP a source each is: when a
# text names several, the deepest wins (Greek over Latin over French), not
# the last one -- scanned Greek words are OCR noise with stray capitals.
_OED_LANG = [
    (r"(?:med|mod|late|Anglo|eccl|scholastic|post-cl|vulgar|pop)\.\s?L\.|L\.L\.|\bL\.", "la"),
    (r"\bGr\.", "grc"),
    (r"\b(?:O|ON|A|mod|Norman |Central )?F\.|\bFr\.", "fr"),
    (r"\bOE\.|\bOld English", "ang"), (r"\bME\.", "enm"),
    (r"\bIt\.", "it"), (r"\bSp\.", "es"), (r"\bPg\.|\bPort\.", "pt"),
    (r"\bON\.|\bIcel\.|\bDa\.|\bSw\.|\bNorw\.|\bOSw\.", "non"),
    (r"\b(?:M|O)?H?G\.|\b(?:M)?Du\.|\b(?:M)?LG\.|\bFlem\.|\b(?:O)?Teut\.|\bGoth\.|\bOS\.", "de"),
    (r"\bArab\.", "ar"), (r"\bPers\.", "fa"), (r"\bTurk\.", "tr"),
    (r"\bHeb\.", "he"), (r"\bAram\.|\bSyr\.", "arc"),
    (r"\bSkr\.|\bSkt\.|\bHind\.|\bHindi\b|\bPali\b|\bTamil\b|\bBeng\.", "sa"),
    (r"\bChin\.|\bJap\.|\bMalay\b", "zh"),
    (r"\bCelt\.|\b(?:O|M)?Ir\.|\bGael\.|\bWelsh\b|\bCorn\.|\bBret\.", "ga"),
    (r"\bRuss\.|\bPol\.|\bBoh\.|\bSlav\.", "ru"),
]
_OED_DEPTH = ["grc", "he", "arc", "sa", "ar", "fa", "zh", "la", "ga", "non", "de", "ang", "fr", "it", "es", "pt",
              "tr", "ru", "enm"]
_OBSCURE = re.compile(r"\b(?:of (?:uncertain|doubtful|disputed) (?:origin|etym)|of obscure|obscure origin|"
                      r"origin (?:unknown|obscure|uncertain)|etym(?:ology|\.)? (?:unknown|obscure)|of unknown)", re.I)
_ECHOIC = re.compile(r"\b(?:echoic|imitative|onomatop)", re.I)
_CF = re.compile(r"\b[Cc]f\.|\bCompare\b")


def oed_languages(text: str) -> tuple[str | None, str | None]:
    """(deepest source language, nearest = first-named) in an Oxford-style
    etymology, ignoring everything after "cf." (cognates, not sources)."""
    t = _CF.split(text)[0]
    hits = sorted((m.start(), code) for pat, code in _OED_LANG for m in re.finditer(pat, t))
    if not hits:
        return None, None
    found = {c for _, c in hits}
    deepest = next((c for c in _OED_DEPTH if c in found), hits[-1][1])
    return deepest, hits[0][1]


_FROM_LANG = {
    "latin": "la", "ancient greek": "grc", "greek": "grc", "french": "fr", "old french": "fro",
    "middle english": "enm", "old english": "ang", "italian": "it", "spanish": "es", "portuguese": "pt",
    "old norse": "non", "german": "de", "dutch": "nl", "arabic": "ar", "persian": "fa", "turkish": "tr",
    "hebrew": "he", "sanskrit": "sa", "hindi": "hi", "japanese": "ja", "chinese": "zh", "irish": "ga",
    "scottish gaelic": "gd", "welsh": "cy", "russian": "ru", "yiddish": "yi", "malay": "ms",
}
SUFFIXES = sorted(set("ness ity ities ical ically ous ously ousness ive ively ist ism ize ise ization isation "
                      "able ible ably ibly ment ments al ally ary ery ory ish less lessly ful fully some like ward "
                      "wards wise hood ship dom er ers est ed ing ings ly s es y ic ics".split()), key=len, reverse=True)
PREFIXES = sorted("un re in im il ir dis pre post anti de over under up out sub super inter intra geo bio auto mis non "
                  "semi co counter trans hyper hypo micro macro pseudo proto neo poly mono multi fore be en em ex "
                  "after by off on with mid self half all well ill".split(), key=len, reverse=True)


def peel_bases(word: str) -> list[str]:
    """Candidate base words of a derivative, best first: suffix peels with the
    English spelling rules (heavenli-ness -> heavenly, nickell-ed -> nickel,
    rath-ed -> rathe before rath), then prefix peels."""
    w = word.lower()
    out: list[str] = []
    for s in SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 3:
            stem = w[: -len(s)]
            out += [stem + "e", stem] if s[0] in "aeiouy" else [stem, stem + "e"]
            if stem.endswith("i"):
                out.append(stem[:-1] + "y")
            if len(stem) > 3 and stem[-1] == stem[-2]:
                out.append(stem[:-1])
    for p in PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= 4:
            out.append(w[len(p):].lstrip("-"))
    seen, uniq = set(), []
    for b in out:
        if b and b != w and b not in seen:
            seen.add(b)
            uniq.append(b)
    return uniq


@dataclass(frozen=True)
class Origin:
    root: str          # language code of the earliest attested source, or *mixed* / obscure / imitative
    route: str         # inherited | direct | via:<family> | formed
    method: str        # which source decided it (see the module docstring)

    @property
    def family(self) -> str:
        return family(self.root)


class Resolver:
    """Resolves words to Origins from data loaded once (see load())."""

    def __init__(self, etym, oed_rows, wikt_text, definitions, affix_roots, compound_parts):
        self.etym = etym                      # (lang, term) -> (links [(lang, word)], parts)
        self.oed_by_id = {i: (h, e) for i, h, e in oed_rows}
        self.oed_first: dict[str, int] = {}
        for i, h, e in oed_rows:
            if e and h and h not in self.oed_first:
                self.oed_first[h] = i
        self.wikt_text = wikt_text            # term -> free-text etymology
        self.defs = definitions               # lemma -> definition
        self.affix = affix_roots              # term -> root
        self.cparts = compound_parts          # lemma_lc -> verified parts
        self.resolve = lru_cache(maxsize=None)(self._resolve)

    # -- Wiktionary chains -------------------------------------------------
    def _path(self, lang, word, depth=0, seen=None):
        """Languages from `lang` back toward the root. "*form*" marks a word
        formed from its own language's parts; "*mixed*" parts of different
        families."""
        seen = seen if seen is not None else set()
        if depth > 10 or (lang, word) in seen:
            return [lang]
        seen.add((lang, word))
        e = self.etym.get((lang, word))
        if not e:
            return [lang]
        links, parts = e
        if links:
            out = [lang] + [l for l, _ in links]
            l, w = links[-1]
            if w:
                out += self._path(l, w, depth + 1, seen)[1:]
            if not (parts and all(x in ENGLISH_STAGES or x.startswith("*") for x in out)):
                return out
        if parts:
            roots = [self._root(self._path(lang, p, depth + 1, seen)) for p in parts]
            roots = [r for r in roots if r]
            if roots:
                fams = {family(r) for r in roots}
                return [lang, "*form*", roots[0] if len(fams) == 1 else "*mixed*"]
        return [lang]

    @staticmethod
    def _root(path):
        cut = next((i for i, l in enumerate(path) if is_proto(l)), len(path))
        path = path[:cut] if cut else path
        named = [l for l in path if not l.startswith("*")]
        if path and path[-1] == "*mixed*":
            return "*mixed*"
        if len(named) > 1 or (named and named[-1] != "en"):
            return named[-1]
        return None

    @staticmethod
    def _route(path, root):
        # only the attested stretch counts (the same cut as _root): a proto-
        # language past Old English is not a route ("via Proto-West-Germanic")
        cut = next((i for i, l in enumerate(path) if is_proto(l)), len(path))
        path = path[:cut] if cut else path
        if "*form*" in path:
            return "formed"
        hop = next((l for l in path[1:] if not l.startswith("*") and l not in ENGLISH_STAGES), None)
        if hop is None:
            return "inherited"
        return "direct" if family(hop) == family(root) else f"via:{family(hop)}"

    # -- the cascade -------------------------------------------------------
    def _known(self, w):
        return ("en", w) in self.etym or w in self.oed_first or w in self.defs

    def _resolve(self, word, depth=0):
        if depth > 6:
            return None
        w = word.lower()
        key = ("en", word) if ("en", word) in self.etym else ("en", w)
        p = self._path(*key)
        r = self._root(p)
        if r:
            return Origin(r, self._route(p, r), "wiktionary")

        d = self.defs.get(word) or ""
        from .crossref import pointer_targets
        targets = sorted(pointer_targets(d))
        m = re.match(r"\s*See ([A-Za-z][A-Za-z'-]+)\.?\s*$", d)
        if m:
            targets.append(m.group(1))
        for t in targets:
            if t and t.lower() != w:
                o = self.resolve(t, depth + 1)
                if o:
                    return Origin(o.root, o.route, "pointer")

        formation = None
        if w in self.oed_first:
            o = self._from_oed(self.oed_first[w], depth)
            if o and o.method == "0dict":
                return o
            formation = o

        for base in [self.affix.get(w)] + peel_bases(w):
            if base and base != w and self._known(base):
                o = self.resolve(base, depth + 1)
                if o:
                    return Origin(o.root, "formed", "derivative")

        if formation:
            return formation

        parts = self.cparts.get(w)
        if parts:
            got = [self.resolve(x, depth + 1) for x in parts]
            if all(got):
                fams = {o.family for o in got}
                return Origin(got[-1].root if len(fams) == 1 else "*mixed*", "formed", "compound")

        t = self.wikt_text.get(w)
        if t:
            m = re.match(r"\s*(?:From|Borrowed from|Learned borrowing from|Ultimately from)\s+"
                         r"((?:[A-Z][a-z]+ ){0,2}[A-Z][a-z]+)", t)
            if m and m.group(1).lower() in _FROM_LANG:
                return Origin(_FROM_LANG[m.group(1).lower()], "direct", "wiktionary-text")
            m = re.match(r"\s*From\s+([a-z][a-z-]+)\b", t)
            if m and m.group(1) != w:
                o = self.resolve(m.group(1), depth + 1)
                if o:
                    return Origin(o.root, o.route, "wiktionary-text")
        return None

    def _from_oed(self, eid, depth):
        if depth > 6:
            return None
        _, text = self.oed_by_id[eid]
        if not text:
            return None
        if _OBSCURE.search(_CF.split(text)[0][:80]):
            return Origin("obscure", "direct", "0dict")
        deepest, nearest = oed_languages(text)
        if deepest:
            return Origin(deepest, "direct" if family(nearest) == family(deepest) else f"via:{family(nearest)}", "0dict")
        if _ECHOIC.search(text):
            return Origin("imitative", "direct", "0dict")
        roots = []
        for kind, v in self._oed_bases(text, eid):
            if kind == "id":
                h, t = self.oed_by_id[v]
                o = (self.resolve(h, depth + 1) if h else None) or (self._from_oed(v, depth + 1) if t else None)
            else:
                o = self.resolve(v, depth + 1)
            if o:
                roots.append(o)
        if roots:
            fams = {o.family for o in roots}
            return Origin(roots[0].root if len(fams) == 1 else "*mixed*", "formed", "0dict-formation")
        return None

    def _oed_bases(self, text, eid):
        """English bases of an Oxford "f. X + -y" formation; "next"/"prec."
        are the neighbouring entries."""
        t = _CF.split(text)[0]
        m = re.search(r"(?:\bf\.|[Ff]ormed (?:as|on|f\.)|\bfrom)\s+(.{1,60}?)(?:[.;:]\s|$)", t)
        if not m:
            return []
        out = []
        for part in re.split(r"\s*\+\s*|\s*,\s*", m.group(1)):
            p = part.strip().strip("*").strip()
            if p in ("next", "prec", "prec.", "next word"):
                nid = eid + (1 if p.startswith("next") else -1)
                if nid in self.oed_by_id:
                    out.append(("id", nid))
            elif re.fullmatch(r"[a-z][a-z'-]+", p) and not p.endswith("-") and not p.startswith("-"):
                out.append(("word", p))
        return out


def load(conn, schema: str, wikt_schema: str = "wikt", oed_schema: str = "oed",
         vocab_schema: str = "vocab") -> Resolver:
    """Build a Resolver from the database. Any source table that doesn't
    exist (a test schema, a fresh install) is simply an empty source."""
    from .db import _safe_schema
    s, g, o, v = (_safe_schema(x) for x in (schema, wikt_schema, oed_schema, vocab_schema))

    def table(name):
        cur.execute("SELECT to_regclass(%s)", (name,))
        return cur.fetchone()[0] is not None

    with conn.cursor() as cur:
        etym = {}
        if table(f"{g}.etym"):
            cur.execute(f"SELECT lang, term, links, parts FROM {g}.etym")
            for lang, term, links, parts in cur:
                etym[(lang, term)] = ([tuple(x.split(":", 1)) if ":" in x else (x, "") for x in links], parts)
        oed_rows = []
        if table(f"{o}.entry"):
            cur.execute(f"SELECT id, lower(headword_norm), etymology FROM {o}.entry ORDER BY id")
            oed_rows = cur.fetchall()
        wikt_text = {}
        if table(f"{v}.wiktionary"):
            cur.execute(f"SELECT lower(term), etymology FROM {v}.wiktionary WHERE coalesce(etymology, '') <> ''")
            for t, e in cur:
                wikt_text.setdefault(t, e)
        cur.execute(f"SELECT lemma, definition FROM {s}.word WHERE active")
        defs = dict(cur.fetchall())
        affix = {}
        if table(f"{g}.affix_root"):
            cur.execute(f"SELECT term, root FROM {g}.affix_root")
            affix = dict(cur.fetchall())
        cparts = {}
        if table(f"{s}.compound_verdict"):
            cur.execute(f"SELECT lemma_lc, parts FROM {s}.compound_verdict WHERE parts IS NOT NULL")
            cparts = dict(cur.fetchall())
    conn.commit()
    return Resolver(etym, oed_rows, wikt_text, defs, affix, cparts)


def language_names(conn, wikt_schema: str = "wikt") -> dict[str, str]:
    """Language code -> name (from the dump itself), plus the etymology-only codes."""
    from .db import _safe_schema
    g = _safe_schema(wikt_schema)
    names = dict(_ETY_ONLY_NAMES)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{g}.lang_name",))
        if cur.fetchone()[0] is not None:
            cur.execute(f"SELECT code, name FROM {g}.lang_name")
            names.update(dict(cur.fetchall()))
    conn.commit()
    return names


def compute(conn, schema: str, progress=print) -> dict:
    """`concordance word-origins`: resolve every active word and replace
    word_origin in one transaction. Deterministic and quick, so always a
    full recompute."""
    import time
    from collections import Counter
    from .db import _safe_schema
    s = _safe_schema(schema)
    t0 = time.time()
    resolver = load(conn, schema)
    names = language_names(conn)
    progress(f"  sources loaded ({time.time() - t0:.0f}s)")
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma FROM {s}.word WHERE active")
        words = cur.fetchall()
    rows, by_method = [], Counter()
    for wid, lemma in words:
        o = resolver.resolve(lemma)
        if o:
            by_method[o.method] += 1
            rows.append((wid, o.root, names.get(o.root, o.root) if o.root not in SPECIAL else SPECIAL[o.root],
                         o.family, o.route, o.method))
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {s}.word_origin")
        with cur.copy(f"COPY {s}.word_origin (word_id, root_lang, root_lang_name, family, route, method) "
                      f"FROM STDIN") as copy:
            for r in rows:
                copy.write_row(r)
    conn.commit()
    return {"words": len(words), "resolved": len(rows), "by_method": dict(by_method),
            "seconds": round(time.time() - t0, 1)}
