"""IPA for a word no dictionary transcribes, built from the IPA of the word it
is made from (`concordance derive-ipa`).

Most of the vocabulary nothing pronounces is regular word-building on a word
something does pronounce: inflections (masticating, betokeneth), stress-
neutral affixes (ghosthood, bloomlessness, unmonklike), compounds (raincape)
and respellings ("Obsolete spelling of shapeless"). For those the sound is
the base's IPA plus the affix's, with English's regular junction rules:

  - -s/-es and -ed take their voicing from the base's last sound
    (/s z ɪz/, /t d ɪd/);
  - spelling changes at the seam are undone to find the base (stop+p+ing,
    happ+i+ly, masticat(e)+ing, gent(le)+ly), and -le bases lose their
    syllabic l before a vowel (tickl̩ -> ticklɪŋ);
  - the base keeps its stress; a one-syllable base gets the primary stress
    once the word has more than one syllable; a compound is stressed on its
    first part, its second part demoted to secondary (ˈreɪnˌkeɪp).

Only affixes that leave the base's stress alone are used. -ity, -ic, -ion,
-ian, -ious and the like move the stress (ˈsɪvɪl -> sɪˈvɪlɪti) and aren't
derivable this way, so words built with them are left alone. Peeling an
affix off by spelling alone is safe for the unambiguous ones (-ness,
-hood, un-); the rest (-y, -ist, in-, be-, ...) need Wiktionary's etymology
to name the same root, so "corner" is never corn + -er.

Bases come from word.ipa (real sources only) and Wiktionary's sound table
(General American preferred), and can themselves be derived, two levels
deep (bloomlessness <- bloomless <- bloom). The result is written with
ipa_source 'derived' and word.ipa_derived_from saying how ("bloomless +
-ness"); a real transcription arriving later replaces it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .db import _safe_schema

VOWELS = "aeiouæɑɒɔəɛɜɪʊʌɐɚɝyøœɘɵɤɯ"
_VOWEL_RUN = re.compile(f"[{VOWELS}]+|.̩")       # a nucleus, or a syllabic consonant
VOICELESS = set("ptkfθsʃ") | {"tʃ"}
SIBILANTS = {"s", "z", "ʃ", "ʒ", "tʃ", "dʒ"}
MAX_DEPTH = 2

# suffix spelling -> (IPA, needs corroboration from Wiktionary's etymology)
# All stress-neutral. -s/-es/-ed are handled by rule (_inflect), not here.
SUFFIXES: dict[str, tuple[str, bool]] = {
    "ness": ("nəs", False), "less": ("ləs", False), "lessly": ("ləsli", False),
    "lessness": ("ləsnəs", False), "fully": ("fəli", False), "ful": ("fəl", False),
    "fulness": ("fəlnəs", False), "ly": ("li", False), "ment": ("mənt", False),
    "ship": ("ʃɪp", False), "hood": ("hʊd", False), "dom": ("dəm", False),
    "like": ("ˌlaɪk", False), "wise": ("ˌwaɪz", False), "ward": ("wərd", False),
    "wards": ("wərdz", False), "ing": ("ɪŋ", False), "ingly": ("ɪŋli", False),
    "er": ("ər", True), "est": ("ɪst", False), "eth": ("ɪθ", False), "ish": ("ɪʃ", False),
    "ling": ("lɪŋ", False), "some": ("səm", False), "proof": ("ˌpruf", False),
    "worthy": ("ˌwərði", False), "ally": ("li", False), "y": ("i", True),
    "ist": ("ɪst", True), "ism": ("ɪzəm", True), "able": ("əbəl", True),
    "ize": ("aɪz", True), "ise": ("aɪz", True), "en": ("ən", True), "kin": ("kɪn", True),
    "let": ("lət", True), "ster": ("stər", True), "ee": ("ˈi", True), "ette": ("ˈɛt", True),
    "ry": ("ri", True), "ery": ("əri", True), "age": ("ɪdʒ", True), "ess": ("ɪs", True),
}
# suffixes beginning with a vowel sound: the base may have lost a silent e,
# doubled its last consonant, or (for -le bases) its syllabic l
_VOWEL_SUFFIXES = {"ing", "ingly", "er", "est", "eth", "ish", "y", "ist", "ism", "able",
                   "ize", "ise", "en", "ee", "ette", "ery", "age", "ess", "ed", "es"}

# prefix -> (IPA, needs corroboration). Polysyllabic ones carry a secondary stress.
PREFIXES: dict[str, tuple[str, bool]] = {
    "un": ("ʌn", False), "non": ("ˌnɑn", False), "re": ("ˌri", True), "pre": ("ˌpri", True),
    "mis": ("ˌmɪs", False), "dis": ("dɪs", True), "out": ("ˌaʊt", False),
    "over": ("ˌoʊvər", False), "under": ("ˌʌndər", False), "counter": ("ˌkaʊntər", False),
    "anti": ("ˌænti", False), "semi": ("ˌsɛmi", False), "super": ("ˌsupər", False),
    "sub": ("ˌsʌb", True), "inter": ("ˌɪntər", True), "fore": ("ˌfɔr", False),
    "self": ("ˌsɛlf", False), "up": ("ˌʌp", True), "be": ("bɪ", True), "a": ("ə", True),
    "en": ("ɪn", True), "em": ("ɪm", True), "in": ("ɪn", True), "im": ("ɪm", True),
    "il": ("ɪl", True), "ir": ("ɪr", True), "hyper": ("ˌhaɪpər", False),
    "multi": ("ˌmʌlti", False), "mid": ("ˌmɪd", True), "pseudo": ("ˌsudoʊ", False),
    "ultra": ("ˌʌltrə", False), "extra": ("ˌɛkstrə", True), "co": ("ˌkoʊ", True),
    "de": ("ˌdi", True), "post": ("ˌpoʊst", True), "proto": ("ˌproʊtoʊ", False),
    "quasi": ("ˌkweɪzaɪ", False), "tri": ("ˌtraɪ", True), "bi": ("ˌbaɪ", True),
    "arch": ("ˌɑrtʃ", True), "after": ("ˌæftər", False), "with": ("wɪð", True),
    "mal": ("ˌmæl", True), "neo": ("ˌnioʊ", True), "sur": ("sər", True),
}
MIN_BASE = 3
_AFFIX_LIKE = set(SUFFIXES) | set(PREFIXES) | {
    "ian", "tor", "ite", "ion", "ic", "ical", "ity", "ous", "ious", "al", "ary", "ory", "ate", "ive",
    "ure", "ance", "ence", "ant", "ent", "oid", "ose", "ine", "ide", "ium", "ia", "logy", "ology",
    # Greek combining forms move the stress (fəˈtɑmɪtɚ, not ˈfoʊtoʊˌmitɚ)
    "meter", "metre", "metry", "graphy", "nomy", "pathy", "phagy", "cracy", "phony", "scopy", "logist"}


# --- IPA strings ----------------------------------------------------------

_DELIMITED = re.compile(r"[/\[]([^/\[\]]+)[/\]]")
# narrow-transcription marks the synthesizer has no use for (non-syllabic,
# voiceless, aspirated, dental, velarized, nasalized, ...); syllabic ̩ and
# length ː stay
_NARROW = re.compile("[\u0300-\u0328\u032a-\u036fʰʲʷˠˤ‿]")


def clean(ipa: str | None) -> str | None:
    """One phonemic transcription, plain: the first of several ("/x/ [y]"),
    its first variant, without slashes, tie bars, syllable dots, spaces or
    narrow diacritics."""
    if not ipa:
        return None
    text = ipa.strip()
    m = _DELIMITED.search(text)
    if m:
        text = m.group(1)
    text = re.split(r"[,;~]", text)[0].strip()
    text = text.replace("\u0361", "").replace("\u035c", "")
    text = _NARROW.sub("", text).replace(".", "").replace(" ", "").replace("g", "ɡ")
    text = text.replace("ɫ", "l")
    if not text or "-" in text or "…" in text or "/" in text:
        return None
    return text


def syllables(ipa: str) -> int:
    return len(_VOWEL_RUN.findall(ipa))


def _stressed(ipa: str) -> bool:
    return "ˈ" in ipa


def _final_phoneme(ipa: str) -> str:
    bare = ipa.rstrip("ːˑ̩")
    for two in ("tʃ", "dʒ"):
        if bare.endswith(two):
            return two
    return bare[-1:] if bare else ""


def _drop_syllabic_l(ipa: str) -> str:
    """ˈtɪkl̩ / ˈtɪkəl -> ˈtɪkl, before a vowel-initial suffix."""
    if ipa.endswith("l̩"):
        return ipa[:-2] + "l"
    if ipa.endswith("əl"):
        return ipa[:-2] + "l"
    return ipa


def _join(base: str, suffix_ipa: str) -> str:
    out = base + suffix_ipa
    if not _stressed(base) and syllables(out) > 1 and not suffix_ipa.startswith("ˈ"):
        out = "ˈ" + out
    if suffix_ipa.startswith("ˈ"):       # -ee, -ette take the stress themselves
        out = base.replace("ˈ", "ˌ") + suffix_ipa
    return out


def _prefix(prefix_ipa: str, base: str) -> str:
    if not _stressed(base):
        base = "ˈ" + base
    return prefix_ipa + base


def add_s(base: str) -> str:
    last = _final_phoneme(base)
    return _join(base, "ɪz" if last in SIBILANTS else "s" if last in VOICELESS else "z")


def add_ed(base: str) -> str:
    last = _final_phoneme(base)
    return _join(base, "ɪd" if last in ("t", "d") else "t" if last in VOICELESS else "d")


_RULE_SUFFIXES = {"es": add_s, "s": add_s, "ed": add_ed}



def _zipf(word: str) -> float:
    from wordfreq import zipf_frequency
    return zipf_frequency(word, "en")


def compound(parts_ipa: list[str]) -> str:
    """First part keeps (or gets) the primary stress; the rest are demoted."""
    first = parts_ipa[0] if _stressed(parts_ipa[0]) else "ˈ" + parts_ipa[0]
    rest = []
    for p in parts_ipa[1:]:
        p = p.replace("ˈ", "ˌ")
        rest.append(p if "ˌ" in p or syllables(p) == 0 else "ˌ" + p)
    return first + "".join(rest)


# --- finding the base -------------------------------------------------------

def _base_spellings(stem: str, suffix: str) -> list[str]:
    """Spellings the base may have had before `suffix` was added to it."""
    out = [stem]
    vowel = suffix in _VOWEL_SUFFIXES
    if vowel:
        out.append(stem + "e")                                  # masticat(e)+ing
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiouls":
            out.append(stem[:-1])                               # stop(p)+ing
        if len(stem) > 2 and stem[-1] == stem[-2] == "l":
            out.append(stem[:-1])                               # frivol(l)+ing (British)
    if stem.endswith("i"):
        out.append(stem[:-1] + "y")                             # happ(i)+ly, empt(i)+ness
    if suffix == "ly":
        out.append(stem + "le")                                 # gent(le)+ly, perdurab(le)+ly
    if suffix == "ally":
        out = [stem] if stem.endswith("ic") else []
    return out


@dataclass
class Derivation:
    ipa: str
    how: str          # "bloomless + -ness"
    depth: int


class Deriver:
    def __init__(self, lexicon: dict[str, str], roots: dict[str, str] | None = None,
                 compounds: dict[str, list[str]] | None = None):
        self.lexicon = lexicon            # word -> clean IPA (real sources)
        self.roots = roots or {}          # Wiktionary affix_root: word -> root
        self.compounds = compounds or {}  # word -> parts
        self._memo: dict[tuple[str, int], Derivation | None] = {}
        self.trusted: set[str] = set(lexicon)

    def ipa_of(self, word: str, depth: int) -> Derivation | None:
        """A base's IPA: real if known, else derived (up to MAX_DEPTH)."""
        if word in self.lexicon:
            return Derivation(self.lexicon[word], word, 0)
        if depth >= MAX_DEPTH:
            return None
        d = self.derive(word, depth + 1)
        return d

    def _corroborated(self, word: str, base: str) -> bool:
        root = self.roots.get(word)
        return bool(root) and root == base

    def derive(self, word: str, depth: int = 0, *, base_hint: str | None = None) -> Derivation | None:
        word = word.strip().lower()
        key = (word, depth)
        if key in self._memo and base_hint is None:
            return self._memo[key]
        result = (self._from_compound(word, depth) or self._from_suffix(word, depth, base_hint)
                  or self._from_prefix(word, depth, base_hint))
        if base_hint is None:
            self._memo[key] = result
        return result

    def _suffix_analyses(self, word: str, base_hint: str | None):
        """(suffix, base) splits of `word`, longest suffix first."""
        out = []
        for suffix in sorted(list(SUFFIXES) + list(_RULE_SUFFIXES), key=len, reverse=True):
            if not word.endswith(suffix) or len(word) - len(suffix) < MIN_BASE:
                continue
            stem = word[: -len(suffix)]
            if suffix == "s" and stem.endswith(("s", "u", "i")):
                continue
            risky = SUFFIXES[suffix][1] if suffix in SUFFIXES else False
            for base in _base_spellings(stem, suffix):
                if base_hint and base != base_hint:
                    continue
                if risky and not base_hint and not self._corroborated(word, base):
                    continue
                if suffix == "ling" and base.endswith("l"):
                    continue            # cudgel(l)+ing, not cudgel + -ling
                out.append((suffix, base))
        return out

    def _attach(self, word: str, suffix: str, base: str, got: Derivation) -> Derivation:
        b = got.ipa
        if suffix in _VOWEL_SUFFIXES and base.endswith("le"):
            b = _drop_syllabic_l(b)
        if suffix in ("ly", "ness") and base.endswith("ed") and re.search(r"[^ɪə][td]$", b):
            b = b[:-1] + "ɪd"                                 # composedly: -ed is a syllable before -ly/-ness
        if suffix == "ly" and base.endswith("y") and b.endswith("i"):
            b = b[:-1] + "ɪ"                                  # huffy -> huffɪly
        if suffix in _RULE_SUFFIXES:
            ipa = _RULE_SUFFIXES[suffix](b)
        elif suffix == "ly" and base.endswith("le") and word == base[:-2] + "ly":
            ipa = _join(_drop_syllabic_l(b), "i")              # gent(le)+ly: gentl + i
        else:
            ipa = _join(b, SUFFIXES[suffix][0])
        return Derivation(ipa, f"{got.how} + -{suffix}", got.depth + 1)

    def _from_suffix(self, word: str, depth: int, base_hint: str | None) -> Derivation | None:
        analyses = self._suffix_analyses(word, base_hint)
        # a base that is a real word beats one that only spells like a stem
        # (superseder is supersede + -er, not super + -s + -ed + -er); between
        # real words, the commoner one (soused: souse, not sous)
        real = [(suf, base) for suf, base in analyses if base in self.lexicon]
        if real:
            suf, base = max(real, key=lambda a: (_zipf(a[1]), len(a[1])))
            return self._attach(word, suf, base, Derivation(self.lexicon[base], base, 0))
        if depth >= MAX_DEPTH:
            return None
        for suf, base in analyses:
            got = self.derive(base, depth + 1)
            if got:
                return self._attach(word, suf, base, got)
        return None

    def _from_prefix(self, word: str, depth: int, base_hint: str | None) -> Derivation | None:
        for prefix in sorted(PREFIXES, key=len, reverse=True):
            if not word.startswith(prefix):
                continue
            base = word[len(prefix):].lstrip("-")
            if len(base) < max(MIN_BASE, 4):
                continue
            if base_hint and base != base_hint:
                continue
            pipa, risky = PREFIXES[prefix]
            if risky and not base_hint and not self._corroborated(word, base):
                continue
            got = self.ipa_of(base, depth)
            if got:
                return Derivation(_prefix(pipa, got.ipa), f"{prefix}- + {got.how}", got.depth + 1)
        return None

    def _from_compound(self, word: str, depth: int) -> Derivation | None:
        parts = self.compounds.get(word)
        if not parts or len(parts) < 2:
            return None
        if re.sub(r"[\s-]", "", "".join(parts)).lower() != re.sub(r"[\s-]", "", word):
            return None
        # free words only: a "part" that is really an affix (persona + tor,
        # geometric + ian) isn't a compound, and its sound isn't the word's
        if any(len(p) < 3 or p.lower() in _AFFIX_LIKE for p in parts):
            return None
        got = [self.ipa_of(p.lower(), depth) for p in parts]
        if not all(got):
            return None
        return Derivation(compound([g.ipa for g in got]), " + ".join(g.how for g in got),
                          max(g.depth for g in got) + 1)


# --- definitions that name the base -------------------------------------------

_SPELLING_KINDS = (r"(?:obsolete|archaic|alternative|alternate|dated|rare|rarer|nonstandard|non-standard|"
                   r"eye[- ]dialect|uncommon|variant|british|american|us|uk|canadian|australian|"
                   r"standard|old|older|former|superseded|less|more|common|chiefly|mostly|"
                   r"a|an|the|and|or|of|mis)")
_SPELLING = re.compile(
    r"^(?:" + _SPELLING_KINDS + r"[ ,]+){0,5}spelling of\s+([a-z][a-z' -]*[a-z])(?=$|[\s.;,(:])", re.I)
_NOT_SAME_SOUND = re.compile(r"pronunciation spelling|dialectal|phonetic", re.I)
_INFLECTION = re.compile(
    r"^(?:[a-z-]+[ ,]+){0,4}?(?:present participle|past participle|simple past|past tense|plural|"
    r"third[- ]person singular|second[- ]person singular|comparative|superlative)"
    r"(?:[a-z ,-]+?)? of\s+([a-z][a-z'-]+)", re.I)


def base_from_definition(definition: str | None) -> tuple[str, str] | None:
    """('spelling', X) for a respelling of X -- same sound -- or
    ('inflection', X) for an inflected form of X; else None."""
    d = (definition or "").strip()
    if not d:
        return None
    m = _SPELLING.match(d)
    if m and not _NOT_SAME_SOUND.search(d[: m.end()]):
        return "spelling", m.group(1).lower()
    m = _INFLECTION.match(d)
    if m:
        return "inflection", m.group(1).lower()
    return None


def _skeleton(word: str) -> str:
    """The consonants a spelling spells, roughly: respellings that keep it
    (shapelesse, vertue, aweful) sound the same; ones that don't
    (piratic/piratical, pisky/pixie) are different forms."""
    w = re.sub(r"[\s'-]", "", word.lower())
    w = re.sub(r"e$", "", w)
    for a, b in (("ph", "f"), ("ck", "k"), ("c", "k"), ("q", "k"), ("x", "ks"), ("z", "s"), ("y", "i"), ("w", "u")):
        w = w.replace(a, b)
    w = re.sub(r"(.)\1+", r"\1", w)
    return re.sub(r"[aeiou]", "", w)


def derive_word(deriver: Deriver, lemma: str, definition: str | None) -> Derivation | None:
    word = lemma.strip().lower()
    hint = base_from_definition(definition)
    if hint and hint[1] != word:
        kind, base = hint
        if kind == "spelling" and _skeleton(base) != _skeleton(word):
            hint = None                 # piratic is not piratical, vertice not vertex: different sounds
        elif kind == "spelling":
            words = base.split()
            if len(words) > 1:          # "spelling of above ground": its words, run together
                got = [deriver.ipa_of(w, 0) for w in words]
                if all(got):
                    return Derivation(compound([g.ipa for g in got]), f"respelling of {base}",
                                      max(g.depth for g in got) + 1)
                return None
            got = deriver.ipa_of(base, 0)
            if got:
                return Derivation(got.ipa, f"respelling of {got.how}", got.depth + 1)
        else:
            d = deriver.derive(word, 0, base_hint=base)
            if d:
                return d
    return deriver.derive(word, 0)


# --- loading --------------------------------------------------------------------

def _pick(entries: list[dict]) -> str | None:
    """The transcription most of the term's entries agree on (folded), so a
    homograph's odd one out -- "sob" also holds S.O.B. /ˌɛsoʊˈbi/ -- loses to
    the word's own; within that, General American and phonemic /.../ first."""
    cands = []
    for e in entries or []:
        c = clean(e.get("ipa"))
        if c:
            tags = e.get("tags") or []
            ga = any(t in ("General-American", "US") for t in tags)
            cands.append((c, ga, (e.get("ipa") or "").lstrip().startswith("[")))
    if not cands:
        return None
    votes = Counter(fold(c) for c, _, _ in cands)
    best = max(votes.values())
    winners = [x for x in cands if votes[fold(x[0])] == best]
    winners.sort(key=lambda x: (x[2], not x[1]))
    return winners[0][0]


def load(conn, schema: str) -> Deriver:
    from . import audio
    s = _safe_schema(schema)
    lexicon: dict[str, str] = {}
    with conn.cursor() as cur:
        # the corpus's own transcriptions from a known source (0 Dict, Wordnik,
        # MW, dictionary.com...) first; Wiktionary's English sound table next;
        # legacy source-less rows only where neither has the word (some of them
        # are a foreign word's own transcription)
        cur.execute(f"""SELECT lower(lemma), ipa, ipa_source FROM {s}.word
                        WHERE coalesce(ipa, '') <> '' AND coalesce(ipa_source, '') <> 'derived'""")
        corpus = cur.fetchall()
        for term, ipa, source in corpus:
            c = clean(ipa) if source else None
            if c and audio.looks_like_english_ipa(c):
                lexicon[term] = c
        cur.execute("SELECT term, ipa FROM wikt.sound WHERE jsonb_array_length(ipa) > 0")
        for term, entries in cur.fetchall():
            c = _pick(entries)
            if c and audio.looks_like_english_ipa(c):
                lexicon.setdefault(term.lower(), c)
        trusted = set(lexicon)
        for term, ipa, source in corpus:
            c = clean(ipa) if not source else None
            if c and audio.looks_like_english_ipa(c):
                lexicon.setdefault(term, c)
        cur.execute("SELECT term, root FROM wikt.affix_root")
        roots = {t.lower(): r.lower() for t, r in cur.fetchall()}
        cur.execute("SELECT term, parts FROM wikt.compound")
        compounds = {t.lower(): p for t, p in cur.fetchall()}
        cur.execute(f"SELECT lemma_lc, parts FROM {s}.compound_verdict WHERE parts IS NOT NULL")
        for t, p in cur.fetchall():
            compounds.setdefault(t, p)
    deriver = Deriver(lexicon, roots, compounds)
    deriver.trusted = trusted
    return deriver


def candidates(conn, schema: str) -> list[tuple[int, str, str | None]]:
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT id, lemma, definition FROM {s}.word
                        WHERE active AND coalesce(ipa, '') = '' ORDER BY id""")
        return cur.fetchall()


def run(conn, schema: str, *, limit: int = 0, dry_run: bool = False, log=print) -> dict:
    """Derive IPA for every active word with none; write it as 'derived'."""
    from . import audio
    s = _safe_schema(schema)
    deriver = load(conn, schema)
    rows = candidates(conn, schema)
    if limit:
        rows = rows[:limit]
    stats: Counter = Counter(candidates=len(rows))
    samples = []
    with conn.cursor() as cur:
        for wid, lemma, definition in rows:
            d = derive_word(deriver, lemma, definition)
            if not d or not audio.looks_like_english_ipa(d.ipa):
                stats["not_derivable"] += 1
                continue
            stats["derived"] += 1
            kind = ("respelling" if d.how.startswith("respelling") else
                    "compound" if " + " in d.how and "-" not in d.how else "affix")
            stats[kind] += 1
            if len(samples) < 15:
                samples.append(f"{lemma}: {d.ipa}  ({d.how})")
            if not dry_run:
                cur.execute(f"""UPDATE {s}.word SET ipa = %s, ipa_source = 'derived', ipa_derived_from = %s
                                WHERE id = %s""", (d.ipa, d.how, wid))
    if not dry_run:
        conn.commit()
    for line in samples:
        log("  " + line)
    return dict(stats)


# --- blind check -----------------------------------------------------------------

_FOLD = [("ɚ", "ər"), ("ɝ", "ər"), ("ɜː", "ər"), ("ɜ", "ər"), ("ɹ", "r"), ("ɾ", "t"),
         ("əʊ", "oʊ"), ("ɪə", "ɪr"), ("ɛə", "ɛr"), ("eə", "ɛr"), ("ʊə", "ʊr"), ("ɔə", "ɔr"),
         ("ː", ""), ("ˑ", ""), ("\u0329", ""), ("(", ""), (")", ""), ("ʔ", ""), ("ʍ", "w"), ("hw", "w"),
         ("aɪ", "Y"), ("ɑɪ", "Y"), ("ʌɪ", "Y"), ("aʊ", "W"), ("æʊ", "W"), ("eɪ", "A"), ("ɛɪ", "A"),
         ("oʊ", "O"), ("ɔɪ", "B"),
         ("ɒ", "ɑ"), ("ɔ", "ɑ"), ("a", "æ"), ("e", "ɛ"), ("o", "O"), ("ɐ", "ə"), ("ɨ", "ə"),
         ("ɪ", "ə"), ("ʌ", "ə"), ("ʊ", "u"), ("ər", "ə"), ("rr", "r")]


def fold(ipa: str) -> str:
    """For comparing transcriptions across dialect/notation differences:
    no stress or length, no r except before a vowel (RP vs GA), no yod after
    t/d/n/s/l (tjuː vs tu), GA diphthongs, the reduced vowels (ə ɪ ʌ)
    treated alike."""
    t = ipa.replace("ˈ", "").replace("ˌ", "")
    t = re.sub(r"f(ʊ|ə)?l\u0329?(?=$|nəs|nɪs|i|li)", "fəl", t)       # -ful: fəl / fʊl / fl̩
    t = re.sub(r"ɪ(?=$|li$)", "i", t)                                 # happY: final ɪ ~ i
    t = re.sub(r"[iɪ](?=n[əɪ]s$)", "ə", t)                            # -iness: i ~ ɪ
    for a, b in _FOLD:
        t = t.replace(a, b)
    t = re.sub(r"r(?![æɑɛəiuYWAOB])", "", t)
    t = re.sub(r"(?<=[tdnslθz])j(?=u)", "", t)
    t = re.sub(r"ə?i$", "i", t)
    return t


def _edits(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate(conn, schema: str, sample: int = 3000, log=print) -> dict:
    """Derive IPA for words that HAVE a real transcription, with that word
    hidden from the lexicon, and compare (stress-blind, dialect-folded)."""
    import random

    s = _safe_schema(schema)
    deriver = load(conn, schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT lemma, definition FROM {s}.word
                        WHERE coalesce(ipa, '') <> '' AND coalesce(ipa_source, '') <> 'derived'""")
        corpus = cur.fetchall()
    pool = [(lemma.lower(), d) for lemma, d in corpus if lemma.lower() in deriver.trusted]
    random.Random(7).shuffle(pool)
    stats: Counter = Counter()
    by_kind: dict[str, Counter] = {}
    worst = []
    for lemma, definition in pool:
        if stats["derived"] >= sample:
            break
        truth = deriver.lexicon.pop(lemma)
        deriver._memo.clear()
        d = derive_word(deriver, lemma, definition)
        deriver.lexicon[lemma] = truth
        stats["tried"] += 1
        if not d:
            continue
        stats["derived"] += 1
        a, b = fold(d.ipa), fold(truth)
        e = _edits(a, b)
        kind = d.how.split(" + ")[-1] if " + " in d.how else d.how.split(" ")[0]
        k = by_kind.setdefault(kind if kind.startswith("-") or kind.endswith("-") else
                               ("respelling" if kind == "respelling" else "compound/prefix"), Counter())
        k["n"] += 1
        if e == 0:
            stats["exact"] += 1
            k["exact"] += 1
        elif e <= 1:
            stats["one_off"] += 1
            k["one_off"] += 1
        else:
            worst.append((e, lemma, d.ipa, truth, d.how))
        stress_ok = (d.ipa.replace("ˌ", "").find("ˈ") >= 0) == (truth.find("ˈ") >= 0)
        stats["stress_marked_agrees"] += stress_ok
    log(f"tried {stats['tried']}, derived {stats['derived']}: {stats['exact']} exact, "
        f"{stats['one_off']} one sound off (folded)")
    for kind, k in sorted(by_kind.items(), key=lambda kv: -kv[1]["n"])[:25]:
        log(f"  {kind:14} n={k['n']:5}  exact {k['exact'] / k['n']:.0%}  ≤1 off {(k['exact'] + k['one_off']) / k['n']:.0%}")
    worst.sort(reverse=True)
    for e, lemma, got, truth, how in worst[:40]:
        log(f"  ✗ {lemma}: derived {got} vs {truth}  ({how})")
    return dict(stats)
