"""Analogy edge/anchor selection (§ analogies).

Pure SQL-reading module -- no LLM, no live WordNet calls. Everything it needs
(verified relation edges, the full-WordNet fanout used for the ambiguity
exclusion set, the anchor bank of ordinary-term edges) was precomputed by the
concordance/analogies.py backfill. Kept separate from webapp/backend/quiz.py
(which stays a thin per-question-type assembler, same role it already plays
for mc/true_false/matching) and from concordance/distractors.py (which is
specifically about picking wrong OPTION words, not about picking which
relation EDGE to test).

Storage/slot convention (fixed by concordance/analogies.py, consumed here):
every word_relation_edge row is directional, term_a -> term_b, with term_a
always the "specific" (many-to-one-testable) side for directional families
(hypernym: term_a=specific, term_b=general; holonym_*: term_a=part/member/
substance, term_b=whole). For an edge to be usable as the TESTED edge for a
given vocab word, that word's own wn_relation_term row must be term_a of the
edge -- this is a deliberate simplification (a vocab word that only ever
appears as term_b of verified edges, e.g. only as a common hypernym target
for other words, isn't tested as a C-slot in v1).

  Style A (vocab-only): tested edge has BOTH term_a (=C, revealed) and term_b
  (=D, asked) as vocab words. Anchor (A:B) = a DIFFERENT verified vocab<->vocab
  edge of the same relation_family.

  Style B (one-hard-term): tested edge has term_a (=C, revealed) a vocab word
  and term_b (=D, asked) an ordinary WordNet word. Anchor (A:B) = a verified
  ordinary<->ordinary edge (the discovered anchor bank) of the same
  relation_family.

Both styles always ask for D; the vocab word is always C, so
target_word_ids[0] (spaced-repetition credit) is always that vocab word's id
regardless of style.

The ambiguity rule (see distractors.py's synonym-exclusion paragraph, which
this extends): wrong options for C's "___" slot must exclude every word
D-independently valid for the SAME relation_type from C -- the exclusion_
lemmas set below, built from wn_relation_fanout(C, relation_type) (the full,
non-vocab-restricted target set, transitive where applicable) union D's own
synonym set. When C has no fanout (a definition-pattern-only edge, where C
has no WordNet synset at all), the fallback is D's own fanout for that
relation_type plus D's synonyms -- for the transitive families this is
provably the same set C's own closure would produce, since D is C's direct
parent, so C's ancestor closure = {D} union D's ancestors.

Conversely, trap_lemmas -- real completions of the SAME relation using A or B
instead of C -- are the authentic MAT "wrong term, right relation" distractor
and are explicitly NOT excluded; see distractors.select_analogy_distractors.

trap_lemmas draws from the SAME transitive fanout as exclusion_lemmas
(_fanout_lemmas), which is correct for exclusion (every ancestor really is a
valid alternate answer, so all of them must be caught) but wrong for a trap:
WordNet's noun hierarchy has one root, entity.n.01, with direct hyponyms
abstraction.n.06 (lemma names "abstraction"/"abstract entity") and
physical_entity.n.01 -- both sit atop virtually every noun's hypernym chain
(confirmed live: 'entity' in 80% of terms' hypernym fanout, 'abstraction'/
'abstract entity' in 45%). Reused as traps, they show up as a distractor on
nearly every is_a-family analogy regardless of what's actually being tested
-- reported live as "abstraction"/"abstract entity" appearing constantly.
A trap is supposed to be a plausible, SPECIFIC wrong answer; these are
satisfied by nearly every noun in English, so they test nothing.
_TOO_GENERIC_FOR_TRAP strips them from the trap pool only, not from
exclusion_lemmas -- they're still legitimate completions, just useless ones.

Item quality rules (2026-11-20 review of live items):
  * The anchor must test the SAME relation the question does: same
    relation_type (definition-pattern kind_of/part_of count as hypernym/
    holonym_part) AND the same part of speech on each side, so its direction
    matches -- matching on relation_family alone paired INSANE:INSANITY
    (adj->noun) with VASCULARITY:vascular (noun->adj), FRANCE:EC with
    CHOKECHERRY:prunus. Anchors the review-analogy-anchors pass rejected
    (minor sense, rare word, near-synonym pair like PUB:TAVERN) are never used.
  * Traps are ranked most-specific first (a lemma's generality = how many
    terms carry it in their fanout for that relation), POS-matched to D and
    sampled -- they used to be sorted alphabetically and cut, so 'act'/
    'action'/'activity' won on nearly every is_a item.
  * A tested edge whose answer is spelled inside C (TORPIDITY -> torpid,
    DIVULGER -> divulge) is skipped outright: the answer can be read off C.
    A "which form?" variant (wrong options = other forms of C's stem) was
    tried and dropped -- picking the suffix is still spelling, not meaning.
  * Wrong options come first from C's OWN neighbourhood through a relation
    other than the tested one (2026-11-22, _contrast_lemmas): its siblings,
    narrower kinds, parts, wholes, opposite... plus one or two of D's
    siblings, so D isn't the lone general word. Measured on the old design:
    in 62% of items the answer was by far the option closest in meaning to
    C, because every wrong option came from the anchor's side -- the item
    could be solved without the anchor. Exclusion now covers the tested
    relation's whole family (a definition-pattern kind_of item still
    excludes C's WordNet hypernyms; a member-of word on a part-of item).
  * Skipped too: holonym_member edges whose answer is a Latin taxon (prunus,
    ranunculaceae -- trivia, not vocabulary), and edges where C is a plural/
    inflected form (PHOTOMICROGRAPHS -> picture) of a word D's form doesn't
    match.
"""

from __future__ import annotations

import functools
import os
import random
import re
import time
from dataclasses import dataclass, field


# entity.n.01 (WordNet's sole noun-hierarchy root) and its direct hyponyms
# -- abstraction.n.06, physical_entity.n.01, thing.n.08 -- see module
# docstring for why these are stripped from trap_lemmas specifically, not
# exclusion_lemmas. "thing" appears far less often than the other two in
# this corpus's fanout (thing.n.08's own subtree is narrow), but it's
# structurally the same near-root, information-free case, not a better
# trap for being rarer -- included on that basis, not frequency.
_TOO_GENERIC_FOR_TRAP = {"entity", "abstraction", "abstract entity", "physical entity", "thing"}

# Relation types that state the same relation from a different source, so
# either can anchor the other. Every other type only anchors itself.
_TYPE_GROUP = {"definition_pattern_kind_of": "hypernym", "definition_pattern_part_of": "holonym_part"}

_TRANSITIVE_TYPES = {"hypernym", "holonym_part", "holonym_member", "holonym_substance"}

# A trap lemma carried in more than this share of all terms' fanout for the
# relation (and by at least _GENERIC_MIN_TERMS of them) is too general to be
# a specific wrong answer -- 'act', 'artifact', 'organism'.
_GENERIC_SHARE = 0.02
_GENERIC_MIN_TERMS = 50
_TRAPS_PER_ANCHOR_WORD = 6
_GENERALITY_TTL_S = 3600

_TAXON_GLOSS = re.compile(
    r"^(?:an? )?(?:large |small )?(?:genus|family|order|class|subclass|suborder|division|phylum|tribe)\b"
    r"|\b(?:genus|family) of\b", re.I)

# Wiktionary definitions that mark an entry as an inflection/variant of
# another word rather than a form of its own.
# WordNet lists a taxon's formal name among its synonyms ('family
# iguanidae', 'order amphipoda') even when the gloss doesn't say so.
_TAXON_SYNONYM = re.compile(
    r"^(?:genus|family|order|class|subclass|suborder|superfamily|subfamily|division|phylum|subphylum|"
    r"tribe|kingdom|subkingdom|superorder|infraorder) [a-z]", re.I)

_FORM_OF_DEF = re.compile(
    r"^\W*(?:\([^)]*\)\s*)?(?:plural|(?:simple )?past|present participle|past participle|third[- ]person|"
    r"comparative|superlative|alternative (?:form|spelling)|obsolete (?:form|spelling)|archaic (?:form|spelling)|"
    r"misspelling|nonstandard (?:form|spelling)|eye dialect)\b", re.I)

_WN_POS_EQ = {"s": "a"}  # WordNet satellite adjectives are adjectives


@dataclass
class AnalogyAssembly:
    style: str                  # 'vocab_only' | 'one_hard_term'
    a_lemma: str
    b_lemma: str
    c_lemma: str
    relation_type: str
    target_word_id: int         # the vocab word being probed -- target_word_ids[0]
    d_term_id: int
    d_word_id: int | None       # set iff D is itself a vocab word (style A)
    d_lemma: str
    d_pos: str                  # canonical POS, for distractor POS-matching
    exclusion_lemmas: set[str]
    trap_lemmas: list[str]
    # A's and B's own synonyms -- a trap/option equal to one is just the
    # anchor restated (FRANCE:EC offering 'eec'/'common market').
    anchor_synonyms: set[str] = field(default_factory=set)
    # Wrong options from C's own neighbourhood through another relation,
    # as (lemma, strategy) -- see _contrast_lemmas.
    contrast_lemmas: list[tuple[str, str]] = field(default_factory=list)
    tested_edge_id: int | None = None
    anchor_edge_id: int | None = None
    relation_family: str | None = None


def _type_group(relation_type: str) -> str:
    return _TYPE_GROUP.get(relation_type, relation_type)


def _group_members(relation_type: str) -> list[str]:
    g = _type_group(relation_type)
    return sorted({g, *(t for t, grp in _TYPE_GROUP.items() if grp == g)})


def _family(relation_type: str) -> str | None:
    from .analogies import _RELATION_FAMILY

    return _RELATION_FAMILY.get(relation_type)


def _family_members(relation_type: str) -> list[str]:
    """Every relation type in the tested one's family: a word completing
    ANY of them reads as a right answer (a member-of word on a part-of
    item), so exclusion covers them all."""
    from .analogies import _RELATION_FAMILY

    fam = _RELATION_FAMILY.get(relation_type)
    return sorted({relation_type, *_group_members(relation_type),
                   *(t for t, f in _RELATION_FAMILY.items() if fam and f == fam)})


def _wn_pos_eq(p: str | None) -> str | None:
    return _WN_POS_EQ.get(p, p)


def _is_taxon(gloss: str, synonyms: list[str], synset_name: str | None = None) -> bool:
    return (bool(_TAXON_GLOSS.search(gloss or "")) or any(_TAXON_SYNONYM.search(x) for x in synonyms or [])
            or _synset_is_taxon(synset_name))


@functools.lru_cache(maxsize=50000)
def _synset_is_taxon(synset_name: str | None) -> bool:
    """WordNet's own view: the synset is a genus/family/... (a hyponym of
    taxonomic_group) -- 'alyssum' as genus Alyssum, even when the word's
    dictionary gloss describes the plant."""
    if not synset_name:
        return False
    from .distractors import _wordnet

    wn = _wordnet()
    if wn is None:
        return False
    try:
        syn = wn.synset(synset_name)
    except Exception:  # noqa: BLE001 -- unknown synset name
        return False
    return any(h.name() == "taxonomic_group.n.01" for h in syn.closure(lambda x: x.hypernyms()))


@functools.lru_cache(maxsize=50000)
def _whole_is_taxon(part_synset: str | None, whole_lemma: str) -> bool:
    """The WordNet whole of `part_synset` named `whole_lemma` is a taxon --
    madwort's member holonym 'alyssum' is genus Alyssum, though the word
    alyssum's own row holds the plant sense."""
    if not part_synset:
        return False
    from .distractors import _wordnet

    wn = _wordnet()
    if wn is None:
        return False
    try:
        syn = wn.synset(part_synset)
    except Exception:  # noqa: BLE001 -- unknown synset name
        return False
    key = whole_lemma.lower().replace(" ", "_")
    wholes = syn.member_holonyms() + syn.part_holonyms() + syn.substance_holonyms()
    return any(_synset_is_taxon(h.name()) for h in wholes if key in {n.lower() for n in h.lemma_names()})


def shares_stem(c: str, d: str) -> bool:
    """True when D is (near-)spelled inside C or vice versa -- the answer
    can be picked by spelling alone (TORPIDITY/torpid, DOGAL/doge,
    IMPERFECT/perfect, FUNCTIONING/function)."""
    c, d = c.lower(), d.lower()
    short, long_ = sorted((c, d), key=len)
    if len(short) >= 3 and short in long_:
        return True
    p = len(os.path.commonprefix([c, d]))
    return p >= max(3, -(-6 * len(short) // 10))  # ceil(0.6 * len(short))


def _term_row(cur, schema: str, word_id: int) -> dict | None:
    cur.execute(f"SELECT id, word_id, lemma, wn_pos, synonym_lemmas, synset_name FROM {schema}.wn_relation_term "
                f"WHERE word_id = %s", (word_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return {"id": row[0], "word_id": row[1], "lemma": row[2], "wn_pos": row[3], "synonym_lemmas": row[4] or [],
            "synset_name": row[5]}


def _tested_edge_candidates(cur, schema: str, target_term_id: int, style: str, exclude_word_ids: set[int]) -> list[dict]:
    """Verified edges with target as term_a, partitioned by style: style
    'vocab_only' wants term_b to be a vocab word, style 'one_hard_term' wants
    term_b to be ordinary (word_id IS NULL)."""
    cur.execute(
        f"""SELECT e.id, e.term_b_id, e.relation_type, e.relation_family, e.pos_a, e.pos_b,
                   tb.word_id, tb.lemma, tb.wn_pos, tb.synonym_lemmas, tb.gloss, tb.synset_name
            FROM {schema}.word_relation_edge e
            JOIN {schema}.wn_relation_term tb ON tb.id = e.term_b_id
            WHERE e.term_a_id = %s AND e.verification_status = 'verified'""",
        (target_term_id,),
    )
    rows = cur.fetchall()
    out = []
    for (edge_id, term_b_id, relation_type, relation_family, pos_a, pos_b,
         tb_word_id, tb_lemma, tb_wn_pos, tb_syns, tb_gloss, tb_synset) in rows:
        is_vocab_b = tb_word_id is not None
        if style == "vocab_only" and not is_vocab_b:
            continue
        if style == "one_hard_term" and is_vocab_b:
            continue
        if tb_word_id is not None and tb_word_id in exclude_word_ids:
            continue
        out.append({"edge_id": edge_id, "term_b_id": term_b_id, "relation_type": relation_type,
                    "relation_family": relation_family, "pos_a": pos_a, "pos_b": pos_b, "d_word_id": tb_word_id,
                    "d_lemma": tb_lemma, "d_wn_pos": tb_wn_pos, "d_synonyms": tb_syns or [],
                    "d_gloss": tb_gloss or "", "d_synset": tb_synset})
    random.shuffle(out)
    return out


def _find_anchor(cur, schema: str, relation_type: str, pos_a: str, pos_b: str, style: str,
                  exclude_term_ids: set[int], exclude_word_ids: set[int],
                  avoid_edge_ids: set[int] | None = None, recent_edge_ids: set[int] | None = None) -> dict | None:
    """A DIFFERENT verified edge testing the same relation (relation_type,
    or one stating it from another source -- _TYPE_GROUP) with the same POS
    on each side, matching style (vocab<->vocab for style A / ordinary<->
    ordinary for style B), sharing no term with the tested edge, and not
    rejected by review-analogy-anchors (reviewed-ok anchors first)."""
    vocab_clause = "IS NOT NULL" if style == "vocab_only" else "IS NULL"
    cur.execute(
        f"""SELECT ta.id, tb.id, ta.lemma, tb.lemma, ta.word_id, tb.word_id,
                   ta.synonym_lemmas, tb.synonym_lemmas, e.id,
                   ta.gloss, tb.gloss, ta.synset_name, tb.synset_name
            FROM {schema}.word_relation_edge e
            JOIN {schema}.wn_relation_term ta ON ta.id = e.term_a_id
            JOIN {schema}.wn_relation_term tb ON tb.id = e.term_b_id
            WHERE e.relation_type = ANY(%s) AND e.pos_a = %s AND e.pos_b = %s
              AND e.verification_status = 'verified'
              AND e.anchor_status IS DISTINCT FROM 'rejected'
              AND ta.word_id {vocab_clause} AND tb.word_id {vocab_clause}
              AND coalesce(e.anchor_clarity, 3) >= 3
            ORDER BY (e.anchor_status = 'ok') DESC NULLS LAST, coalesce(e.anchor_clarity, 3) >= 4 DESC,
                     random() LIMIT 40""",
        (_group_members(relation_type), pos_a, pos_b),
    )
    part_of = _family(relation_type) == "part_of"
    rows = [r for r in cur.fetchall() if r[8] not in (avoid_edge_ids or set())
            and not (part_of and (_is_taxon(r[9], r[6], r[11]) or _is_taxon(r[10], r[7], r[12])
                                  or _whole_is_taxon(r[11], r[3])))]
    # Anchors this user saw recently go last, not out: a small pool still works.
    rows.sort(key=lambda r: r[8] in (recent_edge_ids or set()))
    for ta_id, tb_id, ta_lemma, tb_lemma, ta_word_id, tb_word_id, ta_syns, tb_syns, edge_id, *_ in rows:
        if ta_id in exclude_term_ids or tb_id in exclude_term_ids:
            continue
        if (ta_word_id is not None and ta_word_id in exclude_word_ids) or \
           (tb_word_id is not None and tb_word_id in exclude_word_ids):
            continue
        return {"edge_id": edge_id, "a_id": ta_id, "b_id": tb_id, "a_lemma": ta_lemma, "b_lemma": tb_lemma,
                "synonyms": {x.lower() for x in (ta_syns or []) + (tb_syns or [])}}
    return None


def _fanout_lemmas(cur, schema: str, term_id: int, relation_type: str | list[str]) -> set[str]:
    types = [relation_type] if isinstance(relation_type, str) else relation_type
    cur.execute(f"SELECT target_lemma FROM {schema}.wn_relation_fanout WHERE term_id = %s AND relation_type = ANY(%s)",
                (term_id, types))
    return {r[0] for r in cur.fetchall()}


_generality_cache: dict[tuple[str, str], tuple[float, int, dict[str, int]]] = {}


def _generality(cur, schema: str, relation_type: str) -> tuple[int, dict[str, int]]:
    """(terms with any fanout for relation_type, {lemma: how many of those
    terms carry it}) -- cached per process for an hour; the fanout only
    changes when backfill-analogies runs."""
    key = (schema, relation_type)
    hit = _generality_cache.get(key)
    if hit and time.monotonic() - hit[0] < _GENERALITY_TTL_S:
        return hit[1], hit[2]
    cur.execute(f"""SELECT target_lemma, count(DISTINCT term_id) FROM {schema}.wn_relation_fanout
                    WHERE relation_type = %s GROUP BY 1""", (relation_type,))
    counts = {lemma.lower(): n for lemma, n in cur.fetchall()}
    cur.execute(f"SELECT count(DISTINCT term_id) FROM {schema}.wn_relation_fanout WHERE relation_type = %s",
                (relation_type,))
    total = cur.fetchone()[0] or 0
    _generality_cache[key] = (time.monotonic(), total, counts)
    return total, counts


def _trap_lemmas(cur, schema: str, anchor_term_ids: list[int], relation_type: str, d_wn_pos: str | None,
                 drop: set[str]) -> list[str]:
    """Real completions of the tested relation via A or B instead of C --
    POS-matched to D, most specific first (for transitive relations the
    fanout is the whole ancestor chain, so specificity = rarity of the lemma
    across all terms' fanout), the near-root/generic ones dropped, sampled
    so the same few don't repeat. A's and B's picks are interleaved."""
    total, counts = _generality(cur, schema, relation_type)
    generic_floor = max(_GENERIC_MIN_TERMS, _GENERIC_SHARE * total)
    want_pos = _wn_pos_eq(d_wn_pos)
    per_word: list[list[str]] = []
    for term_id in anchor_term_ids:
        cur.execute(f"""SELECT target_lemma, target_pos FROM {schema}.wn_relation_fanout
                        WHERE term_id = %s AND relation_type = %s""", (term_id, relation_type))
        cands = []
        for lemma, tpos in cur.fetchall():
            lc = lemma.lower()
            if lc in drop or lc in _TOO_GENERIC_FOR_TRAP:
                continue
            if want_pos and _wn_pos_eq(tpos) != want_pos:
                continue
            n = counts.get(lc, 0)
            if n >= generic_floor:
                continue
            cands.append((n, random.random(), lemma))
        cands.sort()
        top = [lemma for _, _, lemma in cands[:_TRAPS_PER_ANCHOR_WORD]]
        random.shuffle(top)
        per_word.append(top)
    out: list[str] = []
    seen: set[str] = set()
    for i in range(max((len(w) for w in per_word), default=0)):
        for w in per_word:
            if i < len(w) and w[i].lower() not in seen:
                seen.add(w[i].lower())
                out.append(w[i])
    return out


# Neighbour kinds stored by analogies.backfill_contrasts / extract_wordnet_
# edges, mapped to the relation family a neighbour of that kind completes
# FROM C. None = a relation no item tests in that direction (siblings,
# more specific kinds, parts of C), so always a wrong answer.
_CONTRAST_KINDS = {
    "sibling_of_hypernym_parent": None,
    "contrast_hyponym": None,
    "contrast_meronym": None,
    "contrast_hypernym": "is_a",
    "contrast_holonym": "part_of",
    "antonym": "opposite",
    "similar_to": "similar",
    "derivationally_related": "derived",
    "attribute": "attribute",
}
_CONTRAST_STRATEGY = {
    "sibling_of_hypernym_parent": "contrast_sibling", "contrast_hyponym": "contrast_narrower",
    "contrast_meronym": "contrast_part", "contrast_hypernym": "contrast_broader",
    "contrast_holonym": "contrast_whole", "antonym": "contrast_opposite", "similar_to": "contrast_similar",
    "derivationally_related": "contrast_derived", "attribute": "contrast_attribute",
}
_CONTRAST_PER_KIND = 4


def _zipf(lemma: str) -> float:
    from wordfreq import zipf_frequency

    return zipf_frequency(lemma, "en")


def _contrast_lemmas(cur, schema: str, c_term_id: int, d_term_id: int, relation_type: str,
                     d_wn_pos: str | None, d_lemma: str, drop: set[str]) -> list[tuple[str, str]]:
    """Wrong options that stand in some OTHER relation to C (its siblings,
    narrower kinds, parts, opposite...), POS-matched to D, plus D's own
    siblings (one option at D's level of generality, so the general word
    isn't the odd one out). Within each kind, words closest to D in
    frequency and length first, so D doesn't stand out as the only common
    or the only one-word option. Kinds interleaved: C-side, D-side, C-side..."""
    family = _family(relation_type)
    kinds = [k for k, fam in _CONTRAST_KINDS.items() if fam is None or fam != family]
    want_pos = _wn_pos_eq(d_wn_pos)
    d_zipf, d_words = _zipf(d_lemma), len(d_lemma.split())
    cur.execute(f"""SELECT relation_type, target_lemma, target_pos FROM {schema}.wn_relation_fanout
                    WHERE term_id = %s AND relation_type = ANY(%s)""", (c_term_id, kinds))
    rows = [(k, lemma, pos, "c") for k, lemma, pos in cur.fetchall()]
    cur.execute(f"""SELECT target_lemma, target_pos FROM {schema}.wn_relation_fanout
                    WHERE term_id = %s AND relation_type = 'sibling_of_hypernym_parent'""", (d_term_id,))
    rows += [("d_sibling", lemma, pos, "d") for lemma, pos in cur.fetchall()]

    by_kind: dict[str, list[tuple[float, str]]] = {}
    seen: set[str] = set()
    for kind, lemma, pos, _side in rows:
        lc = lemma.lower()
        if not lemma or lc in drop or lc in seen or lc in _TOO_GENERIC_FOR_TRAP:
            continue
        if want_pos and _wn_pos_eq(pos) != want_pos:
            continue
        if _TAXON_SYNONYM.search(lc) or not re.fullmatch(r"[a-z][a-z' -]*[a-z]", lc):
            continue
        seen.add(lc)
        score = abs(_zipf(lemma) - d_zipf) + (1.0 if len(lemma.split()) != d_words else 0.0) + random.random()
        by_kind.setdefault(kind, []).append((score, lemma))
    c_lists, d_list = [], []
    for kind, items in by_kind.items():
        items.sort()
        picked = [(lemma, "d_sibling" if kind == "d_sibling" else _CONTRAST_STRATEGY[kind])
                  for _, lemma in items[:_CONTRAST_PER_KIND]]
        (d_list if kind == "d_sibling" else c_lists).append(picked)
    random.shuffle(c_lists)
    late: list[tuple[str, str]] = []
    if family == "is_a":
        # On a kind-of item C's own siblings read as a second answer (bank
        # pilot 2026-11-22: haversack -> kitbag, sigmoidoscope -> laparoscope
        # sank 27% of the items they appeared in, D's siblings 9%), so they
        # go last.
        late = next((lst for lst in c_lists if lst and lst[0][1] == "contrast_sibling"), [])
        c_lists = [lst for lst in c_lists if lst is not late]
    c_flat = [x for i in range(_CONTRAST_PER_KIND) for lst in c_lists if i < len(lst) for x in [lst[i]]]
    out: list[tuple[str, str]] = []
    d_iter = iter(d_list[0] if d_list else [])
    for i, item in enumerate(c_flat):
        out.append(item)
        if i in (0, 3):  # D-level options as the 2nd and 5th
            nxt = next(d_iter, None)
            if nxt:
                out.append(nxt)
    out.extend(d_iter)
    return out + late


def _exclusion_lemmas(cur, schema: str, c_term_id: int, d_term_id: int, d_lemma: str,
                       d_synonyms: list[str], relation_type: str) -> set[str]:
    # The whole family, not just relation_type: a definition-pattern kind_of
    # edge has no fanout of its own, but C's WordNet hypernyms are still
    # right answers to it.
    types = _family_members(relation_type)
    excl = _fanout_lemmas(cur, schema, c_term_id, types)
    if not excl:
        # C has no fanout for this relation (no synset, e.g. a definition-pattern-only
        # edge) -- D's own closure is the provably-equivalent fallback, see module docstring.
        excl = _fanout_lemmas(cur, schema, d_term_id, types)
    excl = {e.lower() for e in excl}
    excl.add(d_lemma.lower())
    excl |= {s.lower() for s in d_synonyms}
    return excl


def _wikt_entries(cur, terms: list[str]) -> dict[str, list[tuple[str, str]]]:
    cur.execute("SELECT term, part_of_speech, definition FROM vocab.wiktionary WHERE term = ANY(%s)", (terms,))
    out: dict[str, list[tuple[str, str]]] = {}
    for term, pos, definition in cur.fetchall():
        out.setdefault(term, []).append((pos, definition or ""))
    return out


def _is_inflected_form(cur, c: str, pos: str) -> bool:
    """C is a plural noun / inflected verb of another headword (Wiktionary
    lists it only as a form-of entry while the base exists) --
    PHOTOMICROGRAPHS, RARIFYING. Its partner D is the base form, so the
    item's grammar wouldn't line up."""
    c = c.lower()
    bases: list[str] = []
    if pos == "noun" and c.endswith("s") and not c.endswith("ss"):
        bases = [c[:-1]] + ([c[:-2]] if c.endswith("es") else []) + ([c[:-3] + "y"] if c.endswith("ies") else [])
    elif pos == "verb" and (c.endswith("ing") or c.endswith("ed")):
        root = c[:-3] if c.endswith("ing") else c[:-2]
        bases = [root, root + "e"] + ([root[:-1]] if len(root) > 2 and root[-1] == root[-2] else [])
        if c.endswith("ied"):
            bases.append(c[:-3] + "y")
    if not bases:
        return False
    want = {"noun": "NOUN", "verb": "VERB"}[pos]
    entries = _wikt_entries(cur, [c, *bases])
    if not any(p == want for b in bases for p, _ in entries.get(b, [])):
        return False
    own = [defn for p, defn in entries.get(c, []) if p == want]
    return not own or all(_FORM_OF_DEF.search(defn) for defn in own)


def select_analogy_edge(conn, schema: str, target_word_id: int, exclude_ids: set[int],
                        option_count: int = 4, family_counts: dict[str, int] | None = None,
                        avoid_anchor_ids: set[int] | None = None,
                        recent_anchor_ids: set[int] | None = None,
                        skip_tested_edge_ids: set[int] | None = None,
                        extra_exclusions: set[str] | None = None) -> "AnalogyAssembly | None":
    """`family_counts`: relation families already in the session -- the
    word's edges in the least-used family are tried first, so one family
    (is_a, about 70% of edges) doesn't fill the quiz. `avoid_anchor_ids`
    (this session's anchors) are never reused; `recent_anchor_ids` (this
    user's recent ones) only as a last resort."""
    from .model import wordnet_pos

    with conn.cursor() as cur:
        target_term = _term_row(cur, schema, target_word_id)
        if target_term is None:
            return None
        c_lemma = target_term["lemma"]

        candidates = [(style, t) for style in ("vocab_only", "one_hard_term")
                      for t in _tested_edge_candidates(cur, schema, target_term["id"], style, exclude_ids)]
        random.shuffle(candidates)
        candidates.sort(key=lambda st: (family_counts or {}).get(_family(st[1]["relation_type"]), 0))
        for style, tested in candidates:
            if tested["edge_id"] in (skip_tested_edge_ids or set()):
                continue
            d_lemma = tested["d_lemma"]
            if _family(tested["relation_type"]) == "part_of" and (
                    _is_taxon(tested["d_gloss"], tested["d_synonyms"], tested["d_synset"])
                    or _synset_is_taxon(target_term["synset_name"])
                    or _whole_is_taxon(target_term["synset_name"], tested["d_lemma"])):
                continue  # madwort : alyssum (the genus) -- taxonomy, not part-of
            if tested["pos_a"] in ("noun", "verb") and tested["pos_a"] == tested["pos_b"] \
                    and _is_inflected_form(cur, c_lemma, tested["pos_a"]):
                continue

            if shares_stem(c_lemma, d_lemma):
                continue  # answer spelled inside C (DIVULGER/divulge) -- no test of meaning
            exclusion_lemmas = _exclusion_lemmas(
                cur, schema, target_term["id"], tested["term_b_id"], d_lemma,
                tested["d_synonyms"], tested["relation_type"],
            ) | {x.lower() for x in (extra_exclusions or set())}

            anchor = _find_anchor(
                cur, schema, tested["relation_type"], tested["pos_a"], tested["pos_b"], style,
                exclude_term_ids={target_term["id"], tested["term_b_id"]},
                exclude_word_ids=exclude_ids,
                avoid_edge_ids=avoid_anchor_ids, recent_edge_ids=recent_anchor_ids,
            )
            if anchor is None:
                continue
            a_lemma, b_lemma = anchor["a_lemma"], anchor["b_lemma"]
            anchor_syns = anchor["synonyms"] | {a_lemma.lower(), b_lemma.lower()}

            drop = (exclusion_lemmas | anchor_syns
                    | {c_lemma.lower(), d_lemma.lower()}
                    | {s.lower() for s in target_term["synonym_lemmas"]})
            d_wn_pos = tested["d_wn_pos"] or wordnet_pos(tested["pos_b"])
            trap_lemmas = _trap_lemmas(cur, schema, [anchor["a_id"], anchor["b_id"]],
                                       tested["relation_type"], d_wn_pos, drop)
            contrast = _contrast_lemmas(cur, schema, target_term["id"], tested["term_b_id"],
                                        tested["relation_type"], d_wn_pos, d_lemma, drop)
            if tested["relation_type"] == "antonym":
                # Words LIKE C are the natural wrong answer for "opposite of C".
                like_c = ({s.lower() for s in target_term["synonym_lemmas"]}
                          | {x.lower() for x in _fanout_lemmas(cur, schema, target_term["id"], "similar_to")})
                like_c -= exclusion_lemmas | {c_lemma.lower()}
                trap_lemmas = random.sample(sorted(like_c), len(like_c)) + trap_lemmas

            return AnalogyAssembly(
                style=style, a_lemma=a_lemma, b_lemma=b_lemma, c_lemma=c_lemma,
                relation_type=tested["relation_type"], target_word_id=target_word_id,
                d_term_id=tested["term_b_id"], d_word_id=tested["d_word_id"], d_lemma=d_lemma,
                d_pos=tested["pos_b"], exclusion_lemmas=exclusion_lemmas, trap_lemmas=trap_lemmas,
                anchor_synonyms=anchor_syns, contrast_lemmas=contrast,
                tested_edge_id=tested["edge_id"], anchor_edge_id=anchor["edge_id"],
                relation_family=_family(tested["relation_type"]),
            )
    return None


_EXPLAIN = {
    "hypernym": "the first is a kind of the second",
    "definition_pattern_kind_of": "the first is a kind of the second",
    "holonym_part": "the first is part of the second",
    "definition_pattern_part_of": "the first is part of the second",
    "holonym_member": "the first is a member of the second",
    "holonym_substance": "the first is made of the second",
    "antonym": "the two words are opposites",
    "similar_to": "the two words mean nearly the same",
    "derivationally_related": "the second is a related form of the first, built on the same root",
    "attribute": "the first is a value or degree of the second",
    "definition_pattern_characterized_by": "the first is marked by the second",
    "definition_pattern_agent": "the first is one who does or deals in the second",
    "definition_pattern_purpose": "the first is used for the second",
    "definition_pattern_relates_to": "the first relates to the second",
    "definition_pattern_resembling": "the first resembles the second",
}


def explain(relation_type: str, a: str, b: str, c: str, d: str) -> str:
    """One plain sentence naming the relation both pairs share, shown after
    the answer: 'In both pairs the first is a kind of the second:
    cure → medication, celandine → herb.'"""
    rel = _EXPLAIN.get(relation_type, "the two words are related the same way")
    return f"In both pairs {rel}: {a.lower()} → {b.lower()}, {c.lower()} → {d.lower()}."
