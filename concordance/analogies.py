"""Analogy relation extraction + verification backfill (§ analogies).

Populates word_relation_edge (and its supporting wn_relation_term /
wn_relation_fanout / wn_relation_scan tables, see concordance/db's
_SCHEMA_DDL) so webapp/backend/quiz.py's `analogy` question type can assemble
MAT-style "A is to B as C is to ___" items entirely from precomputed data at
quiz-start time -- unlike distractors.py's live generation for mc/true_false/
matching, relation extraction (WordNet traversal + spaCy definition parsing +
LLM verification) is too expensive to redo per quiz-start, so it runs here as
an offline, resumable CLI backfill (`concordance backfill-analogies`, also
chained into `maintain` via a --skip-analogies flag).

Two relation sources feed the same word_relation_edge table:
  * WordNet (extract_wordnet_edges) -- hypernym, holonym_part/member/substance,
    antonym, similar_to, derivationally_related, attribute.
  * word.definition text, parsed with a spaCy Matcher (extract_definition_
    pattern_edges) -- recovers relations for the rarest vocab words, which
    frequently have no useful WordNet synset at all but DO state their
    relation outright ("a wooden collar worn by prisoners" -> kind_of collar).

Every candidate from either source, before it is usable in a quiz, must pass
verify_candidates: a local-LLM pass (same Qwen model judge.py already loads,
never a paid API) that checks the relation against BOTH terms' real
definitions. This is mandatory, not a quality nice-to-have -- live probing
against this vocab found that ~60-80% of raw WordNet "hypernym" edges for
rare/abstract words are actually disguised near-synonyms (e.g. plotter /
contriver), which would ship a quiz question with two right answers.
verify_candidates therefore INVERTS judge.py's keep-bias: an omitted or
unparseable verdict here defaults to REJECTED, not kept -- judge.py's
keep-bias exists because a wrongly-dropped vocabulary word is merely a missed
study opportunity, but a wrongly-kept relation pair is an actively broken
quiz question. Silence must mean discard.

Storage convention (see also the module docstring in analogy_select.py, which
consumes this table): directional relation families (hypernym, holonym_*)
store only the many-to-one direction (specific -> general, part -> whole).
The reverse (hyponym, meronym) is deliberately never generated as its own
edge type -- a hypernym has arbitrarily many hyponyms, so that direction is
exactly the "many valid answers" case the ambiguity rule in analogy_select.py
depends on being able to enumerate in full; storing it as a first-class edge
type would need a second, much larger fanout table for no benefit (the
existing wn_relation_fanout already carries the full hyponym-equivalent set
for exclusion purposes -- see extract_wordnet_edges).

relation_family (is_a / part_of / opposite / similar / derived / agentive /
purpose / attribute / relates_to / resembling) is the bucket analogy_select.py
pairs an edge against a DIFFERENT edge of the same family to build the anchor
(A:B) leg of an item. relates_to/resembling (plus attribute's definition-
pattern-sourced "characterized_by" contributions) are definition-text-mined,
not WordNet-native -- added specifically because WordNet's own hypernym/
holonym/purpose/agentive relations are almost entirely absent for adjective
synsets, so a definition-pattern source is the only way to give that POS any
relation variety at all (see _build_matchers).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from wordfreq import zipf_frequency

from .model import RelationCandidate, normalize_pos, wordnet_pos
from .tokenize import load_nlp

_ANCHOR_ZIPF_MIN = 4.0   # same "plainly frequent" bar validity_score.py uses for anchor-bank eligibility

_RELATION_FAMILY = {
    "hypernym": "is_a",
    "definition_pattern_kind_of": "is_a",
    "holonym_part": "part_of",
    "definition_pattern_part_of": "part_of",
    "holonym_member": "part_of",
    "holonym_substance": "part_of",
    "antonym": "opposite",
    "similar_to": "similar",
    "derivationally_related": "derived",
    "definition_pattern_agent": "agentive",
    "definition_pattern_purpose": "purpose",
    "attribute": "attribute",
    "definition_pattern_characterized_by": "attribute",  # folded in, not a new family -- see _build_matchers
    "definition_pattern_relates_to": "relates_to",
    "definition_pattern_resembling": "resembling",
}

# Relation families whose full target set is traversed via synset.closure()
# (transitively) when building wn_relation_fanout -- see extract_wordnet_edges.
# Every other family's fanout is direct-targets-only; these relations don't chain.
_TRANSITIVE_FAMILIES = {"hypernym", "holonym_part", "holonym_member", "holonym_substance"}

_MAX_VERIFY_PASSES = 3

_nlp_singleton = None


def _get_nlp():
    """Lazy singleton over tokenize.load_nlp() -- same en_core_web_sm model
    tokenize.py already uses for book-text tagging, reused here (not
    reloaded) for definition-text parsing."""
    global _nlp_singleton
    if _nlp_singleton is None:
        _nlp_singleton = load_nlp()
    return _nlp_singleton


def _load_wordnet():
    """Same lazy-load-with-download-fallback shape as
    validity.ValidityGate._load_wordnet -- reused verbatim rather than
    reimplemented differently here. Degrades to None if WordNet data can't
    be obtained at all (the backfill then does definition-pattern extraction
    only, WordNet-sourced relations silently yield nothing)."""
    try:
        from nltk.corpus import wordnet as wn

        wn.synsets("test")
        return wn
    except Exception:
        try:
            import nltk

            nltk.download("wordnet", quiet=True)
            from nltk.corpus import wordnet as wn

            wn.synsets("test")
            return wn
        except Exception:
            return None


def _lemma_words(synset) -> set[str]:
    return {l.name().replace("_", " ").lower() for l in synset.lemmas()}


def _canonical_synset(wn, lemma: str, wn_pos: str, own_definition: str):
    """First synset, except when a cheap bag-of-words overlap between a
    candidate synset's gloss and the word's own definition clearly favors a
    different sense. Returns None if WordNet has no synset for (lemma, wn_pos)."""
    synsets = wn.synsets(lemma, pos=wn_pos)
    if not synsets:
        return None
    if len(synsets) == 1 or not own_definition:
        return synsets[0]
    own_words = set(own_definition.lower().split())
    scored = [(len(own_words & set(s.definition().lower().split())), -i)
              for i, s in enumerate(synsets)]
    best_idx = max(range(len(synsets)), key=lambda i: scored[i])
    return synsets[best_idx] if scored[best_idx][0] > 0 else synsets[0]


# --- term resolution (vocab + ordinary/anchor terms) -------------------------

def _select_term_row(cur, schema: str, *, word_id: int | None, lemma_lc: str, wn_pos: str) -> tuple | None:
    if word_id is not None:
        cur.execute(f"SELECT id FROM {schema}.wn_relation_term WHERE word_id = %s", (word_id,))
    else:
        cur.execute(
            f"SELECT id FROM {schema}.wn_relation_term WHERE word_id IS NULL AND lemma_lc = %s AND wn_pos = %s",
            (lemma_lc, wn_pos),
        )
    return cur.fetchone()


def _find_term_id(cur, schema: str, lemma_lc: str, wn_pos: str) -> int | None:
    """Resolve a lemma+POS to its wn_relation_term id, preferring a vocab-word
    row over an ordinary-term row for the same lemma -- a lemma is never
    allowed two rows (see _get_or_create_ordinary_term), so this is the one
    lookup path every caller (extraction and edge insertion) should use once
    it only has a lemma string, not a known word_id."""
    cur.execute(
        f"""SELECT id FROM {schema}.wn_relation_term
            WHERE lemma_lc = %s AND wn_pos = %s ORDER BY word_id IS NULL LIMIT 1""",
        (lemma_lc, wn_pos),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _upsert_term(cur, schema: str, *, word_id: int | None, lemma: str, wn_pos: str,
                  synset_name: str | None, gloss: str | None, synonym_lemmas: list[str]) -> int:
    """Insert or refresh one wn_relation_term row, returns its id. Plain
    select-then-write (not ON CONFLICT) -- this is a single-process backfill,
    not concurrent writers, so the small race window doesn't matter, and it
    keeps the partial-unique-index conflict targets simple."""
    lemma_lc = lemma.lower()
    zipf = zipf_frequency(lemma_lc, "en")
    is_common = zipf >= _ANCHOR_ZIPF_MIN
    existing = _select_term_row(cur, schema, word_id=word_id, lemma_lc=lemma_lc, wn_pos=wn_pos)
    if existing:
        term_id = existing[0]
        cur.execute(
            f"""UPDATE {schema}.wn_relation_term
                SET synset_name = %s, gloss = %s, synonym_lemmas = %s, zipf = %s,
                    is_common = %s, updated_at = now()
                WHERE id = %s""",
            (synset_name, gloss, synonym_lemmas, zipf, is_common, term_id),
        )
        return term_id
    cur.execute(
        f"""INSERT INTO {schema}.wn_relation_term
                (word_id, lemma, wn_pos, synset_name, gloss, synonym_lemmas, zipf, is_common)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id""",
        (word_id, lemma, wn_pos, synset_name, gloss, synonym_lemmas, zipf, is_common),
    )
    return cur.fetchone()[0]


def resolve_vocab_terms(conn, schema: str, wn, limit: int = 0) -> list[dict]:
    """Upsert a wn_relation_term row for every active/quizzable word whose
    normalized POS has a WordNet mapping and that has real definition text
    (mandatory -- it's the only thing the verification gate has to judge
    against). Words with no WordNet synset still get a row (synset_name
    NULL) -- they're still eligible for definition-pattern extraction."""
    with conn.cursor() as cur:
        query = f"""SELECT w.id, w.lemma, w.part_of_speech, w.definition
                    FROM {schema}.word w
                    JOIN {schema}.word_difficulty wd ON wd.word_id = w.id
                    WHERE w.active AND wd.quizzable = true
                      AND w.part_of_speech IS NOT NULL AND w.definition IS NOT NULL"""
        if limit:
            query += " ORDER BY w.id LIMIT %s"
            cur.execute(query, (limit,))
        else:
            cur.execute(query)
        rows = cur.fetchall()

    out = []
    with conn.cursor() as cur:
        for word_id, lemma, pos, definition in rows:
            wn_pos = wordnet_pos(pos)
            if not wn_pos:
                continue
            synset_name = None
            synonym_lemmas: list[str] = []
            if wn is not None:
                syn = _canonical_synset(wn, lemma.lower(), wn_pos, definition or "")
                if syn is not None:
                    synset_name = syn.name()
                    synonym_lemmas = sorted(_lemma_words(syn) - {lemma.lower()})
            term_id = _upsert_term(cur, schema, word_id=word_id, lemma=lemma, wn_pos=wn_pos,
                                    synset_name=synset_name, gloss=definition, synonym_lemmas=synonym_lemmas)
            out.append({"id": term_id, "word_id": word_id, "lemma": lemma, "wn_pos": wn_pos,
                        "pos": normalize_pos(pos), "synset_name": synset_name, "gloss": definition})
        conn.commit()
    return out


def _get_or_create_ordinary_term(cur, schema: str, wn, lemma: str, wn_pos: str) -> dict | None:
    """A term_b encountered during WordNet traversal or definition-pattern
    parsing -- resolved to whichever row already exists for this lemma+POS
    (preferring a vocab-word row, via _find_term_id, over an ordinary one --
    a lemma that happens to already be a vocab word must never get a second,
    duplicate word_id=NULL row), else newly upserted with word_id=NULL and
    gloss=its own WordNet gloss. This is how the anchor bank (ordinary common
    words used as the A:B leg of a one-hard-term item) gets discovered
    incrementally across backfill runs."""
    lemma_lc = lemma.lower()
    existing_id = _find_term_id(cur, schema, lemma_lc, wn_pos)
    if existing_id is not None:
        cur.execute(f"SELECT id, word_id, lemma, wn_pos, synset_name, gloss "
                    f"FROM {schema}.wn_relation_term WHERE id = %s", (existing_id,))
        r = cur.fetchone()
        return {"id": r[0], "word_id": r[1], "lemma": r[2], "wn_pos": r[3], "synset_name": r[4], "gloss": r[5]}
    synsets = wn.synsets(lemma_lc, pos=wn_pos) if wn is not None else []
    if not synsets:
        return None
    syn = synsets[0]
    synonym_lemmas = sorted(_lemma_words(syn) - {lemma_lc})
    term_id = _upsert_term(cur, schema, word_id=None, lemma=lemma_lc, wn_pos=wn_pos,
                            synset_name=syn.name(), gloss=syn.definition(), synonym_lemmas=synonym_lemmas)
    return {"id": term_id, "word_id": None, "lemma": lemma_lc, "wn_pos": wn_pos,
            "synset_name": syn.name(), "gloss": syn.definition()}


# --- WordNet edge + fanout extraction -----------------------------------------

_DIRECT_RELATIONS = {
    "hypernym": lambda s: s.hypernyms(),
    "holonym_part": lambda s: s.part_holonyms(),
    "holonym_member": lambda s: s.member_holonyms(),
    "holonym_substance": lambda s: s.substance_holonyms(),
    "similar_to": lambda s: s.similar_tos(),
    "attribute": lambda s: s.attributes(),
}

_WN_POS_TO_CANONICAL = {"n": "noun", "v": "verb", "a": "adjective", "r": "adverb", "s": "adjective"}


def extract_wordnet_edges(conn, schema: str, wn, term_row: dict) -> tuple[list[RelationCandidate], int]:
    """WordNet-sourced RelationCandidates with term_row as term_a, plus the
    full (non-vocab-restricted) fanout for each relation type -- written
    directly to wn_relation_fanout as a side effect (not returned, since it
    can be large). Returns (candidates, fanout_row_count)."""
    if wn is None or not term_row.get("synset_name"):
        return [], 0
    synset = wn.synset(term_row["synset_name"])
    lemma_lc = term_row["lemma"].lower()
    candidates: list[RelationCandidate] = []
    fanout_count = 0

    with conn.cursor() as cur:
        for rel_type, rel_fn in _DIRECT_RELATIONS.items():
            family = _RELATION_FAMILY[rel_type]
            all_synsets = wn.synsets(lemma_lc, pos=term_row["wn_pos"])

            # Fanout: transitive closure (unioned over ALL this lemma's synsets)
            # for the families where a distractor N hops away is still a valid
            # completion of the relation; direct targets only (still over
            # every synset, not just the canonical one -- see the candidates
            # comment below) otherwise.
            fanout_targets: set[tuple[str, str]] = set()
            for s in all_synsets:
                targets = s.closure(rel_fn) if rel_type in _TRANSITIVE_FAMILIES else rel_fn(s)
                for other in targets:
                    for lw in _lemma_words(other):
                        if lw != lemma_lc:
                            fanout_targets.add((lw, other.pos()))
            for target_lemma, target_pos in fanout_targets:
                cur.execute(
                    f"""INSERT INTO {schema}.wn_relation_fanout (term_id, relation_type, target_lemma, target_pos)
                        VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                    (term_row["id"], rel_type, target_lemma, target_pos),
                )
                fanout_count += 1

            # Candidates: direct hop only -- never the transitive closure,
            # even for the transitive families, a candidate must be a real
            # testable pair, not a distant ancestor several hops removed --
            # but now over every one of the lemma's synsets, not just the
            # single sense-disambiguated canonical one. A word can carry a
            # genuine relation (an attribute, a similar_to, a hypernym) on a
            # DIFFERENT sense's synset that its own canonical sense doesn't
            # have; verify_candidates' LLM pass is the safety net against any
            # wrong-sense noise this adds, so widening only adds candidates
            # to try, it never weakens an existing guarantee. Folded into the
            # SAME set fanout_targets already computed above (not a second,
            # separately-scoped walk) so wn_relation_fanout -- which
            # analogy_select.py's exclusion-lemma set is built from -- can
            # never end up narrower than what candidates admit; that gap
            # would let a valid alternate answer slip through a quiz as an
            # unexcluded wrong option.
            candidate_targets: set[tuple[str, str]] = fanout_targets if rel_type not in _TRANSITIVE_FAMILIES \
                else {(lw, other.pos()) for s in all_synsets for other in rel_fn(s)
                      for lw in _lemma_words(other) if lw != lemma_lc}
            for lw, other_pos in candidate_targets:
                other_term = _get_or_create_ordinary_term(cur, schema, wn, lw, other_pos)
                if other_term is None:
                    continue
                candidates.append(RelationCandidate(
                    term_a_lemma=term_row["lemma"], term_a_pos=term_row["pos"],
                    term_a_gloss=term_row["gloss"],
                    term_b_lemma=other_term["lemma"], term_b_pos=_WN_POS_TO_CANONICAL.get(other_pos, other_pos),
                    term_b_gloss=other_term["gloss"] or "",
                    relation_type=rel_type, relation_family=family, source=f"wordnet_{rel_type}",
                ))

        # Antonym (lemma-level, symmetric) and derivationally_related (lemma-level).
        for lem in synset.lemmas():
            if lem.name().replace("_", " ").lower() != lemma_lc:
                continue
            for ant in lem.antonyms():
                aw = ant.name().replace("_", " ").lower()
                if aw == lemma_lc:
                    continue
                other_term = _get_or_create_ordinary_term(cur, schema, wn, aw, ant.synset().pos())
                if other_term is None:
                    continue
                cur.execute(
                    f"""INSERT INTO {schema}.wn_relation_fanout (term_id, relation_type, target_lemma, target_pos)
                        VALUES (%s, 'antonym', %s, %s) ON CONFLICT DO NOTHING""",
                    (term_row["id"], aw, ant.synset().pos()),
                )
                candidates.append(RelationCandidate(
                    term_a_lemma=term_row["lemma"], term_a_pos=term_row["pos"], term_a_gloss=term_row["gloss"],
                    term_b_lemma=other_term["lemma"],
                    term_b_pos=_WN_POS_TO_CANONICAL.get(ant.synset().pos(), ant.synset().pos()),
                    term_b_gloss=other_term["gloss"] or "",
                    relation_type="antonym", relation_family=_RELATION_FAMILY["antonym"], source="wordnet_antonym",
                ))
            for drf in lem.derivationally_related_forms():
                dw = drf.name().replace("_", " ").lower()
                if dw == lemma_lc:
                    continue
                other_term = _get_or_create_ordinary_term(cur, schema, wn, dw, drf.synset().pos())
                if other_term is None:
                    continue
                cur.execute(
                    f"""INSERT INTO {schema}.wn_relation_fanout (term_id, relation_type, target_lemma, target_pos)
                        VALUES (%s, 'derivationally_related', %s, %s) ON CONFLICT DO NOTHING""",
                    (term_row["id"], dw, drf.synset().pos()),
                )
                candidates.append(RelationCandidate(
                    term_a_lemma=term_row["lemma"], term_a_pos=term_row["pos"], term_a_gloss=term_row["gloss"],
                    term_b_lemma=other_term["lemma"],
                    term_b_pos=_WN_POS_TO_CANONICAL.get(drf.synset().pos(), drf.synset().pos()),
                    term_b_gloss=other_term["gloss"] or "",
                    relation_type="derivationally_related", relation_family=_RELATION_FAMILY["derivationally_related"],
                    source="wordnet_derivationally_related",
                ))
    return candidates, fanout_count


# --- definition-text pattern extraction ---------------------------------------

def _build_matchers(nlp):
    from spacy.matcher import Matcher

    matcher = Matcher(nlp.vocab)
    matcher.add("kind_of", [[
        {"LOWER": {"IN": ["a", "an"]}}, {"LOWER": {"IN": ["kind", "type", "sort"]}},
        {"LOWER": "of"}, {"POS": "ADJ", "OP": "*"}, {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"},
    ]])
    matcher.add("agent", [[
        {"LOWER": {"IN": ["one", "person", "someone"]}}, {"LOWER": "who"},
        {"POS": "ADV", "OP": "*"}, {"POS": "VERB", "OP": "+"},
    ]])
    matcher.add("part_of", [[
        {"LOWER": "part"}, {"LOWER": "of"}, {"LOWER": {"IN": ["a", "an", "the"]}, "OP": "?"},
        {"POS": "ADJ", "OP": "*"}, {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"},
    ]])
    matcher.add("purpose", [[
        {"LOWER": "used"}, {"LOWER": {"IN": ["for", "to"]}}, {"POS": "VERB", "OP": "+"},
    ]])
    # The three patterns below exist specifically to give adjectives (and any
    # other POS WordNet's native ontology under-serves) analogy relations
    # that don't depend on WordNet's hypernym/holonym/purpose/agentive graph
    # -- that graph structurally has almost nothing for adjective synsets
    # (confirmed empirically: 8 raw is_a candidates across 9,152 adjective
    # vocab terms), so no amount of WordNet-side extraction tuning closes
    # that gap. These read straight off the vocab word's own definition text
    # instead, the same technique kind_of/agent/part_of/purpose already use.
    matcher.add("relates_to", [[
        {"LOWER": "of", "OP": "?"}, {"LOWER": "or", "OP": "?"},
        {"LEMMA": {"IN": ["relate", "pertain"]}}, {"LOWER": "to"},
        {"LOWER": {"IN": ["a", "an", "the"]}, "OP": "?"}, {"POS": "ADJ", "OP": "*"},
        {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"},
    ]])
    matcher.add("resembling", [
        [{"LEMMA": "resemble"}, {"LOWER": {"IN": ["a", "an", "the"]}, "OP": "?"},
         {"POS": "ADJ", "OP": "*"}, {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
        [{"LOWER": "like"}, {"LOWER": {"IN": ["a", "an"]}},
         {"POS": "ADJ", "OP": "*"}, {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
    ])
    # Deliberately folded into the existing "attribute" family (not a new
    # one) at the _PATTERN_LABEL_TO_RELATION/_RELATION_FAMILY layer below --
    # "having X"/"characterized by X" is the same "which dimension/quality
    # does this word vary along" relationship WordNet's own .attributes()
    # link already represents (heavy -> weight), just mined from definition
    # text where WordNet itself is thin (only 22 verified attribute edges
    # existed for the whole adjective vocab before this pattern).
    matcher.add("characterized_by", [
        [{"LEMMA": "have"}, {"LOWER": {"IN": ["a", "an", "the"]}, "OP": "?"},
         {"POS": "ADJ", "OP": "*"}, {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
        [{"LEMMA": "characterize"}, {"LOWER": "by"}, {"LOWER": {"IN": ["a", "an", "the"]}, "OP": "?"},
         {"POS": "ADJ", "OP": "*"}, {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
    ])
    return matcher


_PATTERN_LABEL_TO_RELATION = {
    "kind_of": "definition_pattern_kind_of",
    "agent": "definition_pattern_agent",
    "part_of": "definition_pattern_part_of",
    "purpose": "definition_pattern_purpose",
    "relates_to": "definition_pattern_relates_to",
    "resembling": "definition_pattern_resembling",
    "characterized_by": "definition_pattern_characterized_by",
}

_matcher_singleton = None


def _get_matcher(nlp):
    global _matcher_singleton
    if _matcher_singleton is None:
        _matcher_singleton = _build_matchers(nlp)
    return _matcher_singleton


def extract_definition_pattern_edges(conn, schema: str, wn, nlp, term_row: dict) -> list[RelationCandidate]:
    """spaCy Matcher over term_row's own gloss text -- a vocab word's own
    definition, or an ordinary/anchor term's WordNet synset definition (see
    _get_or_create_ordinary_term); this function doesn't care which. Recovers
    relations WordNet's relation graph doesn't have for exactly the words/
    POS classes WordNet is thinnest on (see relates_to/resembling/
    characterized_by's docstring in _build_matchers). Only emits a candidate
    when the parsed head term resolves to a WordNet synset (needed for the
    exclusion-set fallback in analogy_select.py when term_a itself has no
    synset)."""
    if wn is None or not term_row.get("gloss"):
        return []
    doc = nlp(term_row["gloss"])
    matcher = _get_matcher(nlp)
    matches = matcher(doc)
    candidates: list[RelationCandidate] = []
    seen: set[tuple[str, str]] = set()
    with conn.cursor() as cur:
        for match_id, start, end in matches:
            label = nlp.vocab.strings[match_id]
            relation_type = _PATTERN_LABEL_TO_RELATION[label]
            span = doc[start:end]
            head_tokens = [t for t in span if t.pos_ in ("NOUN", "PROPN", "VERB")]
            if not head_tokens:
                continue
            head = head_tokens[-1].lemma_.lower()
            if head == term_row["lemma"].lower() or (relation_type, head) in seen:
                continue
            seen.add((relation_type, head))
            head_wn_pos = "v" if label == "agent" or label == "purpose" else "n"
            other_term = _get_or_create_ordinary_term(cur, schema, wn, head, head_wn_pos)
            if other_term is None:
                continue
            candidates.append(RelationCandidate(
                term_a_lemma=term_row["lemma"], term_a_pos=term_row["pos"], term_a_gloss=term_row["gloss"],
                term_b_lemma=other_term["lemma"], term_b_pos=_WN_POS_TO_CANONICAL.get(head_wn_pos, "noun"),
                term_b_gloss=other_term["gloss"] or "",
                relation_type=relation_type, relation_family=_RELATION_FAMILY[relation_type],
                source=relation_type,
            ))
    return candidates


# --- LLM verification (reject-biased -- see module docstring) -----------------

def _load_llm(cfg):
    from llama_cpp import Llama

    return Llama(model_path=cfg.model_path, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)


# Precise, per-relation_type descriptions handed to the verifier ALONGSIDE
# the bare type string -- an opaque identifier like "definition_pattern_
# relates_to" gives a judge model nothing to hold the near-synonym rejection
# clause against, and relates_to/resembling/characterized_by are exactly the
# families likely to get reflexively rejected as "just associated" or "just a
# paraphrase" without one (arboreal/tree is a real relates_to pair, not a
# near-synonym one, but the model has no way to know that from the bare name
# alone). similar_to is deliberately exempted from the near-synonym rule
# below since it IS WordNet's own similar-meaning link, not a paraphrase bug.
_RELATION_TYPE_DESCRIPTIONS = {
    "hypernym": "b is a broader, more general category that a is a specific kind of (a IS-A b)",
    "holonym_part": "a is literally a physical part of the whole b",
    "holonym_member": "a is a member of the group/organization b",
    "holonym_substance": "a is made of the substance/material b",
    "antonym": "a and b are opposites in meaning",
    "similar_to": "a and b are near-synonymous / very similar in meaning -- this IS the intended relation for this type, not a paraphrase error",
    "derivationally_related": "a and b are different grammatical forms sharing the same root (e.g. noun/verb/adjective forms of one concept), not necessarily synonyms",
    "attribute": "b is the underlying dimension/quality that a describes a value or degree of (e.g. heavy is a value of weight)",
    "definition_pattern_kind_of": "a's own definition literally states it is a kind, type, or sort of b",
    "definition_pattern_agent": "a's own definition describes a person or agent who does b",
    "definition_pattern_part_of": "a's own definition literally states it is part of b",
    "definition_pattern_purpose": "a's own definition states it is used for or to do b",
    "definition_pattern_relates_to": "a's own definition literally states it relates or pertains to the field, subject, or domain b -- reject if a and b are merely topically associated without the definition actually saying this",
    "definition_pattern_resembling": "a's own definition literally states it resembles or is like b in appearance or quality -- a real resemblance claim, not mere topical association",
    "definition_pattern_characterized_by": "b is the quality, feature, or characteristic that a's own definition states it has (e.g. 'having b' / 'characterized by b') -- b is the DIMENSION a varies along, not a synonym of a",
}

# For judging an EXAMPLE pair the reader sees two words, not definitions:
# the anchor review and clarity score ask whether the relation is plain,
# so the definition-pattern relations are described without "a's own
# definition literally states" (that was the edge verifier's question,
# already answered -- and a hand-picked knife : cut fails it, since
# WordNet's gloss for knife doesn't mention cutting).
_ANCHOR_DESCRIPTIONS = {
    **_RELATION_TYPE_DESCRIPTIONS,
    "definition_pattern_kind_of": "a is a kind, type, or sort of b",
    "definition_pattern_agent": "a is a person or agent who does b",
    "definition_pattern_part_of": "a is part of b",
    "definition_pattern_purpose": "a is used for or to do b",
    "definition_pattern_relates_to": "a means relating or pertaining to b",
    "definition_pattern_resembling": "a means resembling or like b",
    "definition_pattern_characterized_by": "a means having, full of, or marked by b",
}


_VERIFY_SYSTEM = (
    "You are checking candidate word-analogy relation pairs for a vocabulary quiz. "
    "For each pair, word 'b' is claimed to stand in the relationship described by "
    "'relation_description' to word 'a', judged strictly from the two definitions "
    "given -- not from general knowledge of the words. Judge ok=true ONLY if the "
    "described relation genuinely and specifically holds. Reject (ok=false) if the "
    "relation doesn't actually hold, OR if 'a' and 'b' are just near-synonyms / "
    "restatements of each other UNLESS the relation_description itself says that's the "
    "intended relation (a valid analogy relation must be a specific, nameable "
    "relationship matching its own description, not an unrelated paraphrase). When "
    "genuinely unsure, reject. "
    'Output ONLY a JSON array, one object per input, in the same order: '
    '[{"i": <int>, "ok": <true|false>}]. Include every input exactly once.'
)


def _verify_query(llm, batch: list[RelationCandidate]) -> str:
    items = [
        {"i": i, "relation_description": _RELATION_TYPE_DESCRIPTIONS.get(c.relation_type, c.relation_type),
         "a": c.term_a_lemma, "a_def": c.term_a_gloss[:200],
         "b": c.term_b_lemma, "b_def": c.term_b_gloss[:200]}
        for i, c in enumerate(batch)
    ]
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _VERIFY_SYSTEM},
            {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
        ],
        temperature=0.0,
        max_tokens=len(batch) * 12 + 64,
    )
    return out["choices"][0]["message"]["content"]


def _parse_verify_response(text: str) -> list | None:
    """Tolerantly pull the verdict list out of a model reply -- same shape as
    judge._parse_verdicts (code fences, object wrapper, or a bare array)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
    start = min((i for i in (text.find("["), text.find("{")) if i != -1), default=-1)
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
    if isinstance(data, dict):
        return data.get("results") or None
    return data if isinstance(data, list) else None


def apply_verify_verdicts(batch: list[RelationCandidate], verdicts: list | None) -> None:
    """Apply parsed verdicts to `batch` in place, setting .verdict/.verification_note.
    REJECT-biased (the opposite of judge.apply_verdicts's keep-bias): any index
    missing from `verdicts`, or a totally unparseable response, defaults every
    candidate in `batch` to 'rejected' -- an unverified pair must never ship,
    since it would put a possible two-right-answer question in front of a
    quiz-taker. Pure function, side-effecting only on `batch`, so it's testable
    without a real (nondeterministic) LLM call."""
    if verdicts is None:
        for c in batch:
            c.verdict = "rejected"
            c.verification_note = "verification response unparseable"
        return
    by_index = {}
    for v in verdicts:
        if isinstance(v, dict) and isinstance(v.get("i"), int):
            by_index[v["i"]] = v
    for i, c in enumerate(batch):
        v = by_index.get(i)
        if v is None:
            c.verdict = "rejected"
            c.verification_note = "omitted from verification response"
        elif bool(v.get("ok", False)):
            c.verdict = "verified"
        else:
            c.verdict = "rejected"


def verify_candidates(cfg, candidates: list[RelationCandidate], batch_size: int = 20, llm=None) -> None:
    """judge.py-style batched LLM pass, with the re-query-for-completeness
    loop dropped (unlike judge.py's keep-bias, where an omission needs a
    retry to avoid defaulting a real word to keep, an omission here safely
    defaults to reject on the first pass -- no retry needed for correctness,
    only for yield, so a single pass per batch is deliberately simpler than
    judge.LlamaJudge._judge_batch). `llm` is injectable for testing."""
    if not candidates:
        return
    llm = llm or _load_llm(cfg)
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        for attempt in range(_MAX_VERIFY_PASSES):
            try:
                verdicts = _parse_verify_response(_verify_query(llm, batch))
                break
            except Exception:  # noqa: BLE001 -- a flaky model call shouldn't crash the backfill
                verdicts = None
        apply_verify_verdicts(batch, verdicts)
        done = min(i + batch_size, len(candidates))
        if done % (batch_size * 5) == 0 or done == len(candidates):
            verified_so_far = sum(1 for c in candidates[:done] if c.verdict == "verified")
            print(f"[backfill-analogies]   ...{done}/{len(candidates)} edges verified "
                  f"({verified_so_far} passed so far)")


# --- sibling fanout (one-hard-term distractor plausibility) -------------------

def _common_lemma_words(synset) -> set[str]:
    """_lemma_words minus names: WordNet capitalises people, places and
    Latin taxa (Daniel, Trondheim_Fjord, genus_Ondatra, Althea_officinalis)
    -- useless as wrong options -- while marsh_mallow in the same synset stays."""
    return {lem.name().replace("_", " ").lower() for lem in synset.lemmas()
            if not any(part[:1].isupper() for part in lem.name().split("_"))}


def compute_sibling_fanout(conn, schema: str, wn, d_term: dict) -> int:
    """For a term_b that just became a usable D under a verified is_a-family
    edge: its own synsets' hypernyms' hyponyms (co-hyponyms / "siblings"),
    minus its own synset -- stored as relation_type='sibling_of_hypernym_parent'.
    Only called for terms that ended up as a usable D (not every discovered
    term), to bound the work. Returns the number of fanout rows written."""
    if wn is None or not d_term.get("synset_name"):
        return 0
    synset = wn.synset(d_term["synset_name"])
    lemma_lc = d_term["lemma"].lower()
    siblings: set[tuple[str, str]] = set()
    for parent in synset.hypernyms():
        for sibling in parent.hyponyms():
            if sibling.name() == synset.name():
                continue
            for lw in _common_lemma_words(sibling):
                if lw != lemma_lc:
                    siblings.add((lw, sibling.pos()))
    count = 0
    with conn.cursor() as cur:
        for target_lemma, target_pos in siblings:
            cur.execute(
                f"""INSERT INTO {schema}.wn_relation_fanout (term_id, relation_type, target_lemma, target_pos)
                    VALUES (%s, 'sibling_of_hypernym_parent', %s, %s) ON CONFLICT DO NOTHING""",
                (d_term["id"], target_lemma, target_pos),
            )
            count += 1
    return count


# --- contrast neighbourhood (analogy wrong options from C's own side) ---------
#
# A wrong option drawn from C's own neighbourhood through a DIFFERENT
# relation than the one tested (CELANDINE's sibling buttercup, its part
# petal, its more specific kind lesser celandine) can only be ruled out by
# reading the relation off the example pair -- options from the anchor's
# side can be ruled out just by being unrelated to C. Direct hops on the
# canonical synset only: the point is a near, recognisable word.
_CONTRAST_RELATIONS = {
    "contrast_hyponym": lambda s: s.hyponyms(),
    "contrast_hypernym": lambda s: s.hypernyms(),
    "contrast_meronym": lambda s: s.part_meronyms() + s.member_meronyms() + s.substance_meronyms(),
    "contrast_holonym": lambda s: s.part_holonyms() + s.member_holonyms() + s.substance_holonyms(),
}


def compute_contrast_fanout(conn, schema: str, wn, term: dict) -> int:
    """Direct hyponyms/hypernyms/meronyms/holonyms of a vocab term's
    canonical synset, plus its siblings (compute_sibling_fanout), as
    wn_relation_fanout rows -- the pool analogy_select draws contrast
    options from. Returns rows written."""
    if wn is None or not term.get("synset_name"):
        return 0
    try:
        synset = wn.synset(term["synset_name"])
    except Exception:  # noqa: BLE001 -- stale synset name
        return 0
    lemma_lc = term["lemma"].lower()
    rows = set()
    for rel, fn in _CONTRAST_RELATIONS.items():
        for other in fn(synset):
            for lw in _common_lemma_words(other):
                if lw != lemma_lc:
                    rows.add((rel, lw, other.pos()))
    with conn.cursor() as cur:
        cur.executemany(
            f"""INSERT INTO {schema}.wn_relation_fanout (term_id, relation_type, target_lemma, target_pos)
                VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING""",
            [(term["id"], *r) for r in rows])
    return len(rows) + compute_sibling_fanout(conn, schema, wn, term)


def backfill_contrasts(conn, schema: str, recompute: bool = False, log=print) -> dict:
    """compute_contrast_fanout for every term on either side of a verified
    edge that doesn't have it yet (C needs its neighbourhood; D needs its
    siblings, the D-level wrong options). CPU only. Marked done by a
    'contrast_scanned' row so a term with no neighbours isn't rescanned."""
    wn = _load_wordnet()
    if wn is None:
        log("[analogy-contrasts] WordNet unavailable")
        return {"terms": 0, "rows": 0}
    with conn.cursor() as cur:
        if recompute:
            cur.execute(f"""DELETE FROM {schema}.wn_relation_fanout
                            WHERE relation_type LIKE 'contrast\\_%%' OR relation_type = 'sibling_of_hypernym_parent'""")
        cur.execute(
            f"""SELECT t.id, t.word_id, t.lemma, t.wn_pos, t.synset_name, t.gloss
                FROM {schema}.wn_relation_term t
                WHERE t.synset_name IS NOT NULL
                  AND EXISTS (SELECT 1 FROM {schema}.word_relation_edge e
                              WHERE e.verification_status = 'verified' AND t.id IN (e.term_a_id, e.term_b_id))
                  AND NOT EXISTS (SELECT 1 FROM {schema}.wn_relation_fanout f
                                  WHERE f.term_id = t.id AND f.relation_type = 'contrast_scanned')
                ORDER BY t.id""")
        terms = cur.fetchall()
    log(f"[analogy-contrasts] {len(terms)} terms to scan")
    total = 0
    for i, (tid, wid, lemma, wn_pos, synset_name, gloss) in enumerate(terms, 1):
        term = {"id": tid, "word_id": wid, "lemma": lemma, "wn_pos": wn_pos, "synset_name": synset_name,
                "gloss": gloss}
        total += compute_contrast_fanout(conn, schema, wn, term)
        with conn.cursor() as cur:
            cur.execute(f"""INSERT INTO {schema}.wn_relation_fanout (term_id, relation_type, target_lemma, target_pos)
                            VALUES (%s, 'contrast_scanned', '', '') ON CONFLICT DO NOTHING""", (tid,))
        if i % 500 == 0:
            conn.commit()
            log(f"[analogy-contrasts]   ...{i}/{len(terms)}")
    conn.commit()
    return {"terms": len(terms), "rows": total}


# --- orchestrator --------------------------------------------------------------

_DEFAULT_CHUNK_SIZE = 200  # terms per extract -> verify -> write cycle, see backfill_analogies docstring


def _extract_chunk(conn, schema: str, wn, nlp, chunk: list[tuple]) -> tuple[list["RelationCandidate"], dict, dict]:
    """Extraction only (no LLM), for one chunk of `to_scan` rows -- the first
    third of backfill_analogies's per-chunk cycle. Returns (candidates,
    edges_found_by_term, method_by_term), scoped to just this chunk."""
    candidates: list[RelationCandidate] = []
    edges_found_by_term: dict[int, int] = {}
    method_by_term: dict[int, str] = {}
    for term_id, word_id, lemma, wn_pos, synset_name, gloss in chunk:
        term_row = {"id": term_id, "word_id": word_id, "lemma": lemma, "wn_pos": wn_pos,
                    "pos": _WN_POS_TO_CANONICAL.get(wn_pos, wn_pos), "synset_name": synset_name, "gloss": gloss}
        term_candidates: list[RelationCandidate] = []
        method = ""
        if synset_name:
            wn_candidates, _fanout_count = extract_wordnet_edges(conn, schema, wn, term_row)
            term_candidates.extend(wn_candidates)
            method = "wordnet"
        # Gloss-driven, not vocab-word-specific -- an ordinary/anchor term's
        # `gloss` is WordNet's own synset definition (see _get_or_create_
        # ordinary_term), and it fires the same lead-in phrasings ("of or
        # relating to X", "resembling X") a vocab word's own definition does
        # (confirmed empirically: ~11%/9%/2% match rates on a sample of
        # ordinary adjective glosses for relates_to/characterized_by/
        # resembling). Running this for ordinary terms too, not just vocab
        # ones, is what makes an ordinary<->ordinary anchor edge possible for
        # every definition-pattern family at all -- previously gated to
        # word_id is not None, which meant style B (one_hard_term) was
        # structurally dead for every definition-pattern-sourced family
        # (kind_of/agent/part_of/purpose, and now relates_to/resembling/
        # characterized_by too), since an anchor needs BOTH sides ordinary.
        if gloss:
            def_candidates = extract_definition_pattern_edges(conn, schema, wn, nlp, term_row)
            if def_candidates:
                term_candidates.extend(def_candidates)
                method = "both" if method else "definition_pattern"
        edges_found_by_term[term_id] = len(term_candidates)
        method_by_term[term_id] = method or "wordnet"
        candidates.extend(term_candidates)
    return candidates, edges_found_by_term, method_by_term


def _write_chunk(conn, schema: str, wn, cfg, chunk: list[tuple], candidates: list["RelationCandidate"],
                  edges_found_by_term: dict, method_by_term: dict) -> tuple[int, int, int]:
    """Write one chunk's already-LLM-verified candidates (word_relation_edge)
    plus its wn_relation_scan resumability markers, in one committed
    transaction -- the last third of backfill_analogies's per-chunk cycle.
    Returns (verified_count, rejected_count, sibling_rows) for this chunk."""
    verified_count = rejected_count = sibling_rows = 0
    # Re-resolve term ids for insertion (candidates carry lemmas, not ids, to
    # stay decoupled from extraction's cursor-scoped lookups); _find_term_id
    # prefers a vocab-word row over an ordinary one for the same lemma+POS.
    with conn.cursor() as cur:
        for c in candidates:
            a_wn_pos = wordnet_pos(c.term_a_pos)
            b_wn_pos = wordnet_pos(c.term_b_pos)
            if not a_wn_pos or not b_wn_pos:
                continue
            term_a_id = _find_term_id(cur, schema, c.term_a_lemma.lower(), a_wn_pos)
            term_b_id = _find_term_id(cur, schema, c.term_b_lemma.lower(), b_wn_pos)
            if term_a_id is None or term_b_id is None:
                continue
            status = c.verdict or "rejected"
            cur.execute(
                f"""INSERT INTO {schema}.word_relation_edge
                        (term_a_id, term_b_id, relation_type, relation_family, pos_a, pos_b, source,
                         verification_status, verification_note, verifier_model, verified_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (term_a_id, term_b_id, relation_type) DO NOTHING
                    RETURNING id, verification_status""",
                (term_a_id, term_b_id, c.relation_type, c.relation_family, c.term_a_pos, c.term_b_pos,
                 c.source, status, c.verification_note, getattr(cfg, "model_path", None)),
            )
            row = cur.fetchone()
            if row is not None:
                if row[1] == "verified":
                    verified_count += 1
                    if c.relation_family == "is_a":
                        cur.execute(f"SELECT id, word_id, lemma, wn_pos, synset_name, gloss "
                                    f"FROM {schema}.wn_relation_term WHERE id = %s", (term_b_id,))
                        d_row = cur.fetchone()
                        if d_row:
                            d_term = {"id": d_row[0], "word_id": d_row[1], "lemma": d_row[2],
                                      "wn_pos": d_row[3], "synset_name": d_row[4], "gloss": d_row[5]}
                            sibling_rows += compute_sibling_fanout(conn, schema, wn, d_term)
                else:
                    rejected_count += 1
        for term_id, word_id, lemma, wn_pos, synset_name, gloss in chunk:
            cur.execute(
                f"""INSERT INTO {schema}.wn_relation_scan (term_id, edges_found, method)
                    VALUES (%s, %s, %s) ON CONFLICT (term_id) DO NOTHING""",
                (term_id, edges_found_by_term.get(term_id, 0), method_by_term.get(term_id, "wordnet")),
            )
        conn.commit()
    return verified_count, rejected_count, sibling_rows


def backfill_analogies(conn, schema: str, cfg, limit: int = 0, batch_size: int = 20,
                        chunk_size: int = _DEFAULT_CHUNK_SIZE) -> dict:
    """CLI entry point (concordance backfill-analogies / maintain). Resumable
    at two grains: a term already in wn_relation_scan is skipped on a later
    run (as before), AND within a single run, terms are processed in chunks
    of `chunk_size` -- extract, LLM-verify, and WRITE (word_relation_edge +
    wn_relation_scan) each fully committed before the next chunk's extraction
    starts, rather than one extract-everything/verify-everything/write-
    everything pass across the WHOLE term list. Verification is the
    expensive, GPU-bound step; the previous single-pass design held every
    LLM-verified candidate in memory until the entire term list had been
    verified, so killing the process mid-run (as happened switching judge
    models mid-backfill) discarded however many thousands of already-
    verified edges hadn't reached the final write loop yet -- pure wasted
    GPU time. Chunking bounds that loss to one chunk's candidates, not the
    whole run. Newly discovered anchor (ordinary) terms have no wn_relation_
    scan row yet, so they're picked up automatically by the NEXT invocation
    of this same command -- no special first-run/later-run handling needed,
    the frontier just saturates after a couple of runs since a common term's
    own relation targets are overwhelmingly common too."""
    wn = _load_wordnet()
    nlp = _get_nlp()

    resolve_vocab_terms(conn, schema, wn, limit=limit)

    with conn.cursor() as cur:
        cur.execute(f"""SELECT t.id, t.word_id, t.lemma, t.wn_pos, t.synset_name, t.gloss
                        FROM {schema}.wn_relation_term t
                        LEFT JOIN {schema}.wn_relation_scan s ON s.term_id = t.id
                        WHERE s.term_id IS NULL AND (t.word_id IS NOT NULL OR t.is_common)
                        ORDER BY t.id"""
                    + (" LIMIT %s" if limit else ""),
                    (limit,) if limit else ())
        to_scan = cur.fetchall()

    chunks = [to_scan[i:i + chunk_size] for i in range(0, len(to_scan), chunk_size)]
    print(f"[backfill-analogies] {len(to_scan)} terms to scan, in {len(chunks)} chunk(s) of up to {chunk_size}...")

    total_edges_found = 0
    total_fanout = 0
    verified_count = 0
    rejected_count = 0
    sibling_rows = 0
    # Loaded once, lazily, on the first chunk that actually has candidates --
    # NOT per chunk. verify_candidates accepts an injectable `llm` for
    # exactly this reason (its own docstring: "injectable for testing"), but
    # its default (llm=None) reloads the model from scratch on every call if
    # you don't pass one -- fine for the old single-pass-over-everything
    # design (one call, one load), but chunking without this would reload a
    # multi-GB model from disk on every single chunk.
    llm = None

    for chunk_num, chunk in enumerate(chunks, 1):
        candidates, edges_found_by_term, method_by_term = _extract_chunk(conn, schema, wn, nlp, chunk)
        conn.commit()  # fanout/term rows discovered during this chunk's extraction
        total_edges_found += len(candidates)
        total_fanout += sum(edges_found_by_term.values())

        if candidates:
            if llm is None:
                llm = _load_llm(cfg)
            print(f"[backfill-analogies] chunk {chunk_num}/{len(chunks)}: {len(chunk)} terms, "
                  f"{len(candidates)} candidate edges -- verifying (batches of {batch_size})...")
            verify_candidates(cfg, candidates, batch_size=batch_size, llm=llm)

        chunk_verified, chunk_rejected, chunk_siblings = _write_chunk(
            conn, schema, wn, cfg, chunk, candidates, edges_found_by_term, method_by_term)
        verified_count += chunk_verified
        rejected_count += chunk_rejected
        sibling_rows += chunk_siblings

        print(f"[backfill-analogies] chunk {chunk_num}/{len(chunks)} written "
              f"({verified_count} verified, {rejected_count} rejected so far)")

    # Deterministic release, not left to implicit GC timing -- this is the
    # LAST maintain step, but backfill-analogies is also runnable on its own
    # (--limit / real full-corpus runs, per its own CLI docstring, "should
    # not be started while `concordance maintain` is already in flight" --
    # i.e. the two are expected to run back-to-back on the same box), and a
    # standalone rerun right after should still get a clean GPU. See
    # fill_definitions' matching comment for the live crash this pattern
    # is fixing.
    if llm is not None:
        llm.close()
    return {
        "terms_scanned": len(to_scan),
        "edges_found": total_edges_found,
        "edges_verified": verified_count,
        "edges_rejected": rejected_count,
        "fanout_rows": total_fanout,
        "sibling_rows": sibling_rows,
    }


# --- anchor review (review-analogy-anchors) ------------------------------------
#
# An edge that is a true relation can still be a bad EXAMPLE pair: the A:B
# shown in "A is to B as C is to ___" has to make the relation obvious at a
# glance, or the test-taker solves for the wrong relation. Live items showed
# minor WordNet senses (CLEAN:WEIGHTLIFTING, RESPECT:ITEM), near-synonyms
# passing as hypernyms (PUB:TAVERN, SIZE:MAGNITUDE), abbreviations (TX:US,
# U:NUCLEOTIDE) and inflections (HUMANS:HUMAN). Cheap rules catch the
# mechanical cases; the local model judges the rest. Like verify_candidates,
# silence means reject.

_ANCHOR_SHAPE = re.compile(r"^[a-z][a-z' -]*[a-z]$")
_ANCHOR_SHORT_OK = {"go", "be", "do", "ox", "ax", "up"}

_ANCHOR_SYSTEM_COMMON = (
    "You are choosing EXAMPLE pairs for vocabulary analogy questions of the form "
    "'A is to B as C is to ___'. The example pair (a, b) must teach the relationship at a glance. "
    "Judge ok=true ONLY if ALL hold: (1) the described relation clearly holds between the two words "
    "in their most common everyday meanings (the definitions show the sense meant -- reject if that "
    "sense is not the word's usual one); (2) an ordinary adult reader would see the relation "
    "immediately from the two words alone; (3) a and b are not near-synonyms or two names for the "
    "same thing, unless the relation_description says similar meaning is the relation; (4) neither "
    "word is an abbreviation, a proper name, or jargon only a specialist knows. When in doubt, reject. "
    'Output ONLY a JSON array, one object per input, in order: [{"i": <int>, "ok": <true|false>}].'
)

_ANCHOR_SYSTEM_VOCAB = (
    "You are choosing EXAMPLE pairs for advanced-vocabulary analogy questions of the form "
    "'A is to B as C is to ___'. Both words are rare on purpose. Judge ok=true ONLY if ALL hold: "
    "(1) the described relation clearly and specifically holds between the two words as defined; "
    "(2) someone who knows both words would see that relation immediately; (3) a and b are not "
    "near-synonyms or two names for the same thing, unless the relation_description says similar "
    "meaning is the relation. When in doubt, reject. "
    'Output ONLY a JSON array, one object per input, in order: [{"i": <int>, "ok": <true|false>}].'
)


_CURATED_ZIPF_MIN = 3.0  # hand-picked pairs: everyday words (broom, oven, feline) sit below the mined bar


def _anchor_rule_verdict(wn, a: dict, b: dict, ordinary: bool, curated: bool = False) -> str | None:
    """A rejection reason from the cheap rules, or None to send it to the
    model. `curated` (seed-analogy-anchors pairs) lowers the frequency bar."""
    from .analogy_select import shares_stem

    for t in (a, b):
        lc = t["lemma"].lower()
        if not (_ANCHOR_SHAPE.match(lc) and (len(lc) >= 3 or lc in _ANCHOR_SHORT_OK)):
            return "abbreviation or symbol"
    if a["pos"] == b["pos"] and shares_stem(a["lemma"], b["lemma"]):
        return "same word in another form"
    if ordinary:
        common = ((a["zipf"] or 0) >= _CURATED_ZIPF_MIN and (b["zipf"] or 0) >= _CURATED_ZIPF_MIN) if curated \
            else (a["is_common"] and b["is_common"])
        if not common:
            return "rare word"
        if wn is not None:
            for t in (a, b):
                if not t["synset"]:
                    continue
                try:
                    names = [n.replace("_", " ").lower() for n in wn.synset(t["synset"]).lemma_names()]
                except Exception:  # noqa: BLE001 -- unknown synset name: skip this check
                    names = []
                lc = t["lemma"].lower()
                if any(n != lc and " " not in n and n.startswith(lc) for n in names):
                    return f"clipped form ({lc})"  # info -> information, auto -> automobile
                synsets = wn.synsets(t["lemma"].replace(" ", "_"), "a" if t["wn_pos"] in ("a", "s") else t["wn_pos"])
                if not synsets or synsets[0].name() != t["synset"]:
                    return f"minor sense of {t['lemma']}"
    return None


def _anchor_query(llm, batch: list[dict], vocab: bool) -> str:
    items = [{"i": i, "relation_description": _ANCHOR_DESCRIPTIONS.get(e["relation_type"], e["relation_type"]),
              "a": e["a"]["lemma"], "a_def": (e["a"]["gloss"] or "")[:200],
              "b": e["b"]["lemma"], "b_def": (e["b"]["gloss"] or "")[:200]}
             for i, e in enumerate(batch)]
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": _ANCHOR_SYSTEM_VOCAB if vocab else _ANCHOR_SYSTEM_COMMON},
                  {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
        temperature=0.0,
        max_tokens=len(batch) * 12 + 64,
    )
    return out["choices"][0]["message"]["content"]


def review_anchors(conn, schema: str, cfg, limit: int = 0, recheck: bool = False, batch_size: int = 20,
                   llm=None, log=print) -> dict:
    """Mark every verified edge that could serve as an anchor (both terms
    ordinary, or both vocab -- the two anchor pools analogy_select draws
    from) anchor_status 'ok'/'rejected' with anchor_note saying why.
    Resumable: only unreviewed edges unless `recheck`. Commits per batch."""
    wn = _load_wordnet()
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT e.id, e.relation_type, e.pos_a, e.pos_b, ta.word_id IS NULL,
                       ta.lemma, ta.wn_pos, ta.synset_name, ta.gloss, ta.is_common,
                       tb.lemma, tb.wn_pos, tb.synset_name, tb.gloss, tb.is_common,
                       e.source = 'curated_anchor', ta.zipf, tb.zipf
                FROM {schema}.word_relation_edge e
                JOIN {schema}.wn_relation_term ta ON ta.id = e.term_a_id
                JOIN {schema}.wn_relation_term tb ON tb.id = e.term_b_id
                WHERE e.verification_status = 'verified'
                  AND (ta.word_id IS NULL) = (tb.word_id IS NULL)
                  {"" if recheck else "AND e.anchor_status IS NULL"}
                ORDER BY e.id""" + (" LIMIT %s" if limit else ""),
            (limit,) if limit else (),
        )
        rows = cur.fetchall()

    edges = []
    for (eid, rtype, pos_a, pos_b, ordinary, la, wa, sa, ga, ca, lb, wb, sb, gb, cb, curated, za, zb) in rows:
        edges.append({"id": eid, "relation_type": rtype, "ordinary": ordinary, "curated": curated,
                      "a": {"lemma": la, "pos": pos_a, "wn_pos": wa, "synset": sa, "gloss": ga, "is_common": ca,
                            "zipf": za},
                      "b": {"lemma": lb, "pos": pos_b, "wn_pos": wb, "synset": sb, "gloss": gb, "is_common": cb,
                            "zipf": zb}})

    stats = {"reviewed": 0, "ok": 0, "rejected_rule": 0, "rejected_model": 0}

    def _write(cur, updates: list[tuple[str, str | None, int]]) -> None:
        cur.executemany(f"""UPDATE {schema}.word_relation_edge
                            SET anchor_status = %s, anchor_note = %s, anchor_checked_at = now() WHERE id = %s""",
                        updates)

    to_model: dict[bool, list[dict]] = {True: [], False: []}
    rule_updates = []
    for e in edges:
        reason = _anchor_rule_verdict(wn, e["a"], e["b"], e["ordinary"], e["curated"])
        if reason:
            rule_updates.append(("rejected", reason, e["id"]))
        else:
            to_model[e["ordinary"]].append(e)
    with conn.cursor() as cur:
        _write(cur, rule_updates)
    conn.commit()
    stats["rejected_rule"] = len(rule_updates)
    stats["reviewed"] += len(rule_updates)
    n_model = len(to_model[True]) + len(to_model[False])
    log(f"[review-analogy-anchors] {len(edges)} edges: {len(rule_updates)} rejected by rules, "
        f"{n_model} to the model")

    if n_model:
        own_llm = llm is None
        llm = llm or _load_llm(cfg)
        try:
            done = 0
            for ordinary, pool in to_model.items():
                for i in range(0, len(pool), batch_size):
                    batch = pool[i:i + batch_size]
                    verdicts = None
                    for _ in range(_MAX_VERIFY_PASSES):
                        try:
                            verdicts = _parse_verify_response(_anchor_query(llm, batch, vocab=not ordinary))
                            break
                        except Exception:  # noqa: BLE001 -- a flaky call shouldn't end the run
                            verdicts = None
                    by_i = {v["i"]: v for v in (verdicts or []) if isinstance(v, dict) and isinstance(v.get("i"), int)}
                    updates = []
                    for j, e in enumerate(batch):
                        v = by_i.get(j)
                        if v is not None and bool(v.get("ok", False)):
                            updates.append(("ok", None, e["id"]))
                            stats["ok"] += 1
                        else:
                            note = "model: unclear example" if v is not None else "model: no verdict"
                            updates.append(("rejected", note, e["id"]))
                            stats["rejected_model"] += 1
                    with conn.cursor() as cur:
                        _write(cur, updates)
                    conn.commit()
                    stats["reviewed"] += len(batch)
                    done += len(batch)
                    if done % (batch_size * 10) == 0 or done == n_model:
                        log(f"[review-analogy-anchors]   ...{done}/{n_model} judged ({stats['ok']} ok so far)")
        finally:
            if own_llm:
                llm.close()
    return stats


# --- anchor clarity (score-analogy-anchors) ------------------------------------
#
# ok/rejected is too coarse: TOUGH:FIGHTER and WARD:SOMEBODY pass review but
# teach the relation far less well than CURE:MEDICATION. A 1-5 score lets
# analogy_select prefer the clearest pairs (>= 4) and drop the vague ones
# (<= 2) without re-litigating the binary review.

_CLARITY_SYSTEM = (
    "You rate EXAMPLE pairs for vocabulary analogy questions 'A is to B as C is to ___'. The example pair "
    "must show a relationship so plainly that a reader applies the same relationship to C. For each pair, "
    "given the intended relation, score 1-5: 5 = textbook example, both words familiar, relation obvious "
    "at a glance (dog/animal for 'kind of'); 4 = clear; 3 = correct but takes a moment; 2 = loose, vague or "
    "relies on an unusual sense; 1 = misleading, or the pair suggests a different relation. "
    'Output ONLY a JSON array, one object per input, in order: [{"i": <int>, "score": <1-5>}].'
)


def score_anchors(conn, schema: str, cfg, limit: int = 0, rescore: bool = False, batch_size: int = 20,
                  llm=None, log=print) -> dict:
    """Give every review-ok anchor edge a 1-5 anchor_clarity. Resumable
    (only unscored unless `rescore`); commits per batch."""
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT e.id, e.relation_type, ta.lemma, ta.gloss, tb.lemma, tb.gloss
                FROM {schema}.word_relation_edge e
                JOIN {schema}.wn_relation_term ta ON ta.id = e.term_a_id
                JOIN {schema}.wn_relation_term tb ON tb.id = e.term_b_id
                WHERE e.verification_status = 'verified' AND e.anchor_status = 'ok'
                  {"" if rescore else "AND e.anchor_clarity IS NULL"}
                ORDER BY e.id""" + (" LIMIT %s" if limit else ""),
            (limit,) if limit else ())
        rows = cur.fetchall()
    stats = {"scored": 0, "no_verdict": 0, "dist": {}}
    if not rows:
        return stats
    own_llm = llm is None
    llm = llm or _load_llm(cfg)
    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            items = [{"i": j, "relation": _ANCHOR_DESCRIPTIONS.get(rt, rt), "a": la, "a_def": (ga or "")[:160],
                      "b": lb, "b_def": (gb or "")[:160]} for j, (_, rt, la, ga, lb, gb) in enumerate(batch)]
            verdicts = None
            for _ in range(_MAX_VERIFY_PASSES):
                try:
                    out = llm.create_chat_completion(
                        messages=[{"role": "system", "content": _CLARITY_SYSTEM},
                                  {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
                        temperature=0.0, max_tokens=len(batch) * 14 + 64)
                    verdicts = _parse_verify_response(out["choices"][0]["message"]["content"])
                    break
                except Exception:  # noqa: BLE001
                    verdicts = None
            by_i = {v["i"]: v for v in (verdicts or []) if isinstance(v, dict) and isinstance(v.get("i"), int)}
            updates = []
            for j, row in enumerate(batch):
                try:
                    score = int(by_i[j]["score"])
                except (KeyError, TypeError, ValueError):
                    stats["no_verdict"] += 1
                    continue
                score = min(5, max(1, score))
                stats["dist"][score] = stats["dist"].get(score, 0) + 1
                updates.append((score, row[0]))
            with conn.cursor() as cur:
                cur.executemany(f"UPDATE {schema}.word_relation_edge SET anchor_clarity = %s WHERE id = %s", updates)
            conn.commit()
            stats["scored"] += len(updates)
            if (i // batch_size) % 10 == 0:
                log(f"[score-analogy-anchors]   ...{i + len(batch)}/{len(rows)} ({stats['dist']})")
    finally:
        if own_llm:
            llm.close()
    return stats
