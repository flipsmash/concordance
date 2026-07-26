"""Analogy relation extraction + verification backfill (§ analogies).

Populates word_relation_edge (and its supporting wn_relation_term /
wn_relation_fanout / wn_relation_scan tables, see concordance/db.py's
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
purpose / attribute) is the bucket analogy_select.py pairs an edge against a
DIFFERENT edge of the same family to build the anchor (A:B) leg of an item.
"""

from __future__ import annotations

import json
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
            # Fanout: transitive closure (unioned over ALL this lemma's synsets)
            # for the families where a distractor N hops away is still a valid
            # completion of the relation; direct targets only otherwise.
            fanout_targets: set[tuple[str, str]] = set()
            synsets_to_walk = wn.synsets(lemma_lc, pos=term_row["wn_pos"]) if rel_type in _TRANSITIVE_FAMILIES \
                else [synset]
            for s in synsets_to_walk:
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

            # Candidates: direct hop only, from the canonical synset.
            for other in rel_fn(synset):
                for lw in _lemma_words(other):
                    if lw == lemma_lc:
                        continue
                    other_term = _get_or_create_ordinary_term(cur, schema, wn, lw, other.pos())
                    if other_term is None:
                        continue
                    candidates.append(RelationCandidate(
                        term_a_lemma=term_row["lemma"], term_a_pos=term_row["pos"],
                        term_a_gloss=term_row["gloss"],
                        term_b_lemma=other_term["lemma"], term_b_pos=_WN_POS_TO_CANONICAL.get(other.pos(), other.pos()),
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
    return matcher


_PATTERN_LABEL_TO_RELATION = {
    "kind_of": "definition_pattern_kind_of",
    "agent": "definition_pattern_agent",
    "part_of": "definition_pattern_part_of",
    "purpose": "definition_pattern_purpose",
}

_matcher_singleton = None


def _get_matcher(nlp):
    global _matcher_singleton
    if _matcher_singleton is None:
        _matcher_singleton = _build_matchers(nlp)
    return _matcher_singleton


def extract_definition_pattern_edges(conn, schema: str, wn, nlp, term_row: dict) -> list[RelationCandidate]:
    """spaCy Matcher over the vocab word's own definition text -- recovers
    relations WordNet doesn't have for exactly the words WordNet is thinnest
    on. Only emits a candidate when the parsed head term resolves to a
    WordNet synset (needed for the exclusion-set fallback in
    analogy_select.py when term_a itself has no synset)."""
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


_VERIFY_SYSTEM = (
    "You are checking candidate word-analogy relation pairs for a vocabulary quiz. "
    "For each pair, word 'b' is claimed to stand in the relationship named by 'relation' "
    "to word 'a', judged strictly from the two definitions given -- not from general "
    "knowledge of the words. Judge ok=true ONLY if the relation genuinely and specifically "
    "holds. Reject (ok=false) if the relation doesn't actually hold, OR if 'a' and 'b' are "
    "just near-synonyms / restatements of each other (a valid analogy relation must be a "
    "specific, nameable, non-synonymous relationship, not a paraphrase). When genuinely "
    "unsure, reject. "
    'Output ONLY a JSON array, one object per input, in the same order: '
    '[{"i": <int>, "ok": <true|false>}]. Include every input exactly once.'
)


def _verify_query(llm, batch: list[RelationCandidate]) -> str:
    items = [
        {"i": i, "relation": c.relation_type, "a": c.term_a_lemma, "a_def": c.term_a_gloss[:200],
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
            for lw in _lemma_words(sibling):
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


# --- orchestrator --------------------------------------------------------------

def backfill_analogies(conn, schema: str, cfg, limit: int = 0, batch_size: int = 20) -> dict:
    """CLI entry point (concordance backfill-analogies / maintain). Resumable:
    a term already in wn_relation_scan is skipped on later runs. Newly
    discovered anchor (ordinary) terms have no wn_relation_scan row yet, so
    they're picked up automatically by the NEXT invocation of this same
    command -- no special first-run/later-run handling needed, the frontier
    just saturates after a couple of runs since a common term's own relation
    targets are overwhelmingly common too."""
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

    all_candidates: list[RelationCandidate] = []
    edges_found_by_term: dict[int, int] = {}
    method_by_term: dict[int, str] = {}

    print(f"[backfill-analogies] extracting relations for {len(to_scan)} terms...")
    for i, (term_id, word_id, lemma, wn_pos, synset_name, gloss) in enumerate(to_scan, 1):
        term_row = {"id": term_id, "word_id": word_id, "lemma": lemma, "wn_pos": wn_pos,
                    "pos": _WN_POS_TO_CANONICAL.get(wn_pos, wn_pos), "synset_name": synset_name, "gloss": gloss}
        candidates: list[RelationCandidate] = []
        method = ""
        if synset_name:
            wn_candidates, _fanout_count = extract_wordnet_edges(conn, schema, wn, term_row)
            candidates.extend(wn_candidates)
            method = "wordnet"
        if word_id is not None:   # definition-pattern extraction only makes sense for a real vocab definition
            def_candidates = extract_definition_pattern_edges(conn, schema, wn, nlp, term_row)
            if def_candidates:
                candidates.extend(def_candidates)
                method = "both" if method else "definition_pattern"
        edges_found_by_term[term_id] = len(candidates)
        method_by_term[term_id] = method or "wordnet"
        all_candidates.extend(candidates)
        # Periodic commit: extraction inserts wn_relation_term/wn_relation_fanout
        # rows as it discovers ordinary (anchor-bank) terms, and without this the
        # entire extraction phase (which can run over thousands of terms) stays
        # in one uncommitted transaction until backfill_analogies's own final
        # commit -- a crash or kill partway through would lose all of it, the
        # same failure mode classify_and_store's own docstring describes fixing.
        if i % 100 == 0:
            conn.commit()
            print(f"[backfill-analogies]   ...{i}/{len(to_scan)} terms scanned "
                  f"({len(all_candidates)} candidate edges found so far)")
    conn.commit()

    print(f"[backfill-analogies] verifying {len(all_candidates)} candidate edges "
          f"(batches of {batch_size})...")
    verify_candidates(cfg, all_candidates, batch_size=batch_size)

    verified_count = 0
    rejected_count = 0
    sibling_rows = 0

    # Re-resolve term ids for insertion (candidates carry lemmas, not ids, to
    # stay decoupled from extraction's cursor-scoped lookups above); _find_term_id
    # prefers a vocab-word row over an ordinary one for the same lemma+POS.
    print(f"[backfill-analogies] writing {len(all_candidates)} verified/rejected edges...")
    with conn.cursor() as cur:
        for i, c in enumerate(all_candidates, 1):
            if i % 200 == 0:
                conn.commit()
                print(f"[backfill-analogies]   ...{i}/{len(all_candidates)} edges written "
                      f"({verified_count} verified, {rejected_count} rejected so far)")
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
        for term_id, word_id, lemma, wn_pos, synset_name, gloss in to_scan:
            cur.execute(
                f"""INSERT INTO {schema}.wn_relation_scan (term_id, edges_found, method)
                    VALUES (%s, %s, %s) ON CONFLICT (term_id) DO NOTHING""",
                (term_id, edges_found_by_term.get(term_id, 0), method_by_term.get(term_id, "wordnet")),
            )
        conn.commit()

    return {
        "terms_scanned": len(to_scan),
        "edges_found": len(all_candidates),
        "edges_verified": verified_count,
        "edges_rejected": rejected_count,
        "fanout_rows": sum(edges_found_by_term.values()),
        "sibling_rows": sibling_rows,
    }
