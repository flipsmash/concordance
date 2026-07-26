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
"""

from __future__ import annotations

import random
from dataclasses import dataclass


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


def _term_row(cur, schema: str, word_id: int) -> dict | None:
    cur.execute(f"SELECT id, word_id, lemma, wn_pos, synonym_lemmas FROM {schema}.wn_relation_term "
                f"WHERE word_id = %s", (word_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return {"id": row[0], "word_id": row[1], "lemma": row[2], "wn_pos": row[3], "synonym_lemmas": row[4] or []}


def _tested_edge_candidates(cur, schema: str, target_term_id: int, style: str, exclude_word_ids: set[int]) -> list[dict]:
    """Verified edges with target as term_a, partitioned by style: style
    'vocab_only' wants term_b to be a vocab word, style 'one_hard_term' wants
    term_b to be ordinary (word_id IS NULL)."""
    cur.execute(
        f"""SELECT e.id, e.term_b_id, e.relation_type, e.relation_family, e.pos_b,
                   tb.word_id, tb.lemma, tb.wn_pos, tb.synonym_lemmas
            FROM {schema}.word_relation_edge e
            JOIN {schema}.wn_relation_term tb ON tb.id = e.term_b_id
            WHERE e.term_a_id = %s AND e.verification_status = 'verified'""",
        (target_term_id,),
    )
    rows = cur.fetchall()
    out = []
    for edge_id, term_b_id, relation_type, relation_family, pos_b, tb_word_id, tb_lemma, tb_wn_pos, tb_syns in rows:
        is_vocab_b = tb_word_id is not None
        if style == "vocab_only" and not is_vocab_b:
            continue
        if style == "one_hard_term" and is_vocab_b:
            continue
        if tb_word_id is not None and tb_word_id in exclude_word_ids:
            continue
        out.append({"edge_id": edge_id, "term_b_id": term_b_id, "relation_type": relation_type,
                    "relation_family": relation_family, "pos_b": pos_b, "d_word_id": tb_word_id,
                    "d_lemma": tb_lemma, "d_wn_pos": tb_wn_pos, "d_synonyms": tb_syns or []})
    random.shuffle(out)
    return out


def _find_anchor(cur, schema: str, relation_family: str, style: str,
                  exclude_term_ids: set[int], exclude_word_ids: set[int]) -> tuple[str, str] | None:
    """A DIFFERENT verified edge, same relation_family, matching style
    (vocab<->vocab for style A / ordinary<->ordinary for style B), sharing no
    term with the tested edge. Returns (a_lemma, b_lemma) or None."""
    vocab_clause = "IS NOT NULL" if style == "vocab_only" else "IS NULL"
    cur.execute(
        f"""SELECT ta.id, tb.id, ta.lemma, tb.lemma, ta.word_id, tb.word_id
            FROM {schema}.word_relation_edge e
            JOIN {schema}.wn_relation_term ta ON ta.id = e.term_a_id
            JOIN {schema}.wn_relation_term tb ON tb.id = e.term_b_id
            WHERE e.relation_family = %s AND e.verification_status = 'verified'
              AND ta.word_id {vocab_clause} AND tb.word_id {vocab_clause}
            ORDER BY random() LIMIT 40""",
        (relation_family,),
    )
    for ta_id, tb_id, ta_lemma, tb_lemma, ta_word_id, tb_word_id in cur.fetchall():
        if ta_id in exclude_term_ids or tb_id in exclude_term_ids:
            continue
        if (ta_word_id is not None and ta_word_id in exclude_word_ids) or \
           (tb_word_id is not None and tb_word_id in exclude_word_ids):
            continue
        return ta_lemma, tb_lemma
    return None


def _fanout_lemmas(cur, schema: str, term_id: int, relation_type: str) -> set[str]:
    cur.execute(f"SELECT target_lemma FROM {schema}.wn_relation_fanout WHERE term_id = %s AND relation_type = %s",
                (term_id, relation_type))
    return {r[0] for r in cur.fetchall()}


def _exclusion_lemmas(cur, schema: str, c_term_id: int, d_term_id: int, d_lemma: str,
                       d_synonyms: list[str], relation_type: str) -> set[str]:
    excl = _fanout_lemmas(cur, schema, c_term_id, relation_type)
    if not excl:
        # C has no fanout for this relation (no synset, e.g. a definition-pattern-only
        # edge) -- D's own closure is the provably-equivalent fallback, see module docstring.
        excl = _fanout_lemmas(cur, schema, d_term_id, relation_type)
    excl.add(d_lemma.lower())
    excl |= {s.lower() for s in d_synonyms}
    return excl


def select_analogy_edge(conn, schema: str, target_word_id: int, exclude_ids: set[int]) -> "AnalogyAssembly | None":
    with conn.cursor() as cur:
        target_term = _term_row(cur, schema, target_word_id)
        if target_term is None:
            return None

        styles = ["vocab_only", "one_hard_term"]
        random.shuffle(styles)
        for style in styles:
            tested_candidates = _tested_edge_candidates(cur, schema, target_term["id"], style, exclude_ids)
            for tested in tested_candidates:
                anchor = _find_anchor(
                    cur, schema, tested["relation_family"], style,
                    exclude_term_ids={target_term["id"], tested["term_b_id"]},
                    exclude_word_ids=exclude_ids,
                )
                if anchor is None:
                    continue
                a_lemma, b_lemma = anchor

                exclusion_lemmas = _exclusion_lemmas(
                    cur, schema, target_term["id"], tested["term_b_id"], tested["d_lemma"],
                    tested["d_synonyms"], tested["relation_type"],
                )
                # Trap lemmas: real completions of the SAME relation via A or B
                # instead of C -- looked up by lemma+pos since _find_anchor only
                # returned lemma strings, not term ids.
                trap_lemmas: set[str] = set()
                for lemma in (a_lemma, b_lemma):
                    cur.execute(
                        f"SELECT id FROM {schema}.wn_relation_term WHERE lemma_lc = %s ORDER BY word_id IS NULL LIMIT 1",
                        (lemma.lower(),),
                    )
                    row = cur.fetchone()
                    if row:
                        trap_lemmas |= _fanout_lemmas(cur, schema, row[0], tested["relation_type"])
                trap_lemmas -= exclusion_lemmas
                trap_lemmas -= {a_lemma.lower(), b_lemma.lower(), target_term["lemma"].lower(),
                                tested["d_lemma"].lower()}

                return AnalogyAssembly(
                    style=style, a_lemma=a_lemma, b_lemma=b_lemma, c_lemma=target_term["lemma"],
                    relation_type=tested["relation_type"], target_word_id=target_word_id,
                    d_term_id=tested["term_b_id"], d_word_id=tested["d_word_id"], d_lemma=tested["d_lemma"],
                    d_pos=tested["pos_b"], exclusion_lemmas=exclusion_lemmas, trap_lemmas=sorted(trap_lemmas),
                )
    return None
