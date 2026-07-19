"""Distractor generation for quiz questions (§ quizzing).

Given a target word, produces plausible-but-wrong option words for multiple-choice,
a foil word for true/false, or a set of companion words for matching -- all POS-matched
(non-negotiable) and drawn from a weighted blend of strategies:

  orthographic  looks like the target (pg_trgm lemma similarity) -- only meaningful
                when the *word* is what a quiz-taker chooses among, not its definition
  semantic      near-miss embedding proximity: close enough to be a plausible mix-up,
                far enough not to be a true synonym (a distance *band*, not nearest-only)
  domain        shares a top-level USAS category with the target
  antonym       stubbed -- no antonym data exists anywhere in this pipeline yet; the
                weight key is reserved so a future data source is a drop-in, not a
                config-shape migration
  random        any other eligible word -- also the universal fallback for every
                other strategy's shortfall

Generated live at quiz-start time (not pre-cached): selection depends on per-session
parameters (difficulty range, filters, ratios) that don't cleanly cache across configs,
and this is the same query shape webapp/backend/main.py's word_neighbors already proves
fast enough per-word at this corpus size.

Fallback rule: POS and the difficulty band are never relaxed to fill a shortfall -- a
strategy's deficit spills to the next-weighted strategy, then to random. Only if random
itself can't fill the count under POS+difficulty (a pathologically narrow config) does
the difficulty band widen symmetrically as a last resort, which the caller can see via
DistractorResult.degraded rather than a silent shortfall.

A target's own synonyms are always excluded, from every strategy -- a distractor that's
actually a valid synonym of the correct answer isn't wrong, it's a second correct answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# Cosine-distance band for the "semantic near-miss" strategy: too close (< _SEMANTIC_BAND_MIN)
# risks being a near-synonym even after the explicit synonym exclusion below; too far
# (> _SEMANTIC_BAND_MAX) isn't a plausible mix-up anymore. Starting heuristic, expected to
# need empirical tuning once real quiz data exists -- not an architectural commitment.
_SEMANTIC_BAND_MIN = 0.15
_SEMANTIC_BAND_MAX = 0.45

# Minimum pg_trgm similarity for a lemma to count as an "orthographic lookalike" at all,
# rather than just a coincidental trigram overlap.
_ORTHOGRAPHIC_MIN_SIMILARITY = 0.3

# How far a last-resort difficulty-band widen reaches, symmetrically, when even random
# can't fill the requested count under the original band.
_DEGRADED_WIDEN_POINTS = 15.0

_SIGNAL_COLUMNS = ("definition_vector", "fasttext_vector")


@dataclass
class DistractorConfig:
    difficulty_min: float | None = None
    difficulty_max: float | None = None
    # fraction of the requested count drawn from smart strategies vs. random
    smart_vs_random_ratio: float = 0.7
    # relative weights among the smart strategies; a zero/absent key contributes nothing
    # (e.g. the caller zeroes 'orthographic' under the word_to_definition direction, where
    # a lookalike *word* is invisible since only definitions are shown as options)
    strategy_weights: dict[str, float] = field(
        default_factory=lambda: {"orthographic": 1 / 3, "semantic": 1 / 3, "domain": 1 / 3, "antonym": 0.0}
    )


@dataclass
class DistractorResult:
    candidates: list[dict]  # each: {id, lemma, quiz_definition, strategy}
    degraded: bool = False  # True if the difficulty band had to widen to fill the count,
                             # or the count still couldn't be fully filled


def _target_info(conn, schema: str, word_id: int) -> tuple[str, list[str]]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT lemma, synonyms FROM {schema}.word WHERE id = %s", (word_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"word {word_id} not found")
    return row[0], list(row[1] or [])


def _base_filters(pos: str, cfg: DistractorConfig, exclude_ids: set[int], exclude_lemmas: list[str],
                   require_quiz_definition: bool) -> tuple[list[str], list]:
    filters = ["w.active", "wd.quizzable = true", "w.part_of_speech = %s", "NOT (w.id = ANY(%s))"]
    params: list = [pos, list(exclude_ids)]
    if exclude_lemmas:
        filters.append("NOT (lower(w.lemma) = ANY(%s))")
        params.append([lem.lower() for lem in exclude_lemmas])
    if cfg.difficulty_min is not None:
        filters.append("wd.difficulty >= %s")
        params.append(cfg.difficulty_min)
    if cfg.difficulty_max is not None:
        filters.append("wd.difficulty <= %s")
        params.append(cfg.difficulty_max)
    if require_quiz_definition:
        filters.append("w.quiz_definition IS NOT NULL")
    return filters, params


def _orthographic_candidates(conn, schema, target_lemma, pos, cfg, exclude_ids, exclude_lemmas,
                              require_quiz_definition, limit) -> list[dict]:
    filters, params = _base_filters(pos, cfg, exclude_ids, exclude_lemmas, require_quiz_definition)
    filters.append("similarity(w.lemma, %s) > %s")
    params.extend([target_lemma, _ORTHOGRAPHIC_MIN_SIMILARITY])
    where = " AND ".join(filters)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT w.id, w.lemma, w.quiz_definition
                FROM {schema}.word w
                JOIN {schema}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
                ORDER BY similarity(w.lemma, %s) DESC
                LIMIT %s""",
            (*params, target_lemma, limit),
        )
        rows = cur.fetchall()
    return [{"id": r[0], "lemma": r[1], "quiz_definition": r[2]} for r in rows]


def _semantic_band_candidates(conn, schema, target_word_id, pos, cfg, exclude_ids, exclude_lemmas,
                               require_quiz_definition, limit) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT definition_vector, fasttext_vector FROM {schema}.word_embedding WHERE word_id = %s",
            (target_word_id,),
        )
        row = cur.fetchone()
    if row is None:
        return []
    vec_col = None
    for i, col in enumerate(_SIGNAL_COLUMNS):
        if row[i] is not None:
            vec_col = col
            break
    if vec_col is None:
        return []

    filters, params = _base_filters(pos, cfg, exclude_ids, exclude_lemmas, require_quiz_definition)
    filters.append(f"e.{vec_col} IS NOT NULL")
    filters.append(f"(e.{vec_col} <=> (SELECT {vec_col} FROM {schema}.word_embedding WHERE word_id = %s)) BETWEEN %s AND %s")
    params.extend([target_word_id, _SEMANTIC_BAND_MIN, _SEMANTIC_BAND_MAX])
    where = " AND ".join(filters)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT w.id, w.lemma, w.quiz_definition,
                       e.{vec_col} <=> (SELECT {vec_col} FROM {schema}.word_embedding WHERE word_id = %s) AS distance
                FROM {schema}.word_embedding e
                JOIN {schema}.word w ON w.id = e.word_id
                JOIN {schema}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
                ORDER BY distance
                LIMIT %s""",
            (target_word_id, *params, limit),
        )
        rows = cur.fetchall()
    return [{"id": r[0], "lemma": r[1], "quiz_definition": r[2]} for r in rows]


def _domain_candidates(conn, schema, target_word_id, pos, cfg, exclude_ids, exclude_lemmas,
                        require_quiz_definition, limit) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT DISTINCT left(c.code, 1) FROM {schema}.word_category wc
                JOIN {schema}.category c ON c.id = wc.category_id
                WHERE wc.word_id = %s""",
            (target_word_id,),
        )
        target_fields = [r[0] for r in cur.fetchall()]
    if not target_fields:
        return []

    filters, params = _base_filters(pos, cfg, exclude_ids, exclude_lemmas, require_quiz_definition)
    filters.append(
        f"""EXISTS (SELECT 1 FROM {schema}.word_category wc2
                    JOIN {schema}.category c2 ON c2.id = wc2.category_id
                    WHERE wc2.word_id = w.id AND left(c2.code, 1) = ANY(%s))"""
    )
    params.append(target_fields)
    where = " AND ".join(filters)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT w.id, w.lemma, w.quiz_definition
                FROM {schema}.word w
                JOIN {schema}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
                ORDER BY random()
                LIMIT %s""",
            (*params, limit),
        )
        rows = cur.fetchall()
    return [{"id": r[0], "lemma": r[1], "quiz_definition": r[2]} for r in rows]


def _antonym_candidates(conn, schema, target_word_id, pos, cfg, exclude_ids, exclude_lemmas,
                         require_quiz_definition, limit) -> list[dict]:
    # No antonym data source exists anywhere in this pipeline (confirmed via full-repo
    # grep during planning). Reserved so a future data source is a config drop-in.
    return []


def _random_candidates(conn, schema, pos, cfg, exclude_ids, exclude_lemmas,
                        require_quiz_definition, limit) -> list[dict]:
    filters, params = _base_filters(pos, cfg, exclude_ids, exclude_lemmas, require_quiz_definition)
    where = " AND ".join(filters)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT w.id, w.lemma, w.quiz_definition
                FROM {schema}.word w
                JOIN {schema}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
                ORDER BY random()
                LIMIT %s""",
            (*params, limit),
        )
        rows = cur.fetchall()
    return [{"id": r[0], "lemma": r[1], "quiz_definition": r[2]} for r in rows]


_STRATEGY_FUNCS = {
    "orthographic": lambda conn, schema, wid, lemma, pos, cfg, ex_ids, ex_lem, rqd, limit:
        _orthographic_candidates(conn, schema, lemma, pos, cfg, ex_ids, ex_lem, rqd, limit),
    "semantic": lambda conn, schema, wid, lemma, pos, cfg, ex_ids, ex_lem, rqd, limit:
        _semantic_band_candidates(conn, schema, wid, pos, cfg, ex_ids, ex_lem, rqd, limit),
    "domain": lambda conn, schema, wid, lemma, pos, cfg, ex_ids, ex_lem, rqd, limit:
        _domain_candidates(conn, schema, wid, pos, cfg, ex_ids, ex_lem, rqd, limit),
    "antonym": lambda conn, schema, wid, lemma, pos, cfg, ex_ids, ex_lem, rqd, limit:
        _antonym_candidates(conn, schema, wid, pos, cfg, ex_ids, ex_lem, rqd, limit),
}


def _select_candidates(conn, schema: str, target_word_id: int, pos: str, cfg: DistractorConfig,
                        count: int, exclude_word_ids: set[int],
                        require_quiz_definition: bool) -> DistractorResult:
    target_lemma, synonyms = _target_info(conn, schema, target_word_id)
    exclude_ids = set(exclude_word_ids) | {target_word_id}
    picked: list[dict] = []

    smart_count = round(count * cfg.smart_vs_random_ratio)
    active_weights = {k: v for k, v in cfg.strategy_weights.items() if v and v > 0 and k in _STRATEGY_FUNCS}
    total_w = sum(active_weights.values()) or 1.0
    remaining_smart = smart_count

    for strat, weight in sorted(active_weights.items(), key=lambda kv: -kv[1]):
        if remaining_smart <= 0:
            break
        alloc = min(round(smart_count * (weight / total_w)), remaining_smart)
        if alloc <= 0:
            continue
        got = _STRATEGY_FUNCS[strat](conn, schema, target_word_id, target_lemma, pos, cfg,
                                      exclude_ids, synonyms, require_quiz_definition, alloc)
        for c in got:
            if c["id"] in exclude_ids:
                continue
            c["strategy"] = strat
            picked.append(c)
            exclude_ids.add(c["id"])
        remaining_smart -= len(got)

    random_count = (count - len(picked))
    degraded = False
    if random_count > 0:
        got = _random_candidates(conn, schema, pos, cfg, exclude_ids, synonyms,
                                  require_quiz_definition, random_count)
        for c in got:
            c["strategy"] = "random"
            picked.append(c)
            exclude_ids.add(c["id"])
        random_count -= len(got)

    if random_count > 0 and (cfg.difficulty_min is not None or cfg.difficulty_max is not None):
        widened = replace(
            cfg,
            difficulty_min=None if cfg.difficulty_min is None else max(0.0, cfg.difficulty_min - _DEGRADED_WIDEN_POINTS),
            difficulty_max=None if cfg.difficulty_max is None else min(100.0, cfg.difficulty_max + _DEGRADED_WIDEN_POINTS),
        )
        got = _random_candidates(conn, schema, pos, widened, exclude_ids, synonyms,
                                  require_quiz_definition, random_count)
        for c in got:
            c["strategy"] = "random"
            picked.append(c)
            exclude_ids.add(c["id"])
        random_count -= len(got)
        degraded = True

    return DistractorResult(candidates=picked, degraded=degraded or random_count > 0)


def select_mc_distractors(conn, schema: str, target_word_id: int, pos: str, cfg: DistractorConfig,
                           count: int, exclude_word_ids: set[int] | None = None,
                           require_quiz_definition: bool = False) -> DistractorResult:
    """Up to `count` distractor words for a multiple-choice question about
    target_word_id. `require_quiz_definition=True` under the word_to_definition
    direction, where each option's quiz_definition (not just its lemma) is shown."""
    return _select_candidates(conn, schema, target_word_id, pos, cfg, count,
                               exclude_word_ids or set(), require_quiz_definition)


def select_tf_foil(conn, schema: str, target_word_id: int, pos: str,
                    cfg: DistractorConfig, exclude_word_ids: set[int] | None = None) -> dict | None:
    """One word whose quiz_definition will be shown as the false statement for a
    true/false question about target_word_id. None only if the corpus genuinely
    has no eligible word left (pathological config)."""
    result = _select_candidates(conn, schema, target_word_id, pos, cfg, 1,
                                 exclude_word_ids or set(), require_quiz_definition=True)
    return result.candidates[0] if result.candidates else None


def select_matching_set(conn, schema: str, seed_word_id: int, pos: str, cfg: DistractorConfig,
                         set_size: int, exclude_word_ids: set[int] | None = None) -> DistractorResult:
    """set_size - 1 additional real words (with quiz_definition) to pair with
    seed_word_id into one matching block. The strategies apply to *which words
    belong in the set* here, not to synthetic option generation -- the wrong
    pairings in the rendered matching UI are these words' own real definitions."""
    return _select_candidates(conn, schema, seed_word_id, pos, cfg, set_size - 1,
                               exclude_word_ids or set(), require_quiz_definition=True)
