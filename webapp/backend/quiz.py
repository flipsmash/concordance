"""Quiz-taking API (§ quizzing) -- multiple choice, true/false, and matching,
blendable within one session, with optional spaced-repetition question
selection.

Spaced repetition (concordance/spaced_repetition.py) is a pure selection
bias, not a schema change to questions/answers: word_review_schedule is
updated on every answer regardless of whether the session that produced it
had it enabled, so turning it on later benefits from all prior history
rather than starting cold. Selection prefers -- never hard-filters to --
words that are next_eligible_at <= now() (or never seen), falling back to
not-yet-eligible words whenever the eligible pool can't fill the request.

Imports `main` as a module (not `from webapp.backend.main import SCHEMA, ...`)
and always accesses `_main.SCHEMA`/`_main.get_conn()` via dotted attribute
lookup rather than binding a local copy at import time -- this is what lets
tests keep using the same `main.SCHEMA = disposable_schema` monkeypatch
tests/test_auth.py already established (a bare `from ... import SCHEMA` would
freeze the value at import time and silently ignore that monkeypatch).

Registered into `app` at the *bottom* of main.py, after get_conn/SCHEMA/
require_user/require_admin are all defined -- this module's own top-level
`from webapp.backend import main as _main` resolves against that
already-populated (if still-executing) module object. Moving the
`app.include_router(...)` call earlier in main.py would break this.

--- Per-question-type payload/scoring shapes ---

Every quiz_question.payload is a jsonb blob shaped by its question_type. The
answer key lives in the payload alongside the client-safe fields; _client_view
strips exactly the key fields below before anything is sent to the browser.

  mc:          {prompt, options:[{word_id,label}], correct_word_id,
                nota_is_correct, target_lemma, quiz_definition, degraded}
               -- answer key: correct_word_id, nota_is_correct
  true_false:  {statement_word, statement_definition, is_true, target_lemma,
                quiz_definition, degraded}
               -- answer key: is_true
  matching:    {word_slots:[{word_id,lemma}], definition_slots:[{slot,
                quiz_definition}], correct_mapping:{word_id: slot},
                target_lemma, quiz_definition, degraded}
               -- answer key: correct_mapping
               -- word_slots and definition_slots are shuffled INDEPENDENTLY
               with unrelated ordering/labeling schemes (word order vs. A/B/C
               slot letters) specifically so comparing the two lists never
               reveals the pairing -- only correct_mapping (server-only)
               does. definition_slots deliberately carries no word_id.

quiz_answer gets one row per mc/true_false question, but one row PER PAIR for
a matching question (word_id = that pair's word, response = the submitted
definition_slot) -- matching scores with per-pair credit (a set with 3/4
correct pairs contributes 0.75 toward that question's slot in the test
percentage), computed via a LATERAL per-question aggregate in finish_quiz/
get_quiz_state rather than a flat row count, since a naive COUNT(*) over the
quiz_question/quiz_answer join would fan out on a multi-row matching question
and silently miscount every other question in the same session.
"""

from __future__ import annotations

import random
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from psycopg.types.json import Json
from pydantic import BaseModel, Field

from concordance import distractors as dx
from concordance import spaced_repetition as sr
from concordance import usas_domains
from webapp.backend import main as _main

router = APIRouter()

_SLOT_LETTERS = "ABCDEFGH"


# --- config / request models -------------------------------------------------

class QuizStartRequest(BaseModel):
    length: int = Field(10, ge=1, le=100)
    types: list[Literal["mc", "true_false", "matching"]] = Field(default_factory=lambda: ["mc"])
    mc_choice_count: int = Field(4, ge=2, le=8)
    matching_set_size: int = Field(4, ge=2, le=8)
    nota_enabled: bool = False
    nota_rate: float = Field(0.15, ge=0.0, le=1.0)
    difficulty_min: float | None = Field(None, ge=0, le=100)
    difficulty_max: float | None = Field(None, ge=0, le=100)
    pos: list[str] | None = None
    domains: list[str] | None = None  # usas_domains.DOMAIN_BUCKETS keys (the same 6
                                       # buckets /api/graph/legend uses), not raw USAS codes
    direction: Literal["definition_to_word", "word_to_definition"] = "definition_to_word"
    smart_vs_random_ratio: float = Field(0.7, ge=0.0, le=1.0)
    strategy_weights: dict[str, float] = Field(
        default_factory=lambda: {"orthographic": 1 / 3, "semantic": 1 / 3, "domain": 1 / 3, "antonym": 0.0}
    )
    spaced_repetition_enabled: bool = False
    spaced_repetition_frequency: sr.Frequency = "normal"


class QuizSessionStart(BaseModel):
    session_id: int
    feedback_timing: str
    total_questions: int


class QuizOption(BaseModel):
    word_id: int | None  # None represents the "None of the above" option
    label: str


class QuizWordSlot(BaseModel):
    word_id: int
    lemma: str


class QuizDefinitionSlot(BaseModel):
    slot: str
    quiz_definition: str


class QuizQuestionClient(BaseModel):
    question_id: int
    seq: int
    question_type: str
    prompt: str | None = None                                   # mc / true_false
    options: list[QuizOption] | None = None                     # mc only
    statement_word: str | None = None                           # true_false only
    statement_definition: str | None = None                     # true_false only
    word_slots: list[QuizWordSlot] | None = None                # matching only
    definition_slots: list[QuizDefinitionSlot] | None = None    # matching only


class QuizSessionState(BaseModel):
    session_id: int
    total_questions: int
    answered: int
    completed: bool
    feedback_timing: str
    question: QuizQuestionClient | None = None


class QuizAnswerSubmit(BaseModel):
    question_id: int
    selected_word_id: int | None = None                    # mc: None means "None of the above"
    answer: bool | None = None                              # true_false
    pairs: list[dict] | None = None                         # matching: [{"word_id": int, "definition_slot": str}]


class QuizAnswerResult(BaseModel):
    accepted: bool
    is_correct: bool | None = None                         # mc / true_false
    correct_word_id: int | None = None                     # mc
    correct_label: str | None = None                       # mc
    correct_answer: bool | None = None                     # true_false
    pair_results: list[dict] | None = None                 # matching: [{"word_id", "is_correct", "correct_slot"}]
    quiz_definition: str | None = None


class QuizFinishResult(BaseModel):
    session_id: int
    score_pct: float
    total_questions: int
    correct_count: float  # fractional for sessions containing matching questions


class QuizReviewItem(BaseModel):
    seq: int
    question_type: str
    prompt: str
    your_label: str | None
    correct_label: str
    is_correct: bool
    credit: float  # 1.0/0.0 for mc/true_false, fraction of pairs correct for matching
    target_lemma: str
    quiz_definition: str


class QuizReview(BaseModel):
    session_id: int
    score_pct: float | None
    items: list[QuizReviewItem]


class DomainOption(BaseModel):
    bucket: str
    name: str


class QuizMeta(BaseModel):
    pos_values: list[str]
    domains: list[DomainOption]


class AdminSettingsResponse(BaseModel):
    settings: dict[str, dict]


class AdminSettingUpdate(BaseModel):
    key: str
    value: dict


# --- helpers ------------------------------------------------------------------

def _feedback_timing(conn) -> str:
    with conn.cursor() as cur:
        cur.execute(f"SELECT value FROM {_main.SCHEMA}.app_settings WHERE key = 'quiz_feedback_timing'")
        row = cur.fetchone()
    mode = (row[0] or {}).get("mode") if row else None
    return mode if mode in ("immediate", "end_of_test") else "immediate"


def _select_target_words(conn, body: QuizStartRequest, count: int, exclude_ids: set[int],
                          user_id: int) -> list[dict]:
    filters = ["w.active", "wd.quizzable = true", "w.quiz_definition IS NOT NULL"]
    params: list = []
    if exclude_ids:
        filters.append("NOT (w.id = ANY(%s))")
        params.append(list(exclude_ids))
    if body.pos:
        filters.append("w.part_of_speech = ANY(%s)")
        params.append(body.pos)
    if body.difficulty_min is not None:
        filters.append("wd.difficulty >= %s")
        params.append(body.difficulty_min)
    if body.difficulty_max is not None:
        filters.append("wd.difficulty <= %s")
        params.append(body.difficulty_max)
    if body.domains:
        codes = [code for bucket in body.domains
                 for code in usas_domains.DOMAIN_BUCKETS.get(bucket, {}).get("codes", [])]
        if codes:
            filters.append(
                f"""EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_category wc
                            JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id
                            WHERE wc.word_id = w.id AND left(c.code, 1) = ANY(%s))"""
            )
            params.append(codes)
    where = " AND ".join(filters)

    # Spaced repetition is a preference, never a hard filter: eligible (or
    # never-seen) words sort first, but the LIMIT still falls through to
    # not-yet-eligible ones if that's not enough to fill the request -- a
    # narrow filter config combined with SR-on should degrade gracefully,
    # not return an empty/short question set.
    order_by = "random()"
    if body.spaced_repetition_enabled:
        order_by = "(wrs.next_eligible_at IS NULL OR wrs.next_eligible_at <= now()) DESC, random()"

    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT w.id, w.lemma, w.quiz_definition, w.part_of_speech
                FROM {_main.SCHEMA}.word w
                JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                LEFT JOIN {_main.SCHEMA}.word_review_schedule wrs
                    ON wrs.word_id = w.id AND wrs.user_id = %s
                WHERE {where}
                ORDER BY {order_by}
                LIMIT %s""",
            (user_id, *params, count),
        )
        rows = cur.fetchall()
    return [{"id": r[0], "lemma": r[1], "quiz_definition": r[2], "pos": r[3]} for r in rows]


def _distractor_cfg(body: QuizStartRequest, zero_orthographic: bool) -> dx.DistractorConfig:
    weights = dict(body.strategy_weights)
    if zero_orthographic:
        weights["orthographic"] = 0.0
    return dx.DistractorConfig(
        difficulty_min=body.difficulty_min, difficulty_max=body.difficulty_max,
        smart_vs_random_ratio=body.smart_vs_random_ratio, strategy_weights=weights,
    )


def _build_mc_payload(conn, target: dict, body: QuizStartRequest,
                       exclude_ids: set[int]) -> tuple[dict, list[int], list[int]] | None:
    """One MC question's full payload (answer key included) for `target`, or
    None if the corpus couldn't supply enough distractors for it at all.
    Returns (payload, target_word_ids, consumed_ids) -- target_word_ids is
    what's stored on quiz_question (just the tested word, singular, since
    only it gets a quiz_answer row); consumed_ids is target+distractors, used
    only to keep the rest of the session from reusing any of these words."""
    require_qd = body.direction == "word_to_definition"
    # A lookalike *word* is invisible when only definitions are shown as
    # options -- redistribute its weight rather than waste a slot on it.
    cfg = _distractor_cfg(body, zero_orthographic=require_qd)

    nota_is_correct = body.nota_enabled and random.random() < body.nota_rate
    # Total visible slots = mc_choice_count. NOTA (if enabled) takes one.
    word_slots = body.mc_choice_count - (1 if body.nota_enabled else 0)
    distractor_count = word_slots - (0 if nota_is_correct else 1)
    if distractor_count < 1:
        return None

    result = dx.select_mc_distractors(
        conn, _main.SCHEMA, target["id"], target["pos"], cfg, distractor_count,
        exclude_word_ids=exclude_ids, require_quiz_definition=require_qd,
    )
    if len(result.candidates) < distractor_count:
        if not result.candidates:
            return None
        distractor_count = len(result.candidates)

    def _label(lemma: str, quiz_definition: str | None) -> str:
        return quiz_definition if require_qd else lemma

    options = [
        {"word_id": c["id"], "label": _label(c["lemma"], c.get("quiz_definition"))}
        for c in result.candidates[:distractor_count]
    ]
    consumed_ids = [target["id"]] + [c["id"] for c in result.candidates[:distractor_count]]
    correct_word_id: int | None
    if nota_is_correct:
        correct_word_id = None
    else:
        options.append({"word_id": target["id"], "label": _label(target["lemma"], target["quiz_definition"])})
        correct_word_id = target["id"]
    if body.nota_enabled:
        options.append({"word_id": None, "label": "None of the above"})
    random.shuffle(options)

    prompt = target["quiz_definition"] if body.direction == "definition_to_word" else target["lemma"]
    payload = {
        "prompt": prompt,
        "options": options,
        "correct_word_id": correct_word_id,
        "nota_is_correct": nota_is_correct,
        "degraded": result.degraded,
        "target_lemma": target["lemma"],
        "quiz_definition": target["quiz_definition"],
    }
    return payload, [target["id"]], consumed_ids


def _build_tf_payload(conn, target: dict, body: QuizStartRequest,
                       exclude_ids: set[int]) -> tuple[dict, list[int], list[int]] | None:
    """True/false: 'LEMMA means DEFINITION' -- true half the time (the word's
    own definition), false the rest (a strategy-selected foil's definition).
    Foil selection always needs quiz_definition (the foil's own definition is
    what's shown), so orthographic weight is irrelevant here either way --
    the word itself is shown regardless of which statement is judged.
    Returns (payload, target_word_ids, consumed_ids) -- see _build_mc_payload;
    target_word_ids is just the word being judged, never the foil (the foil
    never gets its own quiz_answer row -- there's nothing to answer about it,
    it's only the source of the false statement's text)."""
    cfg = _distractor_cfg(body, zero_orthographic=False)
    is_true = random.random() < 0.5
    consumed_ids = [target["id"]]

    if is_true:
        statement_definition = target["quiz_definition"]
    else:
        foil = dx.select_tf_foil(conn, _main.SCHEMA, target["id"], target["pos"], cfg,
                                  exclude_word_ids=exclude_ids)
        if foil is None:
            return None
        statement_definition = foil["quiz_definition"]
        consumed_ids.append(foil["id"])

    payload = {
        "statement_word": target["lemma"],
        "statement_definition": statement_definition,
        "is_true": is_true,
        "target_lemma": target["lemma"],
        "quiz_definition": target["quiz_definition"],
        "degraded": False,
    }
    return payload, [target["id"]], consumed_ids


def _build_matching_payload(conn, seed: dict, body: QuizStartRequest,
                             exclude_ids: set[int]) -> tuple[dict, list[int], list[int]] | None:
    """A matching set is direction-agnostic (always word<->definition pairs) --
    the 'distractor' strategies here pick which OTHER real words belong in the
    set, not synthetic options; the wrong pairings a quiz-taker can pick are
    other members' own real quiz_definitions. Orthographic lookalikes matter
    here (a word shown, not just a definition), so unlike word_to_definition
    MC, that strategy stays active. Returns (payload, target_word_ids,
    consumed_ids) -- identical for matching, since every member gets its own
    quiz_answer row (one per pair)."""
    cfg = _distractor_cfg(body, zero_orthographic=False)
    result = dx.select_matching_set(conn, _main.SCHEMA, seed["id"], seed["pos"], cfg,
                                     body.matching_set_size, exclude_word_ids=exclude_ids)
    members = [{"id": seed["id"], "lemma": seed["lemma"], "quiz_definition": seed["quiz_definition"]}]
    members += [{"id": c["id"], "lemma": c["lemma"], "quiz_definition": c["quiz_definition"]}
                for c in result.candidates]
    if len(members) < 2:
        return None

    n = len(members)
    word_order = list(range(n))
    random.shuffle(word_order)
    def_order = list(range(n))
    random.shuffle(def_order)

    word_slots = [{"word_id": members[i]["id"], "lemma": members[i]["lemma"]} for i in word_order]
    def_slots = [{"slot": _SLOT_LETTERS[pos], "member_idx": i, "quiz_definition": members[i]["quiz_definition"]}
                 for pos, i in enumerate(def_order)]
    slot_for_member = {d["member_idx"]: d["slot"] for d in def_slots}
    correct_mapping = {str(m["id"]): slot_for_member[i] for i, m in enumerate(members)}

    payload = {
        "word_slots": word_slots,
        "definition_slots": [{"slot": d["slot"], "quiz_definition": d["quiz_definition"]} for d in def_slots],
        "correct_mapping": correct_mapping,
        "target_lemma": seed["lemma"],
        "quiz_definition": seed["quiz_definition"],
        "degraded": result.degraded,
    }
    member_ids = [m["id"] for m in members]
    return payload, member_ids, member_ids


_BUILDERS = {"mc": _build_mc_payload, "true_false": _build_tf_payload, "matching": _build_matching_payload}


def _client_question(question_id: int, seq: int, question_type: str, payload: dict) -> QuizQuestionClient:
    if question_type == "mc":
        options = [QuizOption(word_id=o["word_id"], label=o["label"]) for o in payload["options"]]
        return QuizQuestionClient(question_id=question_id, seq=seq, question_type=question_type,
                                   prompt=payload["prompt"], options=options)
    if question_type == "true_false":
        return QuizQuestionClient(question_id=question_id, seq=seq, question_type=question_type,
                                   statement_word=payload["statement_word"],
                                   statement_definition=payload["statement_definition"])
    if question_type == "matching":
        word_slots = [QuizWordSlot(**w) for w in payload["word_slots"]]
        def_slots = [QuizDefinitionSlot(slot=d["slot"], quiz_definition=d["quiz_definition"])
                     for d in payload["definition_slots"]]
        return QuizQuestionClient(question_id=question_id, seq=seq, question_type=question_type,
                                   word_slots=word_slots, definition_slots=def_slots)
    raise ValueError(f"unknown question_type {question_type!r}")


def _get_owned_session(conn, session_id: int, user_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, feedback_timing, finished_at, score_pct, config
                FROM {_main.SCHEMA}.quiz_session WHERE id = %s AND user_id = %s""",
            (session_id, user_id),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="quiz session not found")
    return {"id": row[0], "feedback_timing": row[1], "finished_at": row[2], "score_pct": row[3], "config": row[4]}


def _update_review_schedule(cur, user_id: int, word_id: int, is_correct: bool, frequency: str) -> None:
    """Upserts word_review_schedule -- called for every answered word,
    regardless of whether spaced repetition was enabled for the session that
    produced this answer (see module docstring)."""
    cur.execute(
        f"SELECT streak FROM {_main.SCHEMA}.word_review_schedule WHERE user_id = %s AND word_id = %s",
        (user_id, word_id),
    )
    row = cur.fetchone()
    prior_streak = row[0] if row else 0
    update = sr.next_review(prior_streak, is_correct, frequency)
    cur.execute(
        f"""INSERT INTO {_main.SCHEMA}.word_review_schedule
                (user_id, word_id, streak, last_seen_at, next_eligible_at, correct_count, incorrect_count)
            VALUES (%s, %s, %s, now(), %s, %s, %s)
            ON CONFLICT (user_id, word_id) DO UPDATE SET
                streak = EXCLUDED.streak,
                last_seen_at = now(),
                next_eligible_at = EXCLUDED.next_eligible_at,
                correct_count = {_main.SCHEMA}.word_review_schedule.correct_count + EXCLUDED.correct_count,
                incorrect_count = {_main.SCHEMA}.word_review_schedule.incorrect_count + EXCLUDED.incorrect_count""",
        (user_id, word_id, update.streak, update.next_eligible_at, int(is_correct), int(not is_correct)),
    )


def _mc_or_tf_correct_label(qtype: str, payload: dict) -> str:
    if qtype == "mc":
        correct_word_id = payload["correct_word_id"]
        return "None of the above" if payload["nota_is_correct"] else next(
            (o["label"] for o in payload["options"] if o["word_id"] == correct_word_id), ""
        )
    return "True" if payload["is_true"] else "False"


# --- routes ---------------------------------------------------------------

@router.get("/api/quiz/meta", response_model=QuizMeta)
def quiz_meta(_: dict = Depends(_main.require_user)) -> QuizMeta:
    """POS/domain option lists for the quiz config form. Separate from the
    admin-only /api/pos-values (this is require_user, not require_admin --
    any logged-in quiz-taker needs it, not just the curation UI)."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT DISTINCT w.part_of_speech FROM {_main.SCHEMA}.word w
                JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE w.active AND wd.quizzable = true AND w.part_of_speech IS NOT NULL
                ORDER BY 1"""
        )
        pos_values = [r[0] for r in cur.fetchall()]
    domains = [DomainOption(bucket=e["bucket"], name=e["name"]) for e in usas_domains.legend_entries()]
    return QuizMeta(pos_values=pos_values, domains=domains)


@router.post("/api/quiz/start", response_model=QuizSessionStart)
def start_quiz(body: QuizStartRequest, user: dict = Depends(_main.require_user)) -> QuizSessionStart:
    if not body.types:
        raise HTTPException(status_code=400, detail="at least one question type is required")

    with _main.get_conn() as conn, conn.cursor() as cur:
        # Over-fetch: not every candidate target ends up with enough distractors
        # (e.g. a rare POS+difficulty combination), and matching questions
        # consume several words each, so pull a generous pool up front and
        # draw from it as questions are built -- never touching the same word
        # twice within one session.
        pool_size = body.length * (body.matching_set_size + 2) + 20
        used_ids: set[int] = set()
        pool = _select_target_words(conn, body, pool_size, used_ids, user["id"])
        pool_idx = 0

        # (question_type, payload, target_word_ids -- what quiz_answer rows get
        # written against, NOT the same as every word a question consumed)
        questions: list[tuple[str, dict, list[int]]] = []
        while len(questions) < body.length and pool_idx < len(pool):
            qtype = random.choice(body.types)
            target = pool[pool_idx]
            pool_idx += 1
            if target["id"] in used_ids:
                continue
            built = _BUILDERS[qtype](conn, target, body, used_ids)
            if built is None:
                continue
            payload, target_word_ids, consumed_ids = built
            questions.append((qtype, payload, target_word_ids))
            used_ids.update(consumed_ids)

        feedback_timing = _feedback_timing(conn)
        cur.execute(
            f"""INSERT INTO {_main.SCHEMA}.quiz_session (user_id, config, feedback_timing)
                VALUES (%s, %s, %s) RETURNING id""",
            (user["id"], Json(body.model_dump()), feedback_timing),
        )
        session_id = cur.fetchone()[0]
        for seq, (qtype, payload, target_word_ids) in enumerate(questions, start=1):
            cur.execute(
                f"""INSERT INTO {_main.SCHEMA}.quiz_question
                        (session_id, seq, question_type, target_word_ids, payload)
                    VALUES (%s, %s, %s, %s, %s)""",
                (session_id, seq, qtype, target_word_ids, Json(payload)),
            )
        conn.commit()

    return QuizSessionStart(session_id=session_id, feedback_timing=feedback_timing, total_questions=len(questions))


@router.get("/api/quiz/{session_id}", response_model=QuizSessionState)
def get_quiz_state(session_id: int, user: dict = Depends(_main.require_user)) -> QuizSessionState:
    with _main.get_conn() as conn:
        session = _get_owned_session(conn, session_id, user["id"])
        with conn.cursor() as cur:
            # A question is "answered" once it has one quiz_answer row per
            # word in target_word_ids -- 1 for mc/true_false, matching_set_size
            # for matching (submitted in a single call, never partially).
            cur.execute(
                f"""SELECT q.id, q.seq, q.question_type, q.payload,
                           cardinality(q.target_word_ids) AS expected, count(a.id) AS got
                    FROM {_main.SCHEMA}.quiz_question q
                    LEFT JOIN {_main.SCHEMA}.quiz_answer a ON a.question_id = q.id
                    WHERE q.session_id = %s
                    GROUP BY q.id, q.seq, q.question_type, q.payload, q.target_word_ids
                    ORDER BY q.seq ASC""",
                (session_id,),
            )
            rows = cur.fetchall()

    total = len(rows)
    answered = sum(1 for *_, expected, got in rows if got >= expected)
    next_row = next(((qid, seq, qtype, payload) for qid, seq, qtype, payload, expected, got in rows
                      if got < expected), None)
    question = _client_question(*next_row) if next_row else None
    return QuizSessionState(
        session_id=session_id, total_questions=total, answered=answered,
        completed=question is None, feedback_timing=session["feedback_timing"], question=question,
    )


@router.post("/api/quiz/{session_id}/answer", response_model=QuizAnswerResult)
def answer_quiz_question(session_id: int, body: QuizAnswerSubmit,
                          user: dict = Depends(_main.require_user)) -> QuizAnswerResult:
    with _main.get_conn() as conn:
        session = _get_owned_session(conn, session_id, user["id"])
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT question_type, payload, target_word_ids FROM {_main.SCHEMA}.quiz_question
                    WHERE id = %s AND session_id = %s""",
                (body.question_id, session_id),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="question not found in this session")
            qtype, payload, target_word_ids = row

            cur.execute(f"SELECT 1 FROM {_main.SCHEMA}.quiz_answer WHERE question_id = %s LIMIT 1",
                        (body.question_id,))
            if cur.fetchone() is not None:
                raise HTTPException(status_code=400, detail="question already answered")

            frequency = (session["config"] or {}).get("spaced_repetition_frequency", "normal")

            if qtype == "mc":
                if payload["nota_is_correct"]:
                    is_correct = body.selected_word_id is None
                else:
                    is_correct = body.selected_word_id == payload["correct_word_id"]
                cur.execute(
                    f"""INSERT INTO {_main.SCHEMA}.quiz_answer (question_id, word_id, response, is_correct)
                        VALUES (%s, %s, %s, %s)""",
                    (body.question_id, target_word_ids[0],
                     Json({"selected_word_id": body.selected_word_id}), is_correct),
                )
                _update_review_schedule(cur, user["id"], target_word_ids[0], is_correct, frequency)
            elif qtype == "true_false":
                if body.answer is None:
                    raise HTTPException(status_code=400, detail="answer (true/false) is required")
                is_correct = body.answer == payload["is_true"]
                cur.execute(
                    f"""INSERT INTO {_main.SCHEMA}.quiz_answer (question_id, word_id, response, is_correct)
                        VALUES (%s, %s, %s, %s)""",
                    (body.question_id, target_word_ids[0], Json({"answer": body.answer}), is_correct),
                )
                _update_review_schedule(cur, user["id"], target_word_ids[0], is_correct, frequency)
            elif qtype == "matching":
                if not body.pairs:
                    raise HTTPException(status_code=400, detail="pairs is required for a matching question")
                mapping = payload["correct_mapping"]
                pair_results = []
                for pair in body.pairs:
                    wid = pair.get("word_id")
                    slot = pair.get("definition_slot")
                    correct_slot = mapping.get(str(wid))
                    pair_correct = correct_slot is not None and slot == correct_slot
                    cur.execute(
                        f"""INSERT INTO {_main.SCHEMA}.quiz_answer (question_id, word_id, response, is_correct)
                            VALUES (%s, %s, %s, %s)""",
                        (body.question_id, wid, Json({"definition_slot": slot}), pair_correct),
                    )
                    _update_review_schedule(cur, user["id"], wid, pair_correct, frequency)
                    pair_results.append({"word_id": wid, "is_correct": pair_correct, "correct_slot": correct_slot})
                is_correct = all(p["is_correct"] for p in pair_results)
            else:
                raise HTTPException(status_code=500, detail=f"unknown question_type {qtype!r}")

            conn.commit()

    if session["feedback_timing"] != "immediate":
        return QuizAnswerResult(accepted=True)

    if qtype == "matching":
        return QuizAnswerResult(accepted=True, pair_results=pair_results, quiz_definition=payload["quiz_definition"])

    result = QuizAnswerResult(accepted=True, is_correct=is_correct, quiz_definition=payload["quiz_definition"])
    if qtype == "mc":
        result.correct_word_id = payload["correct_word_id"]
        result.correct_label = _mc_or_tf_correct_label(qtype, payload)
    else:  # true_false
        result.correct_answer = payload["is_true"]
    return result


@router.post("/api/quiz/{session_id}/finish", response_model=QuizFinishResult)
def finish_quiz(session_id: int, user: dict = Depends(_main.require_user)) -> QuizFinishResult:
    with _main.get_conn() as conn:
        session = _get_owned_session(conn, session_id, user["id"])
        with conn.cursor() as cur:
            # Per-question fractional credit (1.0/0.0 for mc/true_false, the
            # fraction of correct pairs for matching), NOT a flat row count --
            # a matching question has one quiz_answer row per pair, so a naive
            # COUNT(*) over the join would fan out and miscount every other
            # question in the session too.
            cur.execute(
                f"""SELECT count(*), COALESCE(SUM(credit), 0)
                    FROM (
                        SELECT q.id, COALESCE(AVG(a.is_correct::int), 0) AS credit
                        FROM {_main.SCHEMA}.quiz_question q
                        LEFT JOIN {_main.SCHEMA}.quiz_answer a ON a.question_id = q.id
                        WHERE q.session_id = %s
                        GROUP BY q.id
                    ) per_question""",
                (session_id,),
            )
            total, credit = cur.fetchone()
            total = total or 0
            credit = float(credit or 0.0)
            if session["finished_at"] is None:
                score_pct = round(100.0 * credit / total, 1) if total else 0.0
                cur.execute(
                    f"""UPDATE {_main.SCHEMA}.quiz_session SET finished_at = now(), score_pct = %s
                        WHERE id = %s""",
                    (score_pct, session_id),
                )
                conn.commit()
            else:
                # Idempotent: a session already finished keeps its original
                # score even if this endpoint is hit again.
                score_pct = session["score_pct"] or 0.0

    return QuizFinishResult(session_id=session_id, score_pct=score_pct, total_questions=total, correct_count=credit)


@router.get("/api/quiz/{session_id}/review", response_model=QuizReview)
def review_quiz(session_id: int, user: dict = Depends(_main.require_user)) -> QuizReview:
    with _main.get_conn() as conn:
        session = _get_owned_session(conn, session_id, user["id"])
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT q.id, q.seq, q.question_type, q.payload
                    FROM {_main.SCHEMA}.quiz_question q
                    WHERE q.session_id = %s
                    ORDER BY q.seq ASC""",
                (session_id,),
            )
            questions = cur.fetchall()
            cur.execute(
                f"""SELECT a.question_id, a.word_id, a.response, a.is_correct
                    FROM {_main.SCHEMA}.quiz_answer a
                    JOIN {_main.SCHEMA}.quiz_question q ON q.id = a.question_id
                    WHERE q.session_id = %s""",
                (session_id,),
            )
            answers_by_question: dict[int, list] = {}
            for qid, word_id, response, is_correct in cur.fetchall():
                answers_by_question.setdefault(qid, []).append((word_id, response, is_correct))

    items = []
    for qid, seq, qtype, payload in questions:
        answers = answers_by_question.get(qid, [])
        if qtype in ("mc", "true_false"):
            is_correct = bool(answers[0][2]) if answers else False
            credit = 1.0 if is_correct else 0.0
            correct_label = _mc_or_tf_correct_label(qtype, payload)
            if qtype == "mc":
                prompt = payload["prompt"]
                if not answers:
                    your_label = None
                else:
                    sel = (answers[0][1] or {}).get("selected_word_id")
                    your_label = "None of the above" if sel is None else next(
                        (o["label"] for o in payload["options"] if o["word_id"] == sel), None
                    )
            else:
                prompt = f"{payload['statement_word']}: {payload['statement_definition']}"
                your_label = None if not answers else str((answers[0][1] or {}).get("answer"))
        else:  # matching
            total_pairs = len(payload["word_slots"])
            correct_pairs = sum(1 for _, _, ok in answers if ok)
            credit = (correct_pairs / total_pairs) if total_pairs else 0.0
            is_correct = correct_pairs == total_pairs and len(answers) == total_pairs
            prompt = f"Match {total_pairs} words to their definitions"
            correct_label = "; ".join(f"{w['lemma']} -> {payload['correct_mapping'][str(w['word_id'])]}"
                                       for w in payload["word_slots"])
            your_label = (f"{correct_pairs}/{total_pairs} pairs correct" if answers else None)

        items.append(QuizReviewItem(
            seq=seq, question_type=qtype, prompt=prompt, your_label=your_label,
            correct_label=correct_label, is_correct=is_correct, credit=credit,
            target_lemma=payload["target_lemma"], quiz_definition=payload["quiz_definition"],
        ))

    return QuizReview(session_id=session_id, score_pct=session["score_pct"], items=items)


# --- admin settings ---------------------------------------------------------

@router.get("/api/admin/settings", response_model=AdminSettingsResponse)
def get_admin_settings(_: dict = Depends(_main.require_admin)) -> AdminSettingsResponse:
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT key, value FROM {_main.SCHEMA}.app_settings")
        rows = cur.fetchall()
    return AdminSettingsResponse(settings={k: v for k, v in rows})


@router.put("/api/admin/settings", response_model=AdminSettingsResponse)
def put_admin_setting(body: AdminSettingUpdate, _: dict = Depends(_main.require_admin)) -> AdminSettingsResponse:
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {_main.SCHEMA}.app_settings (key, value, updated_at) VALUES (%s, %s, now())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()""",
            (body.key, Json(body.value)),
        )
        conn.commit()
        cur.execute(f"SELECT key, value FROM {_main.SCHEMA}.app_settings")
        rows = cur.fetchall()
    return AdminSettingsResponse(settings={k: v for k, v in rows})
