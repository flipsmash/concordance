"""Personal quiz-history / progress API (§ progress).

Nothing here required a schema change: quiz_session/quiz_question/quiz_answer/
word_review_schedule/word_personal_difficulty already carry everything this
module surfaces -- this is a new read/aggregation layer, not new
instrumentation. Every query reaches quiz_answer (which has no user_id column
of its own) via quiz_session.user_id -> quiz_question.session_id ->
quiz_answer.question_id, a path already fully indexed (quiz_session_user_idx,
quiz_question's UNIQUE(session_id, seq) doubling as a session_id index,
quiz_answer_question_idx/quiz_answer_word_idx) -- no new index needed.

Two grain rules, both already solved once by quiz.py's finish_quiz -- copy
that shape, don't rediscover the bug:

  * quiz_answer has one row PER MATCHING PAIR, not one row per question. Any
    aggregate whose unit is "a question" (score trend, accuracy by question
    type) must first collapse to per-question credit (GROUP BY q.id,
    AVG(a.is_correct::int) AS credit) before aggregating further -- a flat
    COUNT/AVG over raw quiz_answer rows overweights matching questions 4:1
    against mc/true_false/analogy.
  * Aggregates whose unit is genuinely "the word" (domain-bucket accuracy,
    book accuracy, per-word history) are correct to query quiz_answer at row
    grain directly -- each such function's docstring says so explicitly.

Other invariants every function here relies on: quiz_session.finished_at/
score_pct are NULL for an abandoned session (always filter finished_at IS NOT
NULL); quiz_answer.question_type/.direction were added via a later ALTER
TABLE with no backfill (NULL on old rows) -- always read question_type from
quiz_question, never quiz_answer. A word can belong to 2+ USAS domain buckets
or 2+ source books; an answer counts toward EVERY one it touches (matching
browse.py's _bucket_counts() non-partitioning convention) -- bars/tables here
deliberately don't sum to the grand total, and UI copy should say so rather
than let it look like a bug. The daily-practice streak buckets by UTC day, a
known MVP simplification (a per-user timezone preference is a natural, not
yet built, stretch item).

Forward-compat note (cross-user comparison is an explicit future want, not
built now): every aggregate helper below takes a plain `user_id` parameter
and returns a `{value, n}`-shaped result (or a list of such). Adding a future
population-wide variant is "call the same helper with a different scoping
predicate and compare," a new function, not a rewrite of these -- keep that
shape when extending this module.

All endpoints require a logged-in user (require_user) -- this is personal
data, never admin/curation surface (require_admin) and never the
account-less browsing path (require_viewer, used by /api/words/{id}).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from concordance import usas_domains
from webapp.backend import main as _main

router = APIRouter()


# --- response models ----------------------------------------------------------

class ProgressStatTile(BaseModel):
    label: str
    value: float | int | None
    unit: str | None = None
    n: int


class ScorePoint(BaseModel):
    session_id: int
    finished_at: datetime
    score_pct: float
    total_questions: int


class AccuracyBucket(BaseModel):
    key: str
    label: str
    correct: float
    total: int
    accuracy_pct: float | None


class BookAccuracyRow(BaseModel):
    book_id: int
    title: str
    author: str | None
    correct: float
    total: int
    accuracy_pct: float | None


class StrugglingWordRow(BaseModel):
    word_id: int
    lemma: str
    correct_count: int
    incorrect_count: int
    streak: int
    miss_rate: float
    last_seen_at: datetime | None
    next_eligible_at: datetime | None


class ProgressOverview(BaseModel):
    tiles: list[ProgressStatTile]
    trend: list[ScorePoint]
    by_question_type: list[AccuracyBucket]
    by_domain: list[AccuracyBucket]


class WordAnswerLogEntry(BaseModel):
    answered_at: datetime
    is_correct: bool
    question_type: str


class WordProgressHistory(BaseModel):
    word_id: int
    answers: list[WordAnswerLogEntry]
    streak: int
    correct_count: int
    incorrect_count: int
    next_eligible_at: datetime | None
    personal_difficulty: float | None


# --- pure helpers (no DB) ------------------------------------------------------

def _compute_daily_streak(days: list[date], today: date) -> int:
    """Consecutive-day practice streak from a list of distinct UTC dates a
    user answered at least one question on. `days` need not be sorted or
    deduped by the caller. A streak counts through today OR yesterday (so a
    user who practiced yesterday but hasn't yet today still sees their streak
    intact) but breaks on any earlier gap."""
    if not days:
        return 0
    unique_days = sorted(set(days), reverse=True)
    if unique_days[0] not in (today, today - timedelta(days=1)):
        return 0
    streak = 1
    for prev, cur in zip(unique_days, unique_days[1:]):
        if (prev.toordinal() - cur.toordinal()) == 1:
            streak += 1
        else:
            break
    return streak


# --- DB-backed aggregate helpers -----------------------------------------------
#
# Every function below takes a plain psycopg cursor + user_id -- no route/auth
# concerns here, so they're directly callable (and directly testable) without
# going through HTTP. See this module's docstring for the join path, the two
# grain rules, and the forward-compat convention they all follow.

_MASTERY_STREAK = 5   # spaced_repetition.py's 2**streak-day interval hits its 30-day
                      # cap at streak 5 -- the threshold for the "5+ correct streak"
                      # KPI tile. Never call this "mastered": word_review_schedule's
                      # own schema comment explicitly disclaims being a mastery system.

_QUESTION_TYPE_LABELS = [
    ("mc", "Multiple choice"),
    ("true_false", "True / False"),
    ("matching", "Matching"),
    ("analogy", "Analogy"),
]


def _kpi_tiles(cur, schema: str, user_id: int) -> list[ProgressStatTile]:
    cur.execute(f"SELECT count(*) FROM {schema}.quiz_session WHERE user_id = %s AND finished_at IS NOT NULL",
                (user_id,))
    quizzes_taken = cur.fetchone()[0]

    cur.execute(
        f"""SELECT count(*), AVG(credit) FROM (
                SELECT q.id, COALESCE(AVG(a.is_correct::int), 0) AS credit
                FROM {schema}.quiz_session sess
                JOIN {schema}.quiz_question q ON q.session_id = sess.id
                LEFT JOIN {schema}.quiz_answer a ON a.question_id = q.id
                WHERE sess.user_id = %s AND sess.finished_at IS NOT NULL
                GROUP BY q.id
            ) per_question""",
        (user_id,),
    )
    questions_answered, lifetime_accuracy = cur.fetchone()

    cur.execute(f"SELECT count(*) FROM {schema}.word_review_schedule WHERE user_id = %s AND streak >= %s",
                (user_id, _MASTERY_STREAK))
    streak_words = cur.fetchone()[0]

    cur.execute(
        f"""SELECT DISTINCT date_trunc('day', a.answered_at AT TIME ZONE 'UTC')::date AS day
            FROM {schema}.quiz_session sess
            JOIN {schema}.quiz_question q ON q.session_id = sess.id
            JOIN {schema}.quiz_answer a ON a.question_id = q.id
            WHERE sess.user_id = %s""",
        (user_id,),
    )
    days = [r[0] for r in cur.fetchall()]
    daily_streak = _compute_daily_streak(days, datetime.now(timezone.utc).date())

    return [
        ProgressStatTile(label="Quizzes taken", value=quizzes_taken, unit=None, n=quizzes_taken),
        ProgressStatTile(label="Lifetime accuracy",
                          value=round(100 * lifetime_accuracy, 1) if questions_answered else None,
                          unit="%", n=questions_answered),
        ProgressStatTile(label="Practice streak", value=daily_streak, unit="days", n=len(days)),
        ProgressStatTile(label="Words on a 5+ streak", value=streak_words, unit=None, n=streak_words),
    ]


def _score_trend(cur, schema: str, user_id: int) -> list[ScorePoint]:
    cur.execute(
        f"""SELECT sess.id, sess.finished_at, sess.score_pct, count(q.id) AS total_questions
            FROM {schema}.quiz_session sess
            JOIN {schema}.quiz_question q ON q.session_id = sess.id
            WHERE sess.user_id = %s AND sess.finished_at IS NOT NULL
            GROUP BY sess.id, sess.finished_at, sess.score_pct
            HAVING count(q.id) > 0
            ORDER BY sess.finished_at ASC""",
        (user_id,),
    )
    return [ScorePoint(session_id=r[0], finished_at=r[1], score_pct=r[2], total_questions=r[3])
            for r in cur.fetchall()]


def _accuracy_by_question_type(cur, schema: str, user_id: int) -> list[AccuracyBucket]:
    """Grain = the question (see module docstring's matching-fan-out rule):
    per-question credit is computed first, then grouped by question_type."""
    cur.execute(
        f"""SELECT question_type, count(*), COALESCE(SUM(credit), 0) FROM (
                SELECT q.id, q.question_type, COALESCE(AVG(a.is_correct::int), 0) AS credit
                FROM {schema}.quiz_session sess
                JOIN {schema}.quiz_question q ON q.session_id = sess.id
                LEFT JOIN {schema}.quiz_answer a ON a.question_id = q.id
                WHERE sess.user_id = %s AND sess.finished_at IS NOT NULL
                GROUP BY q.id, q.question_type
            ) per_question GROUP BY question_type""",
        (user_id,),
    )
    by_type = {qtype: (total, float(correct)) for qtype, total, correct in cur.fetchall()}
    results = []
    for key, label in _QUESTION_TYPE_LABELS:
        total, correct = by_type.get(key, (0, 0.0))
        results.append(AccuracyBucket(key=key, label=label, correct=correct, total=total,
                                       accuracy_pct=round(100 * correct / total, 1) if total else None))
    return results


def _accuracy_by_domain(cur, schema: str, user_id: int) -> list[AccuracyBucket]:
    """Grain = the answer row (unit is the word) -- a word can straddle
    multiple USAS domain buckets, so this is one EXISTS-gated query per
    bucket (copying browse.py's _bucket_counts() convention exactly) rather
    than one GROUP BY: an answer counts toward every bucket its word
    touches, so bucket totals don't sum to the grand total."""
    results = []
    for entry in usas_domains.legend_entries():
        codes = usas_domains.DOMAIN_BUCKETS[entry["bucket"]]["codes"]
        cur.execute(
            f"""SELECT count(*), COALESCE(SUM(a.is_correct::int), 0)
                FROM {schema}.quiz_session sess
                JOIN {schema}.quiz_question q ON q.session_id = sess.id
                JOIN {schema}.quiz_answer a ON a.question_id = q.id
                WHERE sess.user_id = %s AND sess.finished_at IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM {schema}.word_category wc
                      JOIN {schema}.category c ON c.id = wc.category_id
                      WHERE wc.word_id = a.word_id AND left(c.code, 1) = ANY(%s)
                  )""",
            (user_id, codes),
        )
        total, correct = cur.fetchone()
        results.append(AccuracyBucket(key=entry["bucket"], label=entry["name"], correct=float(correct),
                                       total=total, accuracy_pct=round(100 * correct / total, 1) if total else None))
    return results


def _by_book(cur, schema: str, user_id: int) -> list[BookAccuracyRow]:
    """Grain = the answer row -- a word can be sourced from multiple books
    (word_book is a genuine many-to-many), same expected fan-out
    /api/words/{id} already handles for a single word's book list."""
    cur.execute(
        f"""SELECT b.id, b.title, COALESCE(b.author, 'Unknown'),
                   count(*), COALESCE(SUM(a.is_correct::int), 0)
            FROM {schema}.quiz_session sess
            JOIN {schema}.quiz_question q ON q.session_id = sess.id
            JOIN {schema}.quiz_answer a ON a.question_id = q.id
            JOIN {schema}.word_book wb ON wb.word_id = a.word_id
            JOIN {schema}.book b ON b.id = wb.book_id
            WHERE sess.user_id = %s AND sess.finished_at IS NOT NULL
            GROUP BY b.id, b.title, b.author
            ORDER BY count(*) DESC""",
        (user_id,),
    )
    return [BookAccuracyRow(book_id=r[0], title=r[1], author=r[2], correct=float(r[4]), total=r[3],
                             accuracy_pct=round(100 * r[4] / r[3], 1) if r[3] else None)
            for r in cur.fetchall()]


def _struggling_words(cur, schema: str, user_id: int, limit: int) -> list[StrugglingWordRow]:
    """Reads word_review_schedule directly -- it's already the authoritative
    per-word correct/incorrect/streak state, no need to re-derive from
    quiz_answer. A >=2-exposure floor keeps a single-miss word (100% miss
    rate on n=1) from dominating the list -- a deliberate, tunable anti-noise
    filter, not an oversight."""
    cur.execute(
        f"""SELECT wrs.word_id, w.lemma, wrs.correct_count, wrs.incorrect_count, wrs.streak,
                   wrs.last_seen_at, wrs.next_eligible_at
            FROM {schema}.word_review_schedule wrs
            JOIN {schema}.word w ON w.id = wrs.word_id
            WHERE wrs.user_id = %s AND (wrs.correct_count + wrs.incorrect_count) >= 2
            ORDER BY (wrs.incorrect_count::float / NULLIF(wrs.correct_count + wrs.incorrect_count, 0)) DESC,
                     wrs.streak ASC
            LIMIT %s""",
        (user_id, limit),
    )
    return [
        StrugglingWordRow(
            word_id=r[0], lemma=r[1], correct_count=r[2], incorrect_count=r[3], streak=r[4],
            miss_rate=round(r[3] / (r[2] + r[3]), 3), last_seen_at=r[5], next_eligible_at=r[6],
        )
        for r in cur.fetchall()
    ]


def _word_history(cur, schema: str, user_id: int, word_id: int) -> WordProgressHistory:
    cur.execute(
        f"""SELECT a.answered_at, a.is_correct, q.question_type
            FROM {schema}.quiz_session sess
            JOIN {schema}.quiz_question q ON q.session_id = sess.id
            JOIN {schema}.quiz_answer a ON a.question_id = q.id
            WHERE sess.user_id = %s AND a.word_id = %s
            ORDER BY a.answered_at ASC""",
        (user_id, word_id),
    )
    answers = [WordAnswerLogEntry(answered_at=r[0], is_correct=r[1], question_type=r[2]) for r in cur.fetchall()]

    cur.execute(
        f"""SELECT streak, correct_count, incorrect_count, next_eligible_at
            FROM {schema}.word_review_schedule WHERE user_id = %s AND word_id = %s""",
        (user_id, word_id),
    )
    row = cur.fetchone()
    streak, correct_count, incorrect_count, next_eligible_at = row if row else (0, 0, 0, None)

    cur.execute(f"SELECT personal_difficulty FROM {schema}.word_personal_difficulty WHERE user_id = %s AND word_id = %s",
                (user_id, word_id))
    pd_row = cur.fetchone()
    personal_difficulty = pd_row[0] if pd_row else None

    return WordProgressHistory(word_id=word_id, answers=answers, streak=streak, correct_count=correct_count,
                                incorrect_count=incorrect_count, next_eligible_at=next_eligible_at,
                                personal_difficulty=personal_difficulty)


# --- routes ---------------------------------------------------------------

_DEFAULT_STRUGGLING_LIMIT = 25


@router.get("/api/progress/overview", response_model=ProgressOverview)
def get_overview(user: dict = Depends(_main.require_user)) -> ProgressOverview:
    """Bundles the KPI tiles + score trend + question-type/domain breakdowns
    in one round trip -- all four are small, cheap, single-user-scoped
    queries the dashboard's first paint needs together (unlike /books or
    /struggling, which are tables a user may scroll to independently)."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        return ProgressOverview(
            tiles=_kpi_tiles(cur, _main.SCHEMA, user["id"]),
            trend=_score_trend(cur, _main.SCHEMA, user["id"]),
            by_question_type=_accuracy_by_question_type(cur, _main.SCHEMA, user["id"]),
            by_domain=_accuracy_by_domain(cur, _main.SCHEMA, user["id"]),
        )


@router.get("/api/progress/history", response_model=list[ScorePoint])
def get_history(user: dict = Depends(_main.require_user)) -> list[ScorePoint]:
    """Every finished quiz session -- same ScorePoint shape and same
    _score_trend query /overview's spark-line trend already uses, just
    exposed standalone so the calendar history page isn't dragged along
    behind tiles/breakdowns a trend-only fetch doesn't need."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        return _score_trend(cur, _main.SCHEMA, user["id"])


@router.get("/api/progress/books", response_model=list[BookAccuracyRow])
def get_books(user: dict = Depends(_main.require_user)) -> list[BookAccuracyRow]:
    """Plain list, not {items, total} -- a personal user's distinct
    source-book count is inherently small (dozens, not thousands), so the
    frontend fetches this once and sorts client-side rather than going
    through usePagedTable.js's server-side page/sort contract (built for
    much larger, admin-wide tables like Browse.jsx's full word list)."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        return _by_book(cur, _main.SCHEMA, user["id"])


@router.get("/api/progress/struggling", response_model=list[StrugglingWordRow])
def get_struggling(limit: int = _DEFAULT_STRUGGLING_LIMIT,
                    user: dict = Depends(_main.require_user)) -> list[StrugglingWordRow]:
    with _main.get_conn() as conn, conn.cursor() as cur:
        return _struggling_words(cur, _main.SCHEMA, user["id"], limit)


@router.get("/api/progress/words/{word_id}", response_model=WordProgressHistory)
def get_word_history(word_id: int, user: dict = Depends(_main.require_user)) -> WordProgressHistory:
    """Backs the WordDetail.jsx per-word history panel. Deliberately its own
    require_user-gated endpoint rather than folded into GET /api/words/{id}
    (require_viewer, works without an account) -- merging them would either
    leak personal data to viewer-only requests or wrongly tighten a route
    that's intentionally more permissive today. Never 404s on a word with no
    history -- an all-zero/empty WordProgressHistory is a legitimate answer
    (the user just hasn't been quizzed on this word yet), not an error."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        return _word_history(cur, _main.SCHEMA, user["id"], word_id)
