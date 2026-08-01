"""Home/landing page summary (§ home page) -- a handful of corpus-scope
numbers for the Home page's "colophon sentence" (total words/books/authors/
categories) plus a few unscoped, app-wide "impressive" stats. Nothing here
takes filter params; unlike browse.py this is a single fixed snapshot of the
whole corpus, not a facet-combinable listing.

Imports `main` as a module and always accesses `_main.SCHEMA`/
`_main.get_conn()`/`_main.require_viewer` via dotted attribute lookup, not a
bare `from ... import`, matching browse.py/quiz.py/progress.py -- same
reason: registered into `app` at the bottom of main.py, after those names
are defined, so a bare import would freeze an unset value.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from concordance import usas
from concordance.db import PLACEHOLDER_AUTHORS
from webapp.backend import main as _main

router = APIRouter()


class HomeSummary(BaseModel):
    total_words: int
    total_books: int
    total_authors: int
    total_categories: int
    most_acclaimed_book: str | None
    most_acclaimed_author: str | None
    hardest_word: str | None
    quiz_questions_answered: int


@router.get("/api/home/summary", response_model=HomeSummary)
def home_summary(_: dict = Depends(_main.require_viewer)) -> HomeSummary:
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {_main.SCHEMA}.word w WHERE w.active")
        total_words = cur.fetchone()[0]

        # Matches Books.jsx/Authors.jsx's own totals: a plain count(*) FROM
        # book would include books with zero active words, which browse.py's
        # own listing endpoints never count (see browse.py's dedup-rule
        # docstring -- JOIN+GROUP BY is correct here since book/author IS the
        # thing being counted, not filtered-through).
        cur.execute(
            f"""SELECT count(*) FROM (
                    SELECT b.id
                    FROM {_main.SCHEMA}.book b
                    JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                    JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                    WHERE w.active
                    GROUP BY b.id
                ) sub"""
        )
        total_books = cur.fetchone()[0]

        cur.execute(
            f"""SELECT count(*) FROM (
                    SELECT b.author
                    FROM {_main.SCHEMA}.book b
                    JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                    JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                    WHERE w.active AND b.author IS NOT NULL AND b.author != ALL(%s)
                    GROUP BY b.author
                ) sub""",
            (list(PLACEHOLDER_AUTHORS),),
        )
        total_authors = cur.fetchone()[0]

        cur.execute(
            f"""SELECT b.title, b.author FROM {_main.SCHEMA}.book b
                JOIN {_main.SCHEMA}.book_fame bf ON bf.book_id = b.id
                ORDER BY bf.fame_score DESC NULLS LAST LIMIT 1"""
        )
        row = cur.fetchone()
        most_acclaimed_book, most_acclaimed_author = row if row else (None, None)

        cur.execute(
            f"""SELECT w.lemma FROM {_main.SCHEMA}.word w
                JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE w.active ORDER BY wd.difficulty DESC LIMIT 1"""
        )
        row = cur.fetchone()
        hardest_word = row[0] if row else None

        # Deliberately app-wide and unscoped -- not filtered to one user or
        # to finished_at IS NOT NULL the way progress.py's per-user
        # _kpi_tiles() is, so this number isn't directly comparable to a
        # single user's own Progress tile. Grain is per-question (one row
        # per matching question), not raw quiz_answer rows, which fan out
        # 4:1 per matching question -- see progress.py's own module
        # docstring for why.
        cur.execute(
            f"""SELECT count(DISTINCT q.id) FROM {_main.SCHEMA}.quiz_question q
                JOIN {_main.SCHEMA}.quiz_answer a ON a.question_id = q.id"""
        )
        quiz_questions_answered = cur.fetchone()[0]

    # Not a query -- usas.categories()'s level field marks the 21 top-level
    # USAS discourse fields, the same fixed set browse.py's own _TOP_CODES
    # derives at import time (its own comment: "21 static (code, name) pairs
    # derived from a module-level constant, not data that can change at
    # runtime").
    total_categories = len([c for c in usas.categories() if c["level"] == 0])

    return HomeSummary(
        total_words=total_words,
        total_books=total_books,
        total_authors=total_authors,
        total_categories=total_categories,
        most_acclaimed_book=most_acclaimed_book,
        most_acclaimed_author=most_acclaimed_author,
        hardest_word=hardest_word,
        quiz_questions_answered=quiz_questions_answered,
    )
