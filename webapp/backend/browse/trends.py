"""/api/browse/ngram-trend: popularity over time from the local Google Books Ngram data."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel
from webapp.backend import main as _main

from .common import router


# --- /api/browse/ngram-trend ------------------------------------------------
#
# A term's popularity over time from the LOCAL Google Books Ngram table
# (ngram.unigram, loaded by `concordance ngram-bulk-load` -- every lowercase
# term Google counted, not just this app's vocabulary). Per-decade counts
# 1800-2019 are what's stored (yearly series were not kept), normalised by
# ngram.decade_total into occurrences per million words so decades of very
# different corpus size compare fairly. Lowercase-exact, like the rest of
# the app's Ngram use.

_TREND_MAX_TERMS = 4


class NgramTrendSeries(BaseModel):
    term: str
    found: bool
    per_million: list[float]       # one per decade, aligned with NgramTrend.decades
    total_matches: int


class NgramTrend(BaseModel):
    decades: list[int]
    series: list[NgramTrendSeries]


@router.get("/api/browse/ngram-trend", response_model=NgramTrend)
def ngram_trend(term: list[str] = Query(...), _: dict = Depends(_main.require_viewer)) -> NgramTrend:
    terms: list[str] = []
    for t in term:
        t = t.strip().lower()
        if t and t not in terms:
            terms.append(t)
    if not terms:
        raise HTTPException(400, "give at least one term")
    if len(terms) > _TREND_MAX_TERMS:
        raise HTTPException(400, f"at most {_TREND_MAX_TERMS} terms")
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('ngram.unigram'), to_regclass('ngram.decade_total')")
        if None in cur.fetchone():
            raise HTTPException(503, "local Ngram data not loaded (run `concordance ngram-bulk-load`)")
        cur.execute("SELECT decade, match_count FROM ngram.decade_total ORDER BY decade")
        totals = cur.fetchall()
        cur.execute("SELECT term, decade_counts FROM ngram.unigram WHERE term = ANY(%s)", (terms,))
        counts = dict(cur.fetchall())
    decades = [d for d, _ in totals]
    series = []
    for t in terms:
        c = counts.get(t)
        per_million = ([round(n / tot * 1e6, 6) if tot else 0.0 for n, (_, tot) in zip(c, totals)]
                       if c else [0.0] * len(decades))
        series.append(NgramTrendSeries(term=t, found=c is not None, per_million=per_million,
                                       total_matches=sum(c) if c else 0))
    return NgramTrend(decades=decades, series=series)
