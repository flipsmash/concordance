"""Pure tests for progress.py's daily-practice-streak logic -- no DB needed."""

from datetime import date, timedelta

from webapp.backend.progress import _compute_daily_streak

TODAY = date(2026, 7, 25)


def _days_ago(n: int) -> date:
    return TODAY - timedelta(days=n)


def test_empty_history_has_no_streak():
    assert _compute_daily_streak([], TODAY) == 0


def test_run_ending_today():
    days = [_days_ago(0), _days_ago(1), _days_ago(2)]
    assert _compute_daily_streak(days, TODAY) == 3


def test_run_ending_yesterday_still_counts():
    # practiced yesterday and the day before, nothing yet today -- streak stays intact
    days = [_days_ago(1), _days_ago(2), _days_ago(3)]
    assert _compute_daily_streak(days, TODAY) == 3


def test_gap_breaks_the_streak():
    # a 2-day gap between day -1 and day -4 means only the most recent run counts
    days = [_days_ago(0), _days_ago(1), _days_ago(4), _days_ago(5)]
    assert _compute_daily_streak(days, TODAY) == 2


def test_no_recent_activity_is_zero():
    # last practiced 3 days ago -- streak is broken (not "today or yesterday")
    days = [_days_ago(3), _days_ago(4)]
    assert _compute_daily_streak(days, TODAY) == 0


def test_duplicate_and_unsorted_input_handled():
    days = [_days_ago(2), _days_ago(0), _days_ago(0), _days_ago(1), _days_ago(2)]
    assert _compute_daily_streak(days, TODAY) == 3


def test_single_day_today():
    assert _compute_daily_streak([_days_ago(0)], TODAY) == 1
