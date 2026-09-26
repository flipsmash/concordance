"""Run the database tests by default.

DB tests are gated on CONCORDANCE_TEST_DB_URL. When it isn't set, fall back
to the project's own DATABASE_URL (env or .env): every DB test works in its
own throwaway cc_test_* / oed_test_* schema and drops it afterwards, so the
real schemas are never written. Skipping silently let tests go stale unseen
(the browse listing tests broke that way). Set CONCORDANCE_SKIP_DB_TESTS=1
to skip them anyway.
"""

from __future__ import annotations

import os

_status = "CONCORDANCE_TEST_DB_URL set explicitly"


def _connectable(url: str) -> bool:
    try:
        import psycopg
        psycopg.connect(url, connect_timeout=3).close()
        return True
    except Exception:  # noqa: BLE001
        return False


if os.environ.get("CONCORDANCE_SKIP_DB_TESTS"):
    os.environ.pop("CONCORDANCE_TEST_DB_URL", None)
    _status = "skipped (CONCORDANCE_SKIP_DB_TESTS)"
elif not os.environ.get("CONCORDANCE_TEST_DB_URL"):
    from concordance import db

    url = db.database_url()
    if url and _connectable(url):
        os.environ["CONCORDANCE_TEST_DB_URL"] = url
        _status = "using DATABASE_URL (throwaway test schemas)"
    else:
        _status = "SKIPPED -- no reachable DATABASE_URL"


def pytest_report_header(config):
    return f"database tests: {_status}"
