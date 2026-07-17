"""Session/password/invite-token auth. Pure helpers run always; the DB-backed
tests run only when a throwaway Postgres is provided via CONCORDANCE_TEST_DB_URL
(else skipped) -- same convention as test_db.py."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from concordance import db
from webapp.backend import auth


# --- pure helpers (no database) ---------------------------------------------

def test_hash_and_verify_password_roundtrip():
    hashed = auth.hash_password("correct horse battery staple")
    assert auth.verify_password("correct horse battery staple", hashed)
    assert not auth.verify_password("wrong password", hashed)


def test_verify_cf_access_returns_none_when_unconfigured():
    # No CF_ACCESS_TEAM_DOMAIN/CF_ACCESS_AUD in the test environment -> fails
    # closed rather than raising, so the app still works via app-sessions alone.
    from fastapi import Request

    scope = {"type": "http", "headers": [], "method": "GET"}
    request = Request(scope)
    assert auth.verify_cf_access(request) is None


# --- round trip (needs a real, disposable Postgres) --------------------------

_URL = os.environ.get("CONCORDANCE_TEST_DB_URL", "")


def _connectable(url):
    try:
        import psycopg
        psycopg.connect(url, connect_timeout=3).close()
        return True
    except Exception:
        return False


pg = pytest.mark.skipif(not (_URL and _connectable(_URL)),
                        reason="set CONCORDANCE_TEST_DB_URL to a disposable Postgres to run")


@pg
def test_session_create_get_destroy():
    schema = "cc_test_auth"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash, is_admin) "
            f"VALUES ('alice', %s, false) RETURNING id",
            (auth.hash_password("password123"),),
        )
        user_id = cur.fetchone()[0]
    conn.commit()

    token, expires_at = auth.create_session(conn, schema, user_id)
    assert expires_at > datetime.now(timezone.utc)

    user = auth.get_session_user(conn, schema, token)
    assert user == {"id": user_id, "username": "alice", "is_admin": False}

    assert auth.get_session_user(conn, schema, "not-a-real-token") is None

    auth.destroy_session(conn, schema, token)
    assert auth.get_session_user(conn, schema, token) is None

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_session_expiry_is_honored():
    schema = "cc_test_auth"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash) VALUES ('bob', 'x') RETURNING id"
        )
        user_id = cur.fetchone()[0]
        # Insert an already-expired session directly -- create_session always
        # sets a future expiry, so this exercises the "expired" branch of the
        # WHERE s.expires_at > now() check in get_session_user.
        cur.execute(
            f"INSERT INTO {schema}.sessions (token, user_id, expires_at) VALUES (%s,%s,%s)",
            ("expired-token", user_id, datetime.now(timezone.utc) - timedelta(days=1)),
        )
    conn.commit()

    assert auth.get_session_user(conn, schema, "expired-token") is None

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_invite_token_valid_once_then_rejected():
    schema = "cc_test_auth"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.invite_tokens (token, expires_at) VALUES (%s,%s)",
            ("invite-abc", datetime.now(timezone.utc) + timedelta(days=7)),
        )

    def _token_is_valid(tok: str) -> bool:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT id FROM {schema}.invite_tokens
                    WHERE token=%s AND used_at IS NULL AND expires_at > now()""",
                (tok,),
            )
            return cur.fetchone() is not None

    assert _token_is_valid("invite-abc")

    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash) VALUES ('carol', 'x') RETURNING id"
        )
        user_id = cur.fetchone()[0]
        cur.execute(
            f"UPDATE {schema}.invite_tokens SET used_at=now(), used_by_user_id=%s WHERE token=%s",
            (user_id, "invite-abc"),
        )
    conn.commit()

    assert not _token_is_valid("invite-abc")  # already used

    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.invite_tokens (token, expires_at) VALUES (%s,%s)",
            ("invite-expired", datetime.now(timezone.utc) - timedelta(days=1)),
        )
    conn.commit()

    assert not _token_is_valid("invite-expired")  # expired

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


# --- FastAPI dependency enforcement (the actual security boundary) -----------
# require_admin/require_viewer/get_current_user are what stand between an
# anonymous request and the admin curation API (see main.py's require_admin --
# it's the *only* thing enforcing that DELETE /api/words/{id} stays admin-only,
# since Cloudflare Access can't distinguish it from the open GET on the same
# path). Getting is_admin backwards here silently either locks the admin out
# or opens curation to any registered viewer, so this boundary gets exercised
# directly rather than trusted from the session-CRUD tests above.

def test_get_current_user_returns_none_without_cookie():
    from fastapi import Request

    from webapp.backend import main

    request = Request({"type": "http", "method": "GET", "headers": []})
    assert main.get_current_user(request) is None


def test_require_viewer_and_require_admin_reject_anonymous():
    from fastapi import HTTPException, Request

    from webapp.backend import main

    request = Request({"type": "http", "method": "GET", "headers": []})

    with pytest.raises(HTTPException) as exc:
        main.require_viewer(request, None)
    assert exc.value.status_code == 401

    with pytest.raises(HTTPException) as exc:
        main.require_admin(request, None)
    assert exc.value.status_code == 403


def test_require_admin_rejects_non_admin_but_require_viewer_accepts():
    from fastapi import HTTPException, Request

    from webapp.backend import main

    request = Request({"type": "http", "method": "GET", "headers": []})
    viewer = {"id": 1, "username": "alice", "is_admin": False}

    assert main.require_viewer(request, viewer) == viewer

    with pytest.raises(HTTPException) as exc:
        main.require_admin(request, viewer)
    assert exc.value.status_code == 403


def test_require_admin_accepts_admin_session_user():
    from fastapi import Request

    from webapp.backend import main

    request = Request({"type": "http", "method": "GET", "headers": []})
    admin = {"id": 2, "username": "brian", "is_admin": True}

    assert main.require_admin(request, admin) == admin


@pg
def test_get_current_user_reads_cookie_and_loads_real_session():
    from fastapi import HTTPException, Request

    from webapp.backend import main

    schema = "cc_test_auth_deps"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash, is_admin) "
            f"VALUES ('dave', %s, false) RETURNING id",
            (auth.hash_password("password123"),),
        )
        user_id = cur.fetchone()[0]
    conn.commit()
    token, _ = auth.create_session(conn, schema, user_id)
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        request = Request({
            "type": "http", "method": "GET",
            "headers": [(b"cookie", f"{auth.SESSION_COOKIE_NAME}={token}".encode())],
        })
        user = main.get_current_user(request)
        assert user == {"id": user_id, "username": "dave", "is_admin": False}

        # A real, valid, but non-admin session must still be turned away by
        # require_admin -- this is the exact case that protects the curation
        # API's DELETE endpoint from a logged-in-but-ordinary viewer.
        with pytest.raises(HTTPException) as exc:
            main.require_admin(request, user)
        assert exc.value.status_code == 403
        assert main.require_viewer(request, user) == user
    finally:
        main.SCHEMA = old_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()


# --- full HTTP round trip through the ASGI app --------------------------------
# Runs against a disposable schema (apply_schema creates the *entire* app
# schema -- word/category/users/etc -- under any schema name given), so this
# never touches the production `concordance` schema and can't contend with
# deepen/classify's held locks on it. TestClient is built without the `with`
# block so the app's on_startup hook (which calls apply_schema against
# main.SCHEMA) never fires -- schema is applied explicitly below instead.
# base_url uses https:// so the client's cookie jar honors the session
# cookie's Secure flag and actually round-trips it, matching production
# (only reachable over https) rather than plain-http TestClient default.

@pg
def test_register_login_logout_http_round_trip():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_auth_http"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.invite_tokens (token, expires_at) VALUES (%s,%s)",
            ("invite-http", datetime.now(timezone.utc) + timedelta(days=7)),
        )
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash, is_admin) VALUES (%s,%s,true)",
            ("admin-http", auth.hash_password("adminpassword1")),
        )
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")

        # Anonymous: admin-gated route is refused.
        assert client.get("/api/pos-values").status_code == 403

        # Register consumes the invite and sets an httpOnly session cookie.
        res = client.post(
            "/api/auth/register",
            json={"token": "invite-http", "username": "erin", "password": "password123"},
        )
        assert res.status_code == 200, res.text
        assert res.json()["user"]["username"] == "erin"
        assert res.json()["user"]["is_admin"] is False
        set_cookie = res.headers.get("set-cookie", "")
        assert "HttpOnly" in set_cookie and "Secure" in set_cookie

        # Reusing the same invite token is rejected.
        res = client.post(
            "/api/auth/register",
            json={"token": "invite-http", "username": "frank", "password": "password123"},
        )
        assert res.status_code == 400

        # The new session is a viewer, not an admin -- confirms the cookie
        # round-tripped (client.cookies carries it automatically now) and
        # that require_admin still rejects a non-admin session over real HTTP.
        assert client.get("/api/auth/me").json()["user"]["username"] == "erin"
        assert client.get("/api/pos-values").status_code == 403

        client.post("/api/auth/logout")
        assert client.get("/api/auth/me").json()["user"] is None

        # Wrong password is rejected without revealing which part was wrong.
        res = client.post("/api/auth/login", json={"username": "admin-http", "password": "wrong"})
        assert res.status_code == 401

        # Correct admin login passes the admin-gated route over real HTTP.
        res = client.post("/api/auth/login", json={"username": "admin-http", "password": "adminpassword1"})
        assert res.status_code == 200
        assert res.json()["user"]["is_admin"] is True
        assert client.get("/api/pos-values").status_code == 200
    finally:
        main.SCHEMA = old_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()
