"""Session/password/Cloudflare-Access auth for the webapp's own login.

Separate from Cloudflare Access, which gates the admin curation UI at the network
edge and has no concept of "who" beyond "passed Access." This module gives the app
its own users, independent of that -- non-admin accounts never touch Cloudflare
Access at all.

Pure functions here (hashing, session CRUD, CF JWT verification) take `conn`/
`schema` explicitly rather than reaching for a module-level connection, so they're
usable from both main.py's FastAPI dependencies and the `create_admin` CLI command
without a circular import.
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Request

SESSION_COOKIE_NAME = "concordance_session"
SESSION_LIFETIME = timedelta(days=30)

_ph = PasswordHasher()


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _ph.verify(password_hash, password)
    except VerifyMismatchError:
        return False


# --- sessions ---------------------------------------------------------------

def create_session(conn, schema: str, user_id: int) -> tuple[str, datetime]:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + SESSION_LIFETIME
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.sessions (token, user_id, expires_at) VALUES (%s,%s,%s)",
            (token, user_id, expires_at),
        )
    conn.commit()
    return token, expires_at


def get_session_user(conn, schema: str, token: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT u.id, u.username, u.is_admin FROM {schema}.sessions s
                JOIN {schema}.users u ON u.id = s.user_id
                WHERE s.token = %s AND s.expires_at > now()""",
            (token,),
        )
        row = cur.fetchone()
    return {"id": row[0], "username": row[1], "is_admin": row[2]} if row else None


def destroy_session(conn, schema: str, token: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {schema}.sessions WHERE token = %s", (token,))
    conn.commit()


# --- Cloudflare Access JWT verification -------------------------------------
# Redundant extra layer on admin-only routes, not load-bearing (see the plan's
# "Admin auth model" note -- app sessions are the reliable mechanism). Fails
# closed if unconfigured so an unset .env just means this branch never fires,
# not a crash.

_CF_TEAM_DOMAIN = os.environ.get("CF_ACCESS_TEAM_DOMAIN")
_CF_AUD = os.environ.get("CF_ACCESS_AUD")
_jwks_client = jwt.PyJWKClient(f"https://{_CF_TEAM_DOMAIN}/cdn-cgi/access/certs") if _CF_TEAM_DOMAIN else None


def verify_cf_access(request: Request) -> dict | None:
    if _jwks_client is None or not _CF_AUD:
        return None
    token = request.headers.get("Cf-Access-Jwt-Assertion") or request.cookies.get("CF_Authorization")
    if not token:
        return None
    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        return jwt.decode(token, signing_key.key, algorithms=["RS256"], audience=_CF_AUD)
    except jwt.PyJWTError:
        return None
