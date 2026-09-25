"""Connection basics: the DATABASE_URL lookup, schema-name validation, connect()."""

from __future__ import annotations

import os
import re
from pathlib import Path

import psycopg

from ..deepdef import _load_dotenv


DEFAULT_SCHEMA = os.environ.get("CONCORDANCE_DB_SCHEMA", "concordance")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def database_url(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if "DATABASE_URL" not in os.environ:
        _load_dotenv(Path(".env"))
    return os.environ.get("DATABASE_URL", "").strip()


def _safe_schema(schema: str) -> str:
    if not _IDENT.match(schema):
        raise ValueError(f"unsafe schema name: {schema!r}")
    return schema


def connect(url: str | None = None) -> psycopg.Connection:
    resolved = database_url(url)
    if not resolved:
        raise RuntimeError("no DATABASE_URL set (env or .env)")
    return psycopg.connect(resolved)
