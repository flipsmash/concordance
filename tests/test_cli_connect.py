"""Every DB command opens with cli._connect: an unreachable database is a
one-line error and exit 1, never a traceback."""

from __future__ import annotations

from typer.testing import CliRunner

from concordance.cli import app

_BAD_URL = "postgresql://nobody@127.0.0.1:1/none?connect_timeout=2"


def test_unreachable_database_exits_cleanly():
    result = CliRunner().invoke(app, ["load-taxonomy", "--database-url", _BAD_URL])
    assert result.exit_code == 1
    assert "cannot connect" in result.output
    assert "Traceback" not in result.output


def test_connect_hint_is_printed_where_asked(tmp_path):
    csv = tmp_path / "master_vocab.csv"
    csv.write_text("word\n")
    result = CliRunner().invoke(app, ["sync-db", str(csv), "--database-url", _BAD_URL])
    assert result.exit_code == 1
    assert "set DATABASE_URL" in result.output
