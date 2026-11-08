"""Admin "edit spelling" API: respell a word (or merge it into the word that
already has that spelling) and start the background upkeep.

The respelling itself -- rename, alias, fresh definition lookup, clearing
what the old spelling derived -- happens in the request (concordance/
respell.py). The slow part (categories, quiz definition and the rest; it
loads the local model) runs as `concordance refresh-words` in its own
systemd unit, so restarting this web service can't kill it mid-run.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from concordance import respell
from webapp.backend import main as _main

router = APIRouter()
log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]


class RespellRequest(BaseModel):
    spelling: str
    merge: bool = False


class RespellResult(BaseModel):
    word_id: int
    lemma: str
    merged: bool
    definition_refreshed: bool


def start_refresh_job() -> None:
    """Run `concordance refresh-words` in the background, outside this
    service's cgroup. If systemd-run isn't available the job stays queued
    and the next refresh-words (or a later respelling) picks it up."""
    cmd = [sys.executable, "-m", "concordance", "refresh-words", "--schema", _main.SCHEMA]
    unit = f"concordance-refresh-{int(time.time())}"
    try:
        subprocess.run(["systemd-run", "--user", "--collect", f"--unit={unit}",
                        f"--working-directory={_REPO}", "--", *cmd],
                       check=True, capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("couldn't start the refresh job (%s); it stays queued", exc)


@router.post("/api/words/{word_id}/spelling", response_model=RespellResult)
def respell_word(word_id: int, body: RespellRequest, user: dict = Depends(_main.require_admin)) -> RespellResult:
    """409 when the spelling already belongs to another word: the body says
    which, so the page can ask whether to merge (and, for a word that was
    removed, that merging brings it back). Resend with merge=true."""
    with _main.get_conn() as conn:
        try:
            out = respell.respell(conn, _main.SCHEMA, word_id, body.spelling,
                                  by=user.get("username") or "admin", merge=body.merge)
        except respell.RespellCollision as exc:
            c = exc.collision
            raise HTTPException(status_code=409, detail={
                "message": f"“{c.lemma}” is already a word",
                "collision": {"word_id": c.word_id, "lemma": c.lemma, "active": c.active,
                              "inactive_reason": c.inactive_reason, "books": c.books}})
        except respell.RespellError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    start_refresh_job()
    return RespellResult(**out)
