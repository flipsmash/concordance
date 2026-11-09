"""Admin "edit spelling" API: respell a word (or merge it into the word that
already has that spelling) and start the background upkeep.

The respelling itself -- rename, alias, fresh definition lookup, clearing
what the old spelling derived -- happens in the request (concordance/
respell.py). The slow part (categories, quiz definition and the rest; it
loads the local model) runs as `concordance refresh-words` in its own
systemd unit, so restarting this web service can't kill it mid-run.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from concordance import respell
from webapp.backend import jobs
from webapp.backend import main as _main

router = APIRouter()


class RespellRequest(BaseModel):
    spelling: str
    merge: bool = False


class RespellResult(BaseModel):
    word_id: int
    lemma: str
    merged: bool
    definition_refreshed: bool


def start_refresh_job() -> None:
    jobs.start("refresh-words", _main.SCHEMA)


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
