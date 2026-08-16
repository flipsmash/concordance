"""Admin-only read-only browsing of the `oed` schema (§ OED browsing) --
entries/definitions/quotations/pronunciations extracted from scanned OED
volume PDFs by concordance/oed/pipeline.py's CLI ingest, previously only
inspectable via direct SQL. Every endpoint here is `require_admin`, not
`require_viewer` -- unlike browse.py's end-user vocab surface, this is a
curation/QA tool over raw ingest output (needs-review flags, page numbers,
etc.), not something a reader should see.

Imports `main` as a module and always accesses `_main.get_conn()`/
`_main.require_admin` via dotted attribute lookup, not a bare `from ...
import`, matching browse.py/quiz.py/progress.py -- same reason: registered
into `app` at the bottom of main.py, after those names are defined, so a
bare import would freeze an unset value.

Deliberately does NOT use `_main.SCHEMA` (`"concordance"`) -- the `oed`
schema is a second, fixed schema in the same database, not a per-request
configurable one, so `OED_SCHEMA` is imported directly from
concordance.oed.db rather than threaded through as a query param.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from concordance.oed.db import DEFAULT_SCHEMA as OED_SCHEMA
from webapp.backend import main as _main

router = APIRouter()


class OedVolumeRow(BaseModel):
    id: int
    file_name: str
    volume_label: str | None
    status: str
    page_count: int | None


@router.get("/api/admin/oed/volumes", response_model=list[OedVolumeRow])
def oed_volumes(_: dict = Depends(_main.require_admin)) -> list[OedVolumeRow]:
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, file_name, volume_label, status, page_count
                FROM {OED_SCHEMA}.volume ORDER BY id"""
        )
        rows = cur.fetchall()
    return [
        OedVolumeRow(id=r[0], file_name=r[1], volume_label=r[2], status=r[3], page_count=r[4])
        for r in rows
    ]


class OedEntryRow(BaseModel):
    id: int
    headword: str
    homograph_number: int | None
    part_of_speech: str | None
    page_number: int
    volume_id: int
    pronunciation_ipa: str | None
    pronunciation_needs_review: bool
    first_definition: str | None
    concordance_match: str | None  # 'accepted' | 'pruned' | 'rejected' | 'unique' | null (not yet
                                    # checked, or not a lemma entry -- see oed-concordance-match)


class OedEntryPage(BaseModel):
    items: list[OedEntryRow]
    total: int
    page: int
    page_size: int


_SORT_COLUMNS = {
    "headword": "e.headword_norm",
    "page_number": "e.page_number",
    "created_at": "e.created_at",
}


@router.get("/api/admin/oed/entries", response_model=OedEntryPage)
def oed_entries(
    q: str | None = None,
    volume_id: int | None = None,
    letter: str | None = Query(None, min_length=1, max_length=1),
    needs_review: bool | None = None,
    concordance_match: Literal["accepted", "pruned", "rejected", "unique", "unchecked"] | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sort: Literal["headword", "page_number", "created_at"] = "headword",
    dir: Literal["asc", "desc"] = "asc",
    _: dict = Depends(_main.require_admin),
) -> OedEntryPage:
    filters = []
    params: list = []
    if q:
        # headword_norm is already lower(headword) (a generated column, see
        # oed/db.py) -- plain LIKE against a pre-lowered param is equivalent
        # to ILIKE here without the extra per-row lower() call ILIKE implies.
        filters.append("e.headword_norm LIKE %s")
        params.append(f"%{q.lower()}%")
    if volume_id is not None:
        filters.append("e.volume_id = %s")
        params.append(volume_id)
    if letter:
        filters.append("left(e.headword_norm, 1) = %s")
        params.append(letter.lower())
    if needs_review is not None:
        filters.append("e.pronunciation_needs_review = %s")
        params.append(needs_review)
    if concordance_match == "unchecked":
        filters.append("e.lemma AND e.concordance_match IS NULL")
    elif concordance_match is not None:
        filters.append("e.concordance_match = %s")
        params.append(concordance_match)
    where = " AND ".join(filters) if filters else "true"

    order_col = _SORT_COLUMNS[sort]
    order_dir = "ASC" if dir == "asc" else "DESC"

    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {OED_SCHEMA}.entry e WHERE {where}", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""SELECT e.id, e.headword, e.homograph_number, e.part_of_speech,
                       e.page_number, e.volume_id, e.pronunciation_ipa,
                       e.pronunciation_needs_review, e.concordance_match
                FROM {OED_SCHEMA}.entry e
                WHERE {where}
                ORDER BY {order_col} {order_dir}, e.id ASC
                LIMIT %s OFFSET %s""",
            (*params, page_size, (page - 1) * page_size),
        )
        rows = cur.fetchall()

        # One extra query scoped to just this page's entries (never the
        # whole 20k+ table) -- DISTINCT ON picks each entry's first
        # definition by its own sort_order, matching what the detail
        # endpoint treats as sense 1.
        entry_ids = [r[0] for r in rows]
        first_def_by_entry: dict[int, str] = {}
        if entry_ids:
            cur.execute(
                f"""SELECT DISTINCT ON (entry_id) entry_id, definition_text
                    FROM {OED_SCHEMA}.definition
                    WHERE entry_id = ANY(%s)
                    ORDER BY entry_id, sort_order""",
                (entry_ids,),
            )
            first_def_by_entry = dict(cur.fetchall())

    items = [
        OedEntryRow(
            id=r[0], headword=r[1], homograph_number=r[2], part_of_speech=r[3],
            page_number=r[4], volume_id=r[5], pronunciation_ipa=r[6],
            pronunciation_needs_review=r[7], first_definition=first_def_by_entry.get(r[0]),
            concordance_match=r[8],
        )
        for r in rows
    ]
    return OedEntryPage(items=items, total=total, page=page, page_size=page_size)


class OedQuotationOut(BaseModel):
    id: int
    year_raw: str | None
    year: int | None
    year_approx: bool
    author: str | None
    source_title: str | None
    quoted_text: str


class OedDefinitionOut(BaseModel):
    id: int
    sense_label: str | None
    definition_text: str
    quotations: list[OedQuotationOut]


class OedEntryDetail(BaseModel):
    id: int
    volume_id: int
    headword: str
    homograph_number: int | None
    part_of_speech: str | None
    etymology: str | None
    entry_type: str
    page_number: int
    raw_text: str
    pronunciation_ipa: str | None
    pronunciation_raw: str | None
    pronunciation_needs_review: bool
    concordance_match: str | None
    concordance_word_id: int | None  # set only for 'accepted'/'pruned' -- links to the matching
                                      # concordance.word row; null for 'rejected'/'unique'/unchecked
    definitions: list[OedDefinitionOut]


@router.get("/api/admin/oed/entries/{entry_id}", response_model=OedEntryDetail)
def oed_entry_detail(entry_id: int, _: dict = Depends(_main.require_admin)) -> OedEntryDetail:
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, volume_id, headword, homograph_number, part_of_speech,
                       etymology, entry_type, page_number, raw_text,
                       pronunciation_ipa, pronunciation_raw, pronunciation_needs_review,
                       concordance_match, headword_norm
                FROM {OED_SCHEMA}.entry WHERE id = %s""",
            (entry_id,),
        )
        entry_row = cur.fetchone()
        if entry_row is None:
            raise HTTPException(status_code=404, detail="entry not found")

        concordance_word_id = None
        if entry_row[12] in ("accepted", "pruned"):
            cur.execute(f"SELECT id FROM {_main.SCHEMA}.word WHERE lemma_lc = %s", (entry_row[13],))
            match_row = cur.fetchone()
            concordance_word_id = match_row[0] if match_row else None

        cur.execute(
            f"""SELECT id, sense_label, definition_text
                FROM {OED_SCHEMA}.definition WHERE entry_id = %s ORDER BY sort_order""",
            (entry_id,),
        )
        def_rows = cur.fetchall()
        def_ids = [r[0] for r in def_rows]

        quotations_by_def: dict[int, list[OedQuotationOut]] = {d_id: [] for d_id in def_ids}
        if def_ids:
            cur.execute(
                f"""SELECT id, definition_id, year_raw, year, year_approx, author,
                           source_title, quoted_text
                    FROM {OED_SCHEMA}.quotation WHERE definition_id = ANY(%s)
                    ORDER BY definition_id, sort_order""",
                (def_ids,),
            )
            for r in cur.fetchall():
                quotations_by_def[r[1]].append(
                    OedQuotationOut(
                        id=r[0], year_raw=r[2], year=r[3], year_approx=r[4],
                        author=r[5], source_title=r[6], quoted_text=r[7],
                    )
                )

    definitions = [
        OedDefinitionOut(
            id=d[0], sense_label=d[1], definition_text=d[2],
            quotations=quotations_by_def.get(d[0], []),
        )
        for d in def_rows
    ]

    return OedEntryDetail(
        id=entry_row[0], volume_id=entry_row[1], headword=entry_row[2],
        homograph_number=entry_row[3], part_of_speech=entry_row[4],
        etymology=entry_row[5], entry_type=entry_row[6], page_number=entry_row[7],
        raw_text=entry_row[8], pronunciation_ipa=entry_row[9],
        pronunciation_raw=entry_row[10], pronunciation_needs_review=entry_row[11],
        concordance_match=entry_row[12], concordance_word_id=concordance_word_id,
        definitions=definitions,
    )
