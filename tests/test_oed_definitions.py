"""oed.definitions -- definition lookup against the OED reference dataset.
Pure helpers first (no DB needed); definition_lexicon's bulk-query behavior
needs a real, disposable Postgres (see tests/test_oed_db.py's own `pg`
marker for setup)."""
from __future__ import annotations

import os

import pytest

from concordance import db
from concordance.oed import db as oed_db
from concordance.oed import definitions as oed_defs

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


# --- pure helpers -----------------------------------------------------------

def test_clean_one_sense_cuts_at_first_citation_year():
    text = "A stick of cosmetic for colouring the lips. Also attrib. 1880 E. James Amat."
    assert oed_defs._clean_one_sense(text) == "A stick of cosmetic for colouring the lips. Also attrib"


def test_clean_one_sense_handles_uppercase_circa_prefix_glued_to_year():
    # OED's circa/ante marker sits directly against the digits with no
    # word-boundary between them (C1400) -- parse.py's own _YEAR_RE is
    # lowercase-only and misses this; this module's cut regex must not.
    text = "The action of scarifying; an instance of this. C1400 tr. Secreta Secret."
    assert oed_defs._clean_one_sense(text) == "The action of scarifying; an instance of this"


def test_clean_one_sense_strips_see_quot_prefix():
    assert oed_defs._clean_one_sense("(See quot. 1959.) rest of text here that is long enough") == \
        "rest of text here that is long enough"


def test_plausible_rejects_unbalanced_bracket():
    # "veterinary"-style real failure: extract_etymology's bracket match
    # failed upstream, leaving an unclosed '[' in what looks like a sense.
    assert not oed_defs._plausible("[ad. L. veterinarius belonging to beasts of burden")


def test_plausible_rejects_too_short():
    assert not oed_defs._plausible("Obs. rare.")


def test_plausible_accepts_clean_balanced_text():
    assert oed_defs._plausible("A stick of cosmetic for colouring the lips, usu. a shade of pink or red.")


def test_pick_definition_stops_at_first_plausible_sense():
    # "lipstick"-style real failure: the second sense is actually the start
    # of an unrelated headword's entry (a cross-entry split_senses bleed) --
    # pick_definition must never even look at it once sense[0] is good.
    parts = [
        "A stick of cosmetic for colouring the lips. 1880 E. James Amat.",
        "Lipto place-name in Czechoslovakia.] A soft cheese originally made in Hungary.",
    ]
    assert oed_defs._pick_definition(parts) == "A stick of cosmetic for colouring the lips"


def test_pick_definition_falls_through_a_bad_first_sense():
    parts = [
        "[ad. L. unclosed etymology fragment with no real definition here",
        "A genuine, sufficiently long definition that passes every check here.",
    ]
    assert oed_defs._pick_definition(parts) == \
        "A genuine, sufficiently long definition that passes every check here"


def test_pick_definition_empty_when_nothing_plausible():
    assert oed_defs._pick_definition(["[unclosed", "(See quot. 1900.)"]) == ""


def test_pick_sense_prefers_pos_match():
    senses = [
        oed_defs.OedSense(entry_id=1, part_of_speech="v", etymology="", definition="to run quickly"),
        oed_defs.OedSense(entry_id=2, part_of_speech="sb", etymology="", definition="a fast pace"),
    ]
    assert oed_defs.pick_sense("NOUN", senses).entry_id == 2
    assert oed_defs.pick_sense("VERB", senses).entry_id == 1


def test_pick_sense_falls_back_to_first_when_no_pos_match_or_unknown():
    senses = [
        oed_defs.OedSense(entry_id=1, part_of_speech="v", etymology="", definition="a"),
        oed_defs.OedSense(entry_id=2, part_of_speech="sb", etymology="", definition="b"),
    ]
    assert oed_defs.pick_sense("ADJ", senses).entry_id == 1
    assert oed_defs.pick_sense("", senses).entry_id == 1


def test_pick_sense_single_entry_short_circuits():
    senses = [oed_defs.OedSense(entry_id=1, part_of_speech="", etymology="", definition="only one")]
    assert oed_defs.pick_sense("NOUN", senses).entry_id == 1


# --- definition_lexicon (needs a real DB) -----------------------------------

def test_definition_lexicon_empty_input_returns_empty_dict():
    assert oed_defs.definition_lexicon(None, set()) == {}


@pg
def test_definition_lexicon_degrades_gracefully_when_oed_schema_absent():
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS oed_test_defs_absent CASCADE")
    conn.commit()
    assert oed_defs.definition_lexicon(conn, {"whatever"}, schema="oed_test_defs_absent") == {}
    conn.close()


@pg
def test_definition_lexicon_bulk_lookup_filters_and_homographs():
    schema = "oed_test_defs"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    oed_db.apply_schema(conn, schema)

    volume_id = oed_db.upsert_volume(conn, file_name="test.pdf", file_hash_="abc123",
                                      volume_label="Test Volume", page_count=1, schema=schema)

    def _entry(headword, senses, *, lemma=True, homograph_number=None, part_of_speech="n"):
        entry_id = oed_db.insert_entry(
            conn, volume_id=volume_id, headword=headword, homograph_number=homograph_number,
            part_of_speech=part_of_speech, etymology=None, entry_type="main", parent_entry_id=None,
            page_number=1, raw_text="raw", schema=schema)
        oed_db.insert_definitions(conn, entry_id, senses, schema=schema)
        with conn.cursor() as cur:
            cur.execute(f"UPDATE {schema}.entry SET lemma=%s WHERE id=%s", (lemma, entry_id))
        return entry_id

    _entry("lipstick", [
        {"sense_label": None, "definition_text":
            "A stick of cosmetic for colouring the lips. 1880 E. James Amat."},
        {"sense_label": "f", "definition_text":
            "Lipto place-name in Czechoslovakia.] A soft cheese made in Hungary."},
    ])
    _entry("veterinary", [
        {"sense_label": None, "definition_text": "[ad. L. veterinarius belonging or"},
    ], part_of_speech="a sb")
    _entry("bank", [
        {"sense_label": None, "definition_text":
            "The rising ground bordering a river, sufficiently long to pass here."},
    ], homograph_number=1, part_of_speech="sb")
    _entry("bank", [
        {"sense_label": None, "definition_text":
            "An establishment for the custody and lending of money, long enough to pass."},
    ], homograph_number=2, part_of_speech="sb")
    _entry("notalemma", [
        {"sense_label": None, "definition_text":
            "This entry is not itself a lemma so it must not surface at all here."},
    ], lemma=False)
    conn.commit()

    lexicon = oed_defs.definition_lexicon(conn, {"lipstick", "veterinary", "bank", "notalemma"}, schema=schema)

    # lipstick: first sense used, second (cross-entry bleed) never surfaces
    assert len(lexicon["lipstick"]) == 1
    assert lexicon["lipstick"][0].definition.startswith("A stick of cosmetic")
    assert "cheese" not in lexicon["lipstick"][0].definition

    # veterinary: its only sense is an unclosed-bracket fragment -> filtered out entirely
    assert "veterinary" not in lexicon

    # bank: two homograph entries both survive as separate OedSenses
    assert len(lexicon["bank"]) == 2

    # a non-lemma entry never surfaces regardless of content quality
    assert "notalemma" not in lexicon

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()
