import os

import pytest

from concordance import db, origins
from concordance.origins import Origin, Resolver, family, oed_languages, peel_bases


def test_family_by_code_ety_only_code_and_prefix():
    assert family("la") == family("ML.") == family("la-med") == "Latin"
    assert family("grc-koi") == "Greek" and family("qsb-grc") == "Greek"
    assert family("frk") == "Dutch, German & Frankish"
    assert family("ru") == "Slavic" and family("akk") == "Hebrew & Semitic"
    assert family("xyz") == "Other" and family("*mixed*") == "Mixed" and family("obscure") == "Obscure origin"


def test_oed_languages_deepest_by_rank_and_cognates_ignored():
    # nearest is the first named, deepest by rank (not position)
    assert oed_languages("a. F. benignité, ad. L. benignitat-em") == ("la", "fr")
    # OCR'd Greek after "Gr." can contain a stray "F." -- Greek still wins
    assert oed_languages("mod.L., a. Gr. uSpoKe^aAos F. ...") == ("grc", "la")
    # everything after cf. is cognates
    assert oed_languages("f. L. nat-is + -form. Cf. F. natiforme") == ("la", "la")
    assert oed_languages("f. next + -al.") == (None, None)


def test_peel_bases_spelling_rules():
    assert "heavenly" in peel_bases("heavenliness")
    assert "nickel" in peel_bases("nickelled")
    b = peel_bases("rathed")
    assert b.index("rathe") < b.index("rath")                   # e restored first after -ed
    assert "smiling" in peel_bases("unsmiling")


def _resolver(etym=None, oed=(), text=None, defs=None, affix=None, cparts=None):
    etym = {k: ([tuple(x.split(":", 1)) for x in links], parts) for k, (links, parts) in (etym or {}).items()}
    return Resolver(etym, list(oed), text or {}, defs or {}, affix or {}, cparts or {})


def test_chain_root_stops_at_first_proto_and_routes():
    r = _resolver({
        ("en", "benignity"): (["enm:benignite", "fro:benignite", "la:benignitas"], []),
        ("en", "strong"): (["enm:strong", "ang:strang", "gmw-pro:*strang", "gem-pro:*strangaz"], []),
        ("gem-pro", "*strangaz"): (["osx:strang"], []),        # a sideways link past the proto stage
        ("en", "oppugn"): (["la:oppugnare"], []),
    })
    assert r.resolve("benignity") == Origin("la", "via:French", "wiktionary")
    assert r.resolve("strong") == Origin("ang", "inherited", "wiktionary")
    assert r.resolve("oppugn") == Origin("la", "direct", "wiktionary")


def test_dead_end_english_chain_defers_to_parts():
    r = _resolver({
        ("en", "greasy"): (["enm:gresi"], ["grease"]),
        ("en", "grease"): (["enm:grece", "xno:craisse", "la:crassus"], []),
    })
    o = r.resolve("greasy")
    assert o.root == "la" and o.route == "formed"


def test_pointer_oed_formation_and_derivative():
    r = _resolver(
        etym={("en", "amuse"): (["frm:amuser"], [])},
        oed=[(1, "ectypal", "f. next + -al."), (2, "ectype", "ad. Gr. ektupon"),
             (3, "blunge", "Echoic."), (4, "ninny", "Of obscure origin; cf. Sp. niño.")],
        defs={"amuze": "Obsolete form of amuse.", "ectypal": "x", "ectype": "y", "amusingness": "z",
              "amusing": "Entertaining."},   # bases are only peeled through real words
    )
    assert r.resolve("amuze") == Origin("frm", "direct", "pointer")
    assert r.resolve("ectypal").root == "grc"                    # f. next -> ectype -> Gr.
    assert r.resolve("blunge").root == "imitative"
    assert r.resolve("ninny").root == "obscure"                  # stated up front, before the cf.
    o = r.resolve("amusingness")
    assert o.root == "frm" and o.method == "derivative" and o.route == "formed"


def test_mutually_referring_entries_terminate():
    r = _resolver(oed=[(1, "alpha", "f. next + -ic."), (2, "beta", "f. prec. + -ic.")], defs={"alpha": "", "beta": ""})
    assert r.resolve("alpha") is None


def test_compounds_only_from_verified_parts():
    r = _resolver({("en", "strong"): (["ang:strang"], []), ("en", "house"): (["ang:hus"], []),
                   ("en", "pear"): (["la:pirum"], []), ("en", "let"): (["ang:lætan"], [])},
                  cparts={"stronghouse": ["strong", "house"]})
    assert r.resolve("stronghouse") == Origin("ang", "formed", "compound")
    assert r.resolve("pearlet") is None          # no verified split: never pear + let


_URL = os.environ.get("CONCORDANCE_TEST_DB_URL")


@pytest.mark.skipif(not _URL, reason="no test database")
def test_compute_writes_word_origin():
    schema = "cc_test_word_origin"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    with conn.cursor() as cur:
        for lemma, d in [("amuze", "Obsolete form of amuse."), ("zzqx", "no idea")]:
            cur.execute(f"INSERT INTO {schema}.word (lemma, definition, active) VALUES (%s, %s, true)", (lemma, d))
    conn.commit()
    r = _resolver({("en", "amuse"): (["frm:amuser"], [])}, defs={"amuze": "Obsolete form of amuse."})
    orig_load = origins.load
    origins.load = lambda *a, **k: r                 # synthetic sources; the real tables aren't in a test schema
    try:
        st = origins.compute(conn, schema, progress=lambda *_: None)
    finally:
        origins.load = orig_load
    assert st["words"] == 2 and st["resolved"] == 1
    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.lemma, o.root_lang, o.family, o.route, o.method FROM {schema}.word_origin o
                        JOIN {schema}.word w ON w.id = o.word_id""")
        assert cur.fetchall() == [("amuze", "frm", "French", "direct", "pointer")]
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
