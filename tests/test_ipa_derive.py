"""Deriving IPA from a base word's (concordance/ipa_derive.py). Pure logic
over a small hand-made lexicon -- no DB."""

from __future__ import annotations

from concordance import ipa_derive as d

LEX = {
    "bloom": "bluːm", "masticate": "ˈmæstɪkeɪt", "happy": "ˈhæpi", "gentle": "ˈdʒɛntl̩",
    "ghost": "ɡoʊst", "rain": "reɪn", "cape": "keɪp", "shapeless": "ˈʃeɪpləs", "kiss": "kɪs",
    "walk": "wɔk", "want": "wɑnt", "play": "pleɪ", "tickle": "ˈtɪkl̩", "huffy": "ˈhʌfi",
    "compose": "kəmˈpoʊz", "betoken": "bɪˈtoʊkən", "souse": "saʊs", "sous": "suː",
    "monk": "mʌŋk", "stop": "stɑp", "corn": "kɔrn", "supersede": "ˌsupərˈsid",
}


def deriver(**kw):
    return d.Deriver(dict(LEX), **kw)


def test_clean_takes_one_plain_transcription():
    assert d.clean("/ˈsaɪ̯d/ [ˈsaɪ̯d]") == "ˈsaɪd"
    assert d.clean("/ˈd͡ʒɛntl̩/") == "ˈdʒɛntl̩"
    assert d.clean("/pleɪ̯/[pɫ̥eɪ̯~pʰɫeɪ̯]") == "pleɪ"
    assert d.clean("-ˈnjut") is None


def test_pick_goes_with_the_majority_not_an_odd_homograph():
    entries = [{"ipa": "/ˌɛs oʊ ˈbi/", "tags": ["General-American"]}, {"ipa": "/sɒb/", "tags": []},
               {"ipa": "/sɑb/", "tags": ["General-American"]}, {"ipa": "/sɑb/", "tags": []}]
    assert d._pick(entries) == "sɑb"


def test_s_and_ed_take_their_sound_from_the_base():
    assert d.add_s("kɪs") == "ˈkɪsɪz"
    assert d.add_s("wɔk") == "wɔks"
    assert d.add_s("pleɪ") == "pleɪz"
    assert d.add_ed("wɑnt") == "ˈwɑntɪd"
    assert d.add_ed("wɔk") == "wɔkt"
    assert d.add_ed("pleɪ") == "pleɪd"


def test_suffixes_and_spelling_seams():
    r = deriver()
    assert r.derive("masticating").ipa == "ˈmæstɪkeɪtɪŋ"          # dropped e
    assert r.derive("stopping").ipa == "ˈstɑpɪŋ"                  # doubled p
    assert r.derive("happiness").ipa == "ˈhæpinəs"                # y -> i
    assert r.derive("gently").ipa == "ˈdʒɛntli"                   # -le + -ly
    assert r.derive("tickling").ipa == "ˈtɪklɪŋ"                  # syllabic l lost
    assert r.derive("huffily").ipa == "ˈhʌfɪli"
    assert r.derive("ghosthood").ipa == "ˈɡoʊsthʊd"               # monosyllable gets the stress
    assert r.derive("betokeneth").ipa == "bɪˈtoʊkənɪθ"
    r.lexicon["cudgel"] = "ˈkʌdʒəl"
    assert r.derive("cudgelling").ipa == "ˈkʌdʒəlɪŋ"                # British doubled l, not -ling


def test_two_levels_deep_and_how_it_was_built():
    got = deriver().derive("bloomlessness")
    assert got.ipa == "ˈbluːmləsnəs"
    assert got.how == "bloom + -lessness"


def test_ed_is_a_syllable_before_ly():
    assert deriver().derive("composedly").ipa == "kəmˈpoʊzɪdli"


def test_a_real_word_base_beats_a_stem_and_the_commoner_wins():
    r = deriver(roots={"superseder": "supersede"})
    assert r.derive("superseder").how == "supersede + -er"
    r.lexicon.update({"hope": "hoʊp", "hop": "hɑp"})
    assert r.derive("hoped").how == "hope + -ed"
    assert r.derive("hopped").how == "hop + -ed"


def test_risky_affixes_need_wiktionary_to_name_the_root():
    assert deriver().derive("corner") is None                     # not corn + -er
    assert deriver(roots={"cornish": "corn"}).derive("cornish") is not None   # -ish isn't risky


def test_prefix_keeps_the_base_stress():
    assert deriver().derive("unmonklike").ipa == "ʌnˈmʌŋkˌlaɪk"


def test_compound_stress_is_on_the_first_part():
    r = deriver(compounds={"raincape": ["rain", "cape"]})
    assert r.derive("raincape").ipa == "ˈreɪnˌkeɪp"
    # an affix posing as a compound part isn't one
    assert deriver(compounds={"monktor": ["monk", "tor"]}).derive("monktor") is None


def test_definitions_name_the_base():
    assert d.base_from_definition("Obsolete spelling of shapeless.") == ("spelling", "shapeless")
    assert d.base_from_definition("Present participle of masticate.") == ("inflection", "masticate")
    assert d.base_from_definition("Pronunciation spelling of shapeless.") is None
    assert d.base_from_definition("Abbreviated spelling of tutorial.") is None
    r = deriver()
    assert d.derive_word(r, "shapelesse", "Obsolete spelling of shapeless.").ipa == "ˈʃeɪpləs"
    # a "spelling" that changes the consonants is a different form, not the same sound
    assert d.derive_word(r, "shapely", "Alternative spelling of shapeless.") is None


def test_fold_ignores_dialect_and_notation():
    assert d.fold("ˈwɜːldfʊl") == d.fold("ˈwɝldfəl")
    assert d.fold("ˈhʌfɪli") == d.fold("ˈhʌfili")
