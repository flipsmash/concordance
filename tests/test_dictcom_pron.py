"""dictionary.com pronunciation parsing (concordance/dictcom_pron.py) and its
tiers in the IPA cascade. Pages are trimmed copies of the real markup."""

from __future__ import annotations

from concordance import audio, dictcom_pron as dc
from concordance import resolve_pronunciation as rp


def _box(headword, group, ipa=None, pos=(), tag="span"):
    ipa_html = f'<span class="txt-ipa">{ipa}</span>' if ipa else ""
    pos_html = "".join(f'<div class="box-pos-label"><h2>{p}</h2></div>' for p in pos)
    return (f'<div class="box-headword-audio-pronunciation"><div class="box-headword-type">'
            f"<{tag} class='hdr-headword' id='x'>{headword}</{tag}> <sup>1</sup>"
            f'<em class="txt-entry-group-type {group}">X</em></div>'
            f'<div class="box-audio-pronunciation"><span class="txt-phonetic">[x]</span>{ipa_html}</div></div>'
            f'<div class="box-posb-content">{pos_html}</div>')


MINUTE = (_box("minute", "dcom", "/ ˈmɪn ɪt /", ["noun", "verb (used with object)", "adjective"], tag="h1")
          + _box("minute", "dcom", "/ maɪˈnut, -ˈnjut, mɪ- /", ["adjective"])
          + _box("minute", "british", "/ ˈmɪnɪt /", ["noun", "verb"])
          + _box("minute", "british", "/ maɪˈnjuːt /", ["adjective"])
          + _box("minute", "science", "/ mĭn"))


def test_parse_entries_reads_each_headword_block():
    entries = dc.parse_entries(MINUTE)
    assert [(e.headword, e.group) for e in entries] == [
        ("minute", "dcom"), ("minute", "dcom"), ("minute", "british"), ("minute", "british"), ("minute", "science")]
    assert entries[1].ipa == "/ maɪˈnut, -ˈnjut, mɪ- /"
    assert entries[0].pos[0] == "noun"


def test_clean_ipa_takes_first_full_variant_without_syllable_spaces():
    assert dc.clean_ipa("/ hjuˈmɛk tənt, ju- /") == "hjuˈmɛktənt"
    assert dc.clean_ipa("/ lɪˈsɒ trɪ kəs /") == "lɪˈsɒtrɪkəs"
    assert dc.clean_ipa("/ -ˈnjut /") is None
    assert dc.clean_ipa(None) is None


def test_homographs_are_settled_by_part_of_speech():
    entries = dc.parse_entries(MINUTE)
    # minute¹ also lists an adjective sense; the entry led by "adjective" wins
    assert dc.choose_ipa(entries, "minute", "adjective") == ("maɪˈnut", "maɪˈnjuːt")
    assert dc.choose_ipa(entries, "minute", "noun") == ("ˈmɪnɪt", "ˈmɪnɪt")
    # disagreeing homographs and no part of speech: no guess
    assert dc.choose_ipa(entries, "minute", None) == (None, None)


def test_only_the_exact_headword_counts():
    page = _box("humectant", "dcom", "/ hjuˈmɛk tənt /", ["noun"], tag="h1")
    assert dc.choose_ipa(dc.parse_entries(page), "humectate", "verb") == (None, None)


def test_british_only_entry():
    page = _box("zwischenzug", "british", "/ ˈzvɪʃənzuːɡ /", ["noun"], tag="h1")
    assert dc.choose_ipa(dc.parse_entries(page), "zwischenzug", "noun") == (None, "ˈzvɪʃənzuːɡ")


def test_cascade_uses_dictionary_com_only_when_nothing_else_has_it():
    assert rp.resolve_ipa(dictcom_ipa="lɪˈsɒtrɪkəs") == ("lɪˈsɒtrɪkəs", rp.Tier.DICTIONARY_COM)
    assert rp.resolve_ipa(dictcom_ipa_uk="ˈzvɪʃənzuːɡ") == ("ˈzvɪʃənzuːɡ", rp.Tier.DICTIONARY_COM_UK)
    hit = rp.resolve_ipa(local_entries=[("noun", "def", "ˈlkl", False, False)], dictcom_ipa="ˈdc")
    assert hit == ("ˈlkl", rp.Tier.LOCAL_WIKTIONARY)


def test_british_dictionary_com_ipa_gets_the_uk_voice():
    assert audio.ipa_dialect_for_source("dictionary_com_uk") == "uk"
    assert audio.ipa_dialect_for_source("dictionary_com") == "us"
