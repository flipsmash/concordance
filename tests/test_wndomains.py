"""WordNet-Domains -> USAS prior. The lexicon itself is licensed/absent in CI, so
these test the crosswalk and the parsing/mapping logic, not the shipped data."""

from __future__ import annotations

from collections import Counter

from concordance import usas, wndomains


def test_every_crosswalk_target_is_a_real_usas_code():
    valid = {c["code"] for c in usas.categories()}
    bad = {d: code for d, code in wndomains._WND_TO_USAS.items()
           if code is not None and code not in valid}
    assert not bad, bad


def test_usas_prior_applies_crosswalk(monkeypatch):
    monkeypatch.setattr(wndomains, "_lexicon", {"frigate": {"military", "nautical"}})
    assert wndomains.usas_prior("frigate") == Counter({"G3": 1, "M4": 1})


def test_prior_ignores_uncrosswalked_domains(monkeypatch):
    monkeypatch.setattr(wndomains, "_lexicon", {"x": {"history", "humanities", "law"}})
    # history/humanities are intentionally unmapped; only law -> G2.1 survives
    assert wndomains.usas_prior("x") == Counter({"G2.1": 1})


def test_domains_for_unknown_is_empty(monkeypatch):
    monkeypatch.setattr(wndomains, "_lexicon", {})
    assert wndomains.domains_for("zzz") == set()


def test_build_lexicon_parses_wn_and_wnd(tmp_path):
    d = tmp_path / "dict"; d.mkdir()
    # minimal WN1.6 data.noun: two synsets (header line ignored)
    (d / "data.noun").write_text(
        "  1 this is a license header line to skip\n"
        "00050037 04 n 02 frigate 0 warship 0 000 | a warship\n"
        "00001740 03 n 01 entity 0 000 | that which exists\n")
    for pos in ("verb", "adj", "adv"):
        (d / f"data.{pos}").write_text("  header\n")
    wnd = tmp_path / "wnd"
    wnd.write_text("00050037-n\tmilitary nautical\n00001740-n\tfactotum\n")
    out = tmp_path / "lex.tsv"
    n = wndomains.build_lexicon(d, wnd, out)
    assert n == 2                                   # frigate + warship (entity was factotum-only -> dropped)
    lines = dict(l.split("\t") for l in out.read_text().splitlines())
    assert set(lines["frigate"].split()) == {"military", "nautical"}
    assert "entity" not in lines
