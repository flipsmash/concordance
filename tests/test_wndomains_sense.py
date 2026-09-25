"""Sense-aware WordNet-Domains prior (needs the licensed, git-ignored data)."""
from __future__ import annotations

import pytest

from concordance import wndomains

pytestmark = pytest.mark.skipif(not wndomains.SENSE_LEXICON_PATH.exists(),
                                reason="per-sense lexicon not built (licensed data)")


def test_fagot_bundle_sense_gets_no_slur_hint():
    assert wndomains.usas_prior("fagot") == {"S3.2": 1}              # the old, sense-blind merge
    assert not wndomains.usas_prior_for_sense("fagot", "A bundle of sticks or brushwood tied together")


def test_disagreeing_senses_pick_the_matching_one_or_abstain():
    assert "W3" in wndomains.usas_prior_for_sense("bank", "sloping land beside a body of water")
    assert "I1" in wndomains.usas_prior_for_sense("bank", "a financial institution that accepts deposits")
    assert not wndomains.usas_prior_for_sense("bank", "")             # no gloss -> no guess
