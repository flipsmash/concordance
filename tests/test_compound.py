from concordance import compound


def test_wiktionary_split_is_the_only_one_offered():
    assert compound.candidate_splits("dovecote", ["dove", "cote"]) == [["dove", "cote"]]


def test_hyphenated_splits_at_hyphens():
    assert compound.candidate_splits("doss-house") == [["doss", "house"]]
    assert compound.candidate_splits("pocketa-pocketa!") == []


def test_every_common_word_boundary_is_offered():
    splits = compound.candidate_splits("eyeshot")
    assert ["eye", "shot"] in splits and ["eyes", "hot"] in splits
    assert compound.candidate_splits("toolroom") == [["tool", "room"]]
    assert compound.candidate_splits("destrier") == []     # no split into two common words
    assert compound.candidate_splits("x y") == []


def test_rarer_part_sets_the_blend():
    assert compound.rarer_part_zipf(["wood", "chopper"]) < compound.rarer_part_zipf(["wood", "land"])


def test_parse_verdict_accepts_only_offered_splits_and_known_verdicts():
    splits = [["eye", "shot"], ["eyes", "hot"]]
    assert compound.parse_verdict('{"split": "eye+shot", "verdict": "partly"}', splits) == (["eye", "shot"], "partly")
    assert compound.parse_verdict('Sure! {"split": "Eye + Shot", "verdict": "Transparent"}', splits) == \
        (["eye", "shot"], "transparent")
    assert compound.parse_verdict('{"split": null, "verdict": "not_compound"}', splits) == (None, "not_compound")
    assert compound.parse_verdict('{"split": "ey+eshot", "verdict": "opaque"}', splits) is None
    assert compound.parse_verdict('{"split": "eye+shot", "verdict": "maybe"}', splits) is None
    assert compound.parse_verdict('{"split": null, "verdict": "opaque"}', splits) is None
    assert compound.parse_verdict("no idea", splits) is None
