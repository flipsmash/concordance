"""Hand-picked example (anchor) pairs for analogy relations whose mined
anchor pool is tiny.

Example pairs for the one-hard-term style must be two ordinary words, and
the definition-pattern relations (purpose, agent, relates-to, ...) are
mined almost entirely from vocab definitions, so they yield a handful of
ordinary pairs at most -- 303 banked purpose items shared 2 anchors
(bottle : store, radar : detect). These plain pairs widen that pool.

`seed-analogy-anchors` inserts them as verified edges (source
'curated_anchor') with anchor_status left NULL, so review-analogy-anchors
and score-analogy-anchors still judge each one like any mined anchor. A
pair is skipped when either lemma is a vocab word (an anchor must be two
ordinary words) or not in WordNet with that part of speech.
"""

from __future__ import annotations

# relation_type -> (pos_a, pos_b, "a:b a:b ...")
_SEEDS: dict[str, tuple[str, str, str]] = {
    "definition_pattern_purpose": ("noun", "verb", """
        knife:cut pen:write broom:sweep oven:bake shovel:dig needle:sew ladder:climb
        thermometer:measure soap:wash towel:dry bridge:cross chair:sit bed:sleep cup:drink
        spoon:stir phone:call alarm:warn glue:stick rope:tie net:catch gun:shoot helmet:protect
        shield:protect vacuum:clean heater:warm fan:cool school:teach kitchen:cook lock:secure
        scale:weigh map:guide brush:paint camera:record hammer:pound comb:untangle
        refrigerator:cool engine:power ruler:measure eraser:erase pump:raise wallet:hold
        microphone:amplify plough:till sieve:strain""")
    ,
    "definition_pattern_agent": ("noun", "verb", """
        thief:steal pilot:fly surgeon:operate chef:cook tailor:sew author:write soldier:fight
        doctor:heal lawyer:advise coach:train student:learn pupil:study critic:judge actor:perform
        musician:play architect:design burglar:steal murderer:kill messenger:deliver
        servant:serve host:entertain prophet:predict borrower:borrow seller:sell buyer:buy
        voter:vote swimmer:swim runner:run climber:climb teacher:teach baker:bake singer:sing
        driver:drive painter:paint dancer:dance guard:protect witness:see judge:decide
        spy:watch nurse:care translator:translate explorer:explore""")
    ,
    "definition_pattern_relates_to": ("adjective", "noun", """
        marine:sea solar:sun lunar:moon dental:tooth urban:city rural:countryside nasal:nose
        aquatic:water feline:cat canine:dog maternal:mother paternal:father naval:navy oral:mouth
        verbal:word visual:sight musical:music medical:medicine financial:money nuclear:nucleus
        cardiac:heart coastal:coast tropical:tropics polar:pole royal:king domestic:home
        mental:mind annual:year nocturnal:night legal:law criminal:crime military:army
        academic:university economic:economy political:politics educational:education
        agricultural:farming electrical:electricity""")
    ,
    "definition_pattern_characterized_by": ("adjective", "noun", """
        hairy:hair sandy:sand rocky:rock cloudy:cloud muddy:mud rainy:rain windy:wind snowy:snow
        smoky:smoke noisy:noise dusty:dust salty:salt sunny:sun bloody:blood grassy:grass
        wealthy:wealth joyful:joy fearful:fear courageous:courage hopeful:hope stormy:storm
        icy:ice leafy:leaf watery:water greasy:grease""")
    ,
    "attribute": ("adjective", "noun", """
        tall:height heavy:weight hot:temperature fast:speed old:age deep:depth wide:width
        long:length strong:strength rich:wealth brave:courage beautiful:beauty honest:honesty
        safe:safety expensive:price young:age thick:thickness cold:temperature true:truth
        healthy:health loud:volume bright:brightness smart:intelligence large:size
        small:size slow:speed weak:strength poor:wealth""")
    ,
}

# Attribute edges run both ways in WordNet (heavy -> weight, weight -> heavy).
_REVERSED = {"attribute": ("noun", "adjective")}


def seed_pairs() -> list[tuple[str, str, str, str, str]]:
    """(relation_type, pos_a, pos_b, a, b) for every seed pair."""
    out = []
    for rtype, (pa, pb, text) in _SEEDS.items():
        pairs = [tuple(p.split(":")) for p in text.split()]
        out += [(rtype, pa, pb, a, b) for a, b in pairs]
        if rtype in _REVERSED:
            ra, rb = _REVERSED[rtype]
            out += [(rtype, ra, rb, b, a) for a, b in pairs]
    return out


def seed_anchors(conn, schema: str, log=print) -> dict:
    """Insert the seed pairs as verified, unreviewed edges. Idempotent."""
    from .analogies import _RELATION_FAMILY, _get_or_create_ordinary_term, _load_wordnet
    from .model import wordnet_pos

    wn = _load_wordnet()
    stats = {"pairs": 0, "inserted": 0, "existing": 0, "vocab_word": 0, "not_in_wordnet": 0}
    with conn.cursor() as cur:
        for rtype, pos_a, pos_b, a, b in seed_pairs():
            stats["pairs"] += 1
            ta = _get_or_create_ordinary_term(cur, schema, wn, a, wordnet_pos(pos_a))
            tb = _get_or_create_ordinary_term(cur, schema, wn, b, wordnet_pos(pos_b))
            if ta is None or tb is None:
                stats["not_in_wordnet"] += 1
                continue
            if ta["word_id"] is not None or tb["word_id"] is not None:
                stats["vocab_word"] += 1
                continue
            cur.execute(
                f"""INSERT INTO {schema}.word_relation_edge
                        (term_a_id, term_b_id, relation_type, relation_family, pos_a, pos_b, source,
                         verification_status, verification_note, verifier_model, verified_at)
                    VALUES (%s, %s, %s, %s, %s, %s, 'curated_anchor', 'verified',
                            'hand-picked example pair', 'curated', now())
                    ON CONFLICT (term_a_id, term_b_id, relation_type) DO NOTHING RETURNING id""",
                (ta["id"], tb["id"], rtype, _RELATION_FAMILY[rtype], pos_a, pos_b),
            )
            stats["inserted" if cur.fetchone() else "existing"] += 1
    conn.commit()
    log(f"[seed-analogy-anchors] {stats}")
    return stats
