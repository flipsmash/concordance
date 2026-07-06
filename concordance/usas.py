"""The USAS semantic tagset (UCREL, Lancaster) — the category backbone (§ taxonomy).

Full published category system: 21 top-level discourse fields, ~230 categories in
a code-encoded hierarchy (A → A1 → A1.5 → A1.5.2). Source:
http://ucrel.lancs.ac.uk/usas/USASSemanticTagset.pdf

Parent is derived from the dotted code (A1.5.2 → A1.5 → A1 → A), skipping any tier
the tagset doesn't name explicitly. The operational/grammatical Z bins are kept for
fidelity but marked non-assignable so the classifier never files a real word there.
"""

from __future__ import annotations

# code<space>label, one per line — top-level fields are single letters.
_TAGSET = """\
A GENERAL & ABSTRACT TERMS
A1 General
A1.1.1 General actions, making etc.
A1.1.2 Damaging and destroying
A1.2 Suitability
A1.3 Caution
A1.4 Chance, luck
A1.5 Use
A1.5.1 Using
A1.5.2 Usefulness
A1.6 Physical/mental
A1.7 Constraint
A1.8 Inclusion/Exclusion
A1.9 Avoiding
A2 Affect
A2.1 Affect: Modify, change
A2.2 Affect: Cause/Connected
A3 Being
A4 Classification
A4.1 Generally kinds, groups, examples
A4.2 Particular/general; detail
A5 Evaluation
A5.1 Evaluation: Good/bad
A5.2 Evaluation: True/false
A5.3 Evaluation: Accuracy
A5.4 Evaluation: Authenticity
A6 Comparing
A6.1 Comparing: Similar/different
A6.2 Comparing: Usual/unusual
A6.3 Comparing: Variety
A7 Definite (+ modals)
A8 Seem
A9 Getting and giving; possession
A10 Open/closed; Hiding/Hidden; Finding; Showing
A11 Importance
A11.1 Importance: Important
A11.2 Importance: Noticeability
A12 Easy/difficult
A13 Degree
A13.1 Degree: Non-specific
A13.2 Degree: Maximizers
A13.3 Degree: Boosters
A13.4 Degree: Approximators
A13.5 Degree: Compromisers
A13.6 Degree: Diminishers
A13.7 Degree: Minimizers
A14 Exclusivizers/particularizers
A15 Safety/Danger
B THE BODY & THE INDIVIDUAL
B1 Anatomy and physiology
B2 Health and disease
B3 Medicines and medical treatment
B4 Cleaning and personal care
B5 Clothes and personal belongings
C ARTS & CRAFTS
C1 Arts and crafts
E EMOTIONAL ACTIONS, STATES & PROCESSES
E1 General
E2 Liking
E3 Calm/Violent/Angry
E4 Happy/sad
E4.1 Happy/sad: Happy
E4.2 Happy/sad: Contentment
E5 Fear/bravery/shock
E6 Worry, concern, confident
F FOOD & FARMING
F1 Food
F2 Drinks
F3 Cigarettes and drugs
F4 Farming & Horticulture
G GOVERNMENT & THE PUBLIC DOMAIN
G1 Government, Politics & elections
G1.1 Government etc.
G1.2 Politics
G2 Crime, law and order
G2.1 Crime, law and order: Law & order
G2.2 General ethics
G3 Warfare, defence and the army; Weapons
H ARCHITECTURE, BUILDINGS, HOUSES & THE HOME
H1 Architecture, kinds of houses & buildings
H2 Parts of buildings
H3 Areas around or near houses
H4 Residence
H5 Furniture and household fittings
I MONEY & COMMERCE
I1 Money generally
I1.1 Money: Affluence
I1.2 Money: Debts
I1.3 Money: Price
I2 Business
I2.1 Business: Generally
I2.2 Business: Selling
I3 Work and employment
I3.1 Work and employment: Generally
I3.2 Work and employment: Professionalism
I4 Industry
K ENTERTAINMENT, SPORTS & GAMES
K1 Entertainment generally
K2 Music and related activities
K3 Recorded sound etc.
K4 Drama, the theatre & show business
K5 Sports and games generally
K5.1 Sports
K5.2 Games
K6 Children's games and toys
L LIFE & LIVING THINGS
L1 Life and living things
L2 Living creatures generally
L3 Plants
M MOVEMENT, LOCATION, TRAVEL & TRANSPORT
M1 Moving, coming and going
M2 Putting, taking, pulling, pushing, transporting etc.
M3 Movement/transportation: land
M4 Movement/transportation: water
M5 Movement/transportation: air
M6 Location and direction
M7 Places
M8 Remaining/stationary
N NUMBERS & MEASUREMENT
N1 Numbers
N2 Mathematics
N3 Measurement
N3.1 Measurement: General
N3.2 Measurement: Size
N3.3 Measurement: Distance
N3.4 Measurement: Volume
N3.5 Measurement: Weight
N3.6 Measurement: Area
N3.7 Measurement: Length & height
N3.8 Measurement: Speed
N4 Linear order
N5 Quantities
N5.1 Entirety; maximum
N5.2 Exceeding; waste
N6 Frequency etc.
O SUBSTANCES, MATERIALS, OBJECTS & EQUIPMENT
O1 Substances and materials generally
O1.1 Substances and materials generally: Solid
O1.2 Substances and materials generally: Liquid
O1.3 Substances and materials generally: Gas
O2 Objects generally
O3 Electricity and electrical equipment
O4 Physical attributes
O4.1 General appearance and physical properties
O4.2 Judgement of appearance (pretty etc.)
O4.3 Colour and colour patterns
O4.4 Shape
O4.5 Texture
O4.6 Temperature
P EDUCATION
P1 Education in general
Q LINGUISTIC ACTIONS, STATES & PROCESSES
Q1 Communication
Q1.1 Communication in general
Q1.2 Paper documents and writing
Q1.3 Telecommunications
Q2 Speech acts
Q2.1 Speech etc: Communicative
Q2.2 Speech acts
Q3 Language, speech and grammar
Q4 The Media
Q4.1 The Media: Books
Q4.2 The Media: Newspapers etc.
Q4.3 The Media: TV, Radio & Cinema
S SOCIAL ACTIONS, STATES & PROCESSES
S1 Social actions, states & processes
S1.1 Social actions, states & processes: General
S1.1.1 General
S1.1.2 Reciprocity
S1.1.3 Participation
S1.1.4 Deserve etc.
S1.2 Personality traits
S1.2.1 Approachability and Friendliness
S1.2.2 Avarice
S1.2.3 Egoism
S1.2.4 Politeness
S1.2.5 Toughness; strong/weak
S1.2.6 Sensible
S2 People
S2.1 People: Female
S2.2 People: Male
S3 Relationship
S3.1 Relationship: General
S3.2 Relationship: Intimate/sexual
S4 Kin
S5 Groups and affiliation
S6 Obligation and necessity
S7 Power relationship
S7.1 Power, organizing
S7.2 Respect
S7.3 Competition
S7.4 Permission
S8 Helping/hindering
S9 Religion and the supernatural
T TIME
T1 Time
T1.1 Time: General
T1.1.1 Time: General: Past
T1.1.2 Time: General: Present; simultaneous
T1.1.3 Time: General: Future
T1.2 Time: Momentary
T1.3 Time: Period
T2 Time: Beginning and ending
T3 Time: Old, new and young; age
T4 Time: Early/late
W THE WORLD & OUR ENVIRONMENT
W1 The universe
W2 Light
W3 Geographical terms
W4 Weather
W5 Green issues
X PSYCHOLOGICAL ACTIONS, STATES & PROCESSES
X1 General
X2 Mental actions and processes
X2.1 Thought, belief
X2.2 Knowledge
X2.3 Learn
X2.4 Investigate, examine, test, search
X2.5 Understand
X2.6 Expect
X3 Sensory
X3.1 Sensory: Taste
X3.2 Sensory: Sound
X3.3 Sensory: Touch
X3.4 Sensory: Sight
X3.5 Sensory: Smell
X4 Mental object
X4.1 Mental object: Conceptual object
X4.2 Mental object: Means, method
X5 Attention
X5.1 Attention
X5.2 Interest/boredom/excited/energetic
X6 Deciding
X7 Wanting; planning; choosing
X8 Trying
X9 Ability
X9.1 Ability: Ability, intelligence
X9.2 Ability: Success and failure
Y SCIENCE & TECHNOLOGY
Y1 Science and technology in general
Y2 Information technology and computing
Z NAMES & GRAMMATICAL WORDS
Z0 Unmatched proper noun
Z1 Personal names
Z2 Geographical names
Z3 Other proper names
Z4 Discourse Bin
Z5 Grammatical bin
Z6 Negative
Z7 If
Z8 Pronouns etc.
Z9 Trash can
Z99 Unmatched
"""

# Z bins that are operational/grammatical, not real semantic homes for a word.
_NON_ASSIGNABLE = {"Z0", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z99"}


def _pairs() -> list[tuple[str, str]]:
    out = []
    for line in _TAGSET.strip().splitlines():
        code, _, label = line.strip().partition(" ")
        out.append((code, label.strip()))
    return out


def _parent_code(code: str, known: set[str]) -> str | None:
    """A5.1 -> A5; A1.5.2 -> A1.5; A1 -> A; A -> None. Skip tiers not in the set."""
    if "." in code:
        stem = code.rsplit(".", 1)[0]
        while stem and stem not in known:
            if "." in stem:
                stem = stem.rsplit(".", 1)[0]
            else:
                stem = stem[0]         # fall back to the single-letter field
                break
        return stem or None
    if len(code) > 1:                  # e.g. A1 -> A, A15 -> A
        return code[0]
    return None                        # single-letter top field


def categories() -> list[dict]:
    """One dict per USAS category: code, name, parent_code, level, assignable."""
    pairs = _pairs()
    known = {c for c, _ in pairs}
    rows = []
    for code, name in pairs:
        parent = _parent_code(code, known)
        level = 0 if parent is None else code.count(".") + 1
        rows.append({
            "code": code, "name": name, "parent_code": parent, "level": level,
            "assignable": code not in _NON_ASSIGNABLE,
        })
    return rows
