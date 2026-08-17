# Design rationale & philosophy

This document is the "why" behind Concordance. [README.md](README.md) is the
"what" and "how to run it" — start there if you just want to use the thing.
Everything here is a decision that was made deliberately, usually after an
earlier approach broke in a specific, real way. Where that happened, the
evidence is kept alongside the decision rather than paraphrased away.

## Keep-biased, always

The governing principle, stated once so every other section can assume it: a
genuine rarity should survive to your review even at the cost of a little
noise, never the reverse. Losing a real word silently is a worse failure than
making you glance at one more junk row. This shows up everywhere:

- The frequency floor is a floor only — there is deliberately no rarity
  *ceiling*. Mid-frequency words are exactly where the good vocabulary lives,
  so rarity is never used to rule a word *in* or discard it for being obscure.
- The LLM judge emits a minimal per-word verdict (`{"w","k"}`, no free-text
  reason) specifically so a weak local model doesn't truncate its own output
  and silently drop words. Any word the model omits from its response is
  re-queried for up to three passes before the fallback — which is to *keep*
  it, not drop it.
- Nothing the pipeline rejects is deleted outright — see "soft delete" below.
- A word that looks like a foreign term or an archaic-spelling variant gets
  *flagged* for human review, not auto-rejected — see "human review, not
  auto-reject" below.

## Judgment vs. rules — why a floor can't do the judge's job

Frequency alone cannot decide what's worth learning: *tendril* is rarer than
*refectory* by any corpus-frequency measure, yet everyone knows the first and
not the second. A pure frequency cut would either let *tendril*-class words
through (defeating the point) or floor out *refectory*-class words (losing
genuine rarities). This is why the pipeline splits the job in two: a cheap,
deterministic floor removes the unambiguously-too-common tail (stop words,
"table," "said"), and everything above that floor goes to a local LLM judge
whose actual job is the *judgment call*, not the frequency math. A corpus
frequency band (common/uncommon/rare) is passed to the judge as a hint that
steadies its sense of rarity, but it is never a hard cut — the judgment stays
the model's.

## Multi-source validity — never trust one authority

A word is treated as real if *any* authority vouches for it, checked in cost
order: the local `vocab.wiktionary` dump (~500k terms, free, and — unlike
every frequency-derived authority below it — carries no "Proper noun" POS
category to get confused by real names with a big web footprint), then the
SymSpell 82k wordlist, WordNet, and NLTK's 234k-word corpus (which carries the
archaic vocabulary — *destrier*, *bartizan*, *cangue* — that trips up
single-dictionary checks). A foreign-language-context check runs early, since
authorities can be fooled by loanwords sitting in a foreign quotation. Only
after all of that is misspelling considered, and even then only by *relative*
near-neighbor frequency, with a recurrence escape hatch: a "misspelling" that
keeps showing up across a book, or across books, is more likely a real
coinage than a typo. No single source is trusted alone, because no single
source is complete — proper-noun pollution, archaic-vocabulary gaps, and
frequency-derived name contamination each affect a different subset of
authorities, and a word only needs to clear one.

## Human review, not auto-reject, when the signal isn't reliable enough

Two systems in this pipeline flag a word for a person to glance at rather
than making the call automatically, both because a real-scale test found the
automated signal too unreliable to trust unattended:

- **`variant_flag_reason`** (foreign word / archaic-spelling-of-a-common-word
  detection, `validity_score.variant_reject_reason`) — a live 31k-word
  dry-run sweep found it flags ~21% of the vocabulary, and a sample of what
  it flagged was mostly genuine rare vocabulary (*haft*, *glaive*,
  *thurible*, *discomfit*, *kickshaw*, *outlawry*), not the junk it was built
  to catch. Edit-distance closeness to a common word doesn't imply a real
  spelling-variant relationship, and cross-language frequency proximity
  can't distinguish a foreign word from an English word that's *also* a word
  in that language (*haft*, *argent*, *rood* are all real English). So the
  word stays active and defined either way; the flag is only ever a queue
  for a person to check via the review webapp.
- **Analogy-relation candidates** (WordNet hypernym/holonym/antonym edges
  and spaCy-parsed definition patterns, `concordance/analogies.py`) — a live
  check found roughly 60–80% of raw WordNet "hypernym" edges for rare or
  abstract words are actually disguised near-synonyms, unfit for a quiz
  question whose entire point is "these are different but related." Every
  candidate edge, from either source, is mandatory-verified by the local LLM
  against both terms' real definitions before it's usable — not an optional
  quality pass, a required gate.

The same principle runs the other direction too: `validity_score`'s
deterministic estimate (`likely-valid` / `uncertain` / `likely-artifact`,
built from Ngram, wordfreq, WordNet/NLTK wordlists, morphology, and a
SymSpell near-neighbor check) never auto-deletes a word either — it's a
prioritized queue (`flagged_undefined = true AND validity_label =
'likely-artifact'`) for your own prune review, because in practice most
currently-flagged words *do* turn out to be OCR misreads, archaic-spelling
variants no modern dictionary carries as a headword, or foreign fragments —
but "most" isn't "all," and this system doesn't get to unilaterally decide
which is which.

## Soft delete, always

Nothing in this codebase hard-deletes a word. Pruning a term via the review
webapp, or a synonym/plural consolidation, or a junk-POS cast-out, all flip
`word.active = false` — never `DELETE`. Downstream features (quizzing, stats,
audio/ngram history) just filter on `active = true`; a pruning decision is
always reversible, and audio/pronunciation/ngram history for a pruned word
stays intact in case it's ever un-pruned.

## Never silently drop — with one deliberate exception

Every cut the pipeline makes is logged, not discarded — the design goal from
day one was that you should never have to wonder where a word went. For most
reject reasons this means one row in `rejected_word` per **(book, lemma)**,
deliberately *not* deduped across books the way `word` is: the same lemma can
be rejected for a genuinely different reason (or a different recurrence
count) in a different book, and collapsing that away would lose real
information about *why* a specific run of the pipeline made the call it did.

**`frequency_floor` is the one exception**, added 2026-08-17 after a
production incident. A word's frequency-floor status is deterministic — it
never varies by book, unlike every other reason — so per-book rows for it
carried zero additional information, only duplication. Measured live before
the fix: 76.3 million `frequency_floor` rows behind just 58,413 distinct
lemmas, a **1,307× duplication factor**, accounting for ~21GB of a 40GB
table. Worse, the verdict-cache query that powers cross-book judge-skipping
(`fetch_known_verdicts` — see below) was pulling all of those duplicate rows
into Python on every single book of a long batch run; at full accumulated
scale that query's memory footprint climbed into the tens of gigabytes,
which is what put the host into swap and eventually cost an OOM-killed
ingest run. `sync_book_results` now skips the `rejected_word` write for
`frequency_floor` specifically, while still counting it in the printed
per-book summary. The lesson generalizes: "log everything" is right *only*
for information that can actually vary; logging a deterministic fact once
per occurrence is duplication wearing a audit trail's clothes.

## Scale generalization — never bake in today's corpus size

The corpus is meant to keep growing, so nothing here should assume today's
size is close to final. This principle has both a good example and a real
counter-example worth being honest about.

**Where it holds**: `book-similarity`/`author-similarity` never store a full
similarity matrix — only each entity's top-k neighbors above a
`min_shared_words` floor, so *storage* stays flat regardless of corpus size.
Word-relatedness (`concordance embed`) is never a precomputed all-pairs
distance matrix either — at 20k+ active words that's already 200M+ pairs and
only gets worse — so both signals are stored as one fixed-size vector per
word and queried on demand via a pgvector HNSW index, turning "who's near
word X" into an O(log N) query instead of an O(N²) precompute. The
cluster-map/similarity-matrix/dendrogram views are honestly labeled "top N,"
not "all," because both the clustering math (an n×n distance matrix plus an
`eigh` call) and the rendering itself (an 11k-leaf dendrogram is illegible
regardless of load time) stop being tractable well before real corpus scale
— a bounded, principled subset stands in for "everything" rather than
pretending to be exhaustive.

**Where it didn't, and had to be fixed**: `book-similarity`/
`author-similarity`'s own internal computation used to build one corpus-wide
`dot[a][b]`/`shared[a][b]` pair of dicts covering every book pair
simultaneously — described at the time as bounded by `max_df_fraction`
(excluding words used by more than 50% of entities from the metric). That
framing was wrong in a way that only showed up at scale: **a fixed fraction
of a growing corpus is a growing absolute number.** At 26,519 books, a word
sitting comfortably under the 50% cutoff still appeared in over 10,000
books, and the pairwise cost per word is O(df²) — one word alone contributed
~52 million book-pair updates. `max_df_fraction` delays the blowup, it does
not bound it. Measured live: ~29GB of Python dict overhead and climbing,
which drove the same class of memory pressure the `frequency_floor` incident
did. The fix (2026-08-17) restructures the computation to process one
book/author at a time — a local `dot_a`/`shared_a` pair built from a word
index and discarded every iteration, so **peak memory** is bounded by one
row's neighborhood instead of the whole corpus. This is explicitly *not* a
complexity fix: total arithmetic is unchanged (still O(Σ df²) over qualifying
words), and is actually somewhat higher than the old combined pass (a shared
pair's contribution gets computed once per side instead of once combined) —
it trades runtime for a predictable memory ceiling, which was the actual
scarce resource. The underlying quadratic growth is still there and will
eventually make this slow again as the corpus keeps growing; it just won't
take the host down when it does. `max_df_fraction` remains a near-lossless
*approximation* (a word's IDF weight is already close to zero at that
document frequency) — it was never actually a performance bound, and the
docs should say so.

## Approximation over exhaustive computation, once exhaustive stops scaling

A recurring shape: replace an exact all-pairs computation with a bounded
approximation once real corpus scale makes the exact version too expensive,
and be explicit that it's now an approximation. IDF-weighted cosine over
shared vocabulary (not raw Jaccard overlap) is the same instinct applied to
correctness rather than just cost — an earlier unweighted join let a word
shared by nearly every book (*the*, *said*, *table*) count the same as a
shared *cangue*, so common words dominated the score; the recorded live
symptom was "every Shakespeare play pulls in nearly the entire corpus as
co-authors." IDF weighting (`ln(N/df)`) fixes the correctness problem
directly rather than working around it with a stopword list. Classical
(Torgerson) MDS for the relatedness maps is the same idea for a different
reason: one deterministic `eigh` call instead of sklearn's iterative SMACOF,
because a relatedness map that mirror-flips or reshuffles between runs is
worse than a "good enough" projection that's stable.

## Absolute scores, not corpus-relative ones, where the number needs to mean something on its own

Fame scores (`author-fame`/`book-fame`) are an ABSOLUTE 1–10 rating, LLM-judged
against a fixed external rubric anchored on real reference figures
(Shakespeare = 10) — deliberately *not* a corpus-relative percentile. A
percentile would drift every time the corpus grows (today's 90th percentile
isn't the same book as next year's), and would say nothing meaningful about
a book in isolation. The tradeoff is real and accepted: this corpus is
mostly public-domain/pulp-heavy, so score mass sits low rather than spread
evenly 1–10 — an honest reflection of how skewed actual cultural fame is,
not a bug to normalize away.

## Personal signal stays personal — why calibration never touches the shared score

`calibrate-difficulty` nudges a word's difficulty *per user*, from that
user's own first quiz exposure to it, and is deliberately never folded back
into the shared `word_difficulty.difficulty` column. With one dominant rater
(this is presently a single-user deployment), response data only ever
reveals that person's own relative gaps — never "true" item difficulty, no
matter how much of it accumulates. A population-level IRT calibration was
considered and rejected for the same reason: it would produce a
well-calibrated *artifact*, not a well-calibrated *estimate*, as long as
there's essentially one rater behind it. `eta`/scale are hand-tuned via
`app_settings`, not fit from data, for the identical reason. Only a word's
*first-ever* quiz exposure counts toward the personal overlay — a later
re-exposure is evidence the person is learning the word, not independent
evidence about a fixed item's difficulty.

## Considered and rejected: Cloudflare Access in front of the SPA

Cloudflare Access was tried as an edge-layer gate in front of the public
deployment, then removed. Access gates by redirecting an unauthenticated
request to a `cloudflareaccess.com` login page — which only works for a full
page navigation. Every API call this single-page app makes is a background
`fetch()`, and a fetch that gets redirected cross-origin fails the browser's
CORS check outright (a bare "Failed to fetch," no usable status code). Access
couldn't gate a single `/api/*` route without breaking the page that calls
it, and it can't gate `/register` at all without defeating the entire point
of invite-link signup. There was nothing left for it to usefully protect: the
app's own session-based `require_admin`/`require_viewer` gate was already the
load-bearing boundary the whole time (verified end-to-end by
`tests/test_auth.py`), so removing Access didn't expose anything it had been
reliably protecting.

## Proper-noun pollution

Every validity authority this pipeline checks — SymSpell, WordNet, wordfreq,
even the local Wiktionary dump to a lesser degree — is itself somewhat
polluted by real names with a large-enough web/corpus footprint to look like
ordinary vocabulary. Confirmed live (2026-08-17): 1,816 active words carry a
tagger-guessed `proper noun` part of speech with no definition at all —
exactly the profile of a name that survived `propernouns.py` (no mid-sentence
capitalization evidence, usually a one-off occurrence) and that nothing else
in the cascade could resolve. Of the fraction already scored by `deepen`'s
validity estimate, only ~4% were correctly flagged `likely-artifact`; the
rest read as `uncertain` or even `likely-valid`, since frequency/morphology
signals can't tell a real name from a real rare word — spelled correctly,
real corpus footprint, no dominant misspelling twin.

**Shipped, small piece**: `ValidityGate._wordnet_instance_only` (step 1.5)
disqualifies a word whose *only* WordNet synsets are instance-hypernyms — a
specific named individual/place (Ahasuerus, Rahab, Coventry), not a
common-noun category — reusing data already loaded, no new sourcing. Auto-
drops when no other authority vouches either; flags for human review (via
the same `variant_flag_reason` mechanism the foreign/misspelling-variant
detector uses) when one does, since that's a genuine collision this signal
alone can't resolve. **Measured impact is small**: checked live against the
current 621-word undefined-proper-noun bucket, it resolves only 2 of them.
WordNet's instance entries lean toward well-known geographic/historical
entities (continents, geologic eras, famous cities) — confirmed directly
that `ahasuerus`/`oisin`/`rahab` have *zero* WordNet synsets at all, not
instance-only ones, so this fix doesn't touch the exact cases the original
2026-07-12 audit found. It's correct and free, just not where the real
leverage is.

**Shipped, the real leverage (2026-08-17)**: a curated names/places
gazetteer (`concordance/gazetteer.py`, `db.gazetteer_name`) — US Census
2010 surnames (≥100 occurrences, ~162k), NLTK's census-derived given names
(~7.6k), and GeoNames `cities1000` populated places (~110k single-word
entries) — loaded once (`concordance load-gazetteer`, not part of
`maintain`, same one-time/occasional shape as `load-taxonomy`) and checked
in the same step 1.5 slot as the WordNet-instance signal, with identical
collision handling: auto-drop only when SymSpell/NLTK words don't
independently vouch, flag for review otherwise. Measured live against the
same 621-word bucket: **143 matches (22.9%)**, 122 with no competing vouch
at all (safe auto-drop) and 21 genuine collisions (correctly flagged, not
dropped — e.g. `sackman`, and `godel` again). Manually sampled 30 of the
122 clean-drops: every one looks like a genuine surname or obscure place
name, no plausible false positives found. Confirmed the collision guard
holds even for words this project explicitly wants to keep: `house` and
`armiger` (the audit's own canonical "don't lose this" example) are both
also real Census surnames, and both correctly stayed `KEEP` with only a
review flag.

`ValidityGate` takes an optional `gazetteer_names: frozenset[str]`,
loaded once per batch run (`db.fetch_gazetteer_names`, mirroring how the
~9GB judge model / spaCy / SymSpell / WordNet are all loaded once and
reused across every book) rather than per book — a repeat of the
`fetch_known_verdicts` mistake (a heavy, unbounded query re-run on every
single book of a multi-thousand-book batch) was the one real risk in
wiring this in, and `pipeline.process()` only fetches it at all when no
pre-built `gate` was already handed in.
