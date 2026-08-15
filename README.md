# Concordance

Extract interesting vocabulary from books you read (EPUB, text PDF, `.txt`) using
a **local** LLM — no paid API. Rare words are surfaced, common ones and junk are
filtered, and the result lands straight in Postgres, ready to review in the
[web app](#web-app-webapp).

The design is deliberately **keep-biased**: a genuine rarity should survive to
your review even at the cost of a little noise, never the reverse. See the
requirements & architecture spec for the full rationale.

## Pipeline

```
extract → clean → tokenize → frequency-floor → cross-book verdict cache
        → strip-proper-nouns → validity-gate → LLM-judge → dictionary-lookup
        → Postgres (ingest) → maintain's fill-definitions (refill/deepen)
```

- **frequency floor** — a stop-word-style cut of common words (never a rarity *ceiling*)
- **cross-book verdict cache** — a lemma already kept/pruned/judge-rejected in an earlier book is pre-marked from `word`/`rejected_word` and never re-judged: the LLM judge's input is purely `(lemma, frequency band)`, so at temp 0 its verdict on a given lemma is always the same. This is what keeps per-book judge time from scaling with corpus size — cost tracks *distinct new rare words*, which saturates fast on a shared-vocabulary corpus.
- **validity gate** — multi-source, keep-biased. A word is a real word if *any* authority vouches for it — the local `vocab.wiktionary` DB dump (~500k terms, checked first because it's free and carries no "Proper noun" POS to get confused by), then the SymSpell 82k wordlist, **WordNet**, or **NLTK's 234k dictionary corpus** (which carries the archaic vocabulary — *destrier, bartizan, cangue* — that trips up single-dictionary checks). A foreign-language-context check runs early too. Only then is misspelling considered, by *relative* near-neighbor frequency (with a recurrence escape hatch — a "misspelling" that keeps showing up is probably a real coinage). NLTK's `wordnet` and `words` data download automatically on first run.
- **LLM judge** — a local model decides what's worth learning (stubbed until you point it at a model). To keep a weak local model honest it emits a *minimal* per-word verdict (`{"w","k"}`, no free-text reason) so it doesn't truncate its output and silently drop words; any word it omits is re-queried for up to three passes before the keep-biased fallback, so junk can't flood the list by omission. A corpus frequency hint (common / uncommon / rare) steadies its rarity sense but is never a hard cut. Frequency alone can't do this job — *tendril* is rarer than *refectory* yet everyone knows it — which is exactly why the judgment is the model's, not the floor's.
- **dictionary lookup** — one shared cascade (`concordance/resolve.py`), used identically by `maintain`'s `fill-definitions` and the standalone `refill`/`deepen`/`lookup_word.py`: local Wiktionary dump → Free Dictionary API → Wiktionary online → **the OED reference dataset** (`concordance/oed/definitions.py`, no network/quota — degrades to a no-op if `oed-ingest` was never run) → Merriam-Webster's Collegiate API → Wordnik (paced internally to its 5 req/min free-tier cap) → yourdictionary.com → web-search + grounded local-LLM extraction as the true last resort. The OED tier's text was never editorially curated the way the other sources are (OCR'd from scanned volumes; its sense-splitter is a known first pass, not a calibrated parser — see the module docstring), so a hit there also marks the word for human review via `variant_flag_reason` rather than being trusted silently. `ingest`'s own inline cascade stops one tier short — **LOCAL → FREE → OED → Merriam-Webster's Collegiate API** (`concordance/mw.py`, needs `MW_DICTIONARY_API_KEY`) — skipping Wordnik/yourdictionary/web-search as too slow/rate-capped for ingest throughput (OED stays in: local/free, same as LOCAL); whatever's still blank afterward gets the full remaining depth from `maintain`/`refill`/`deepen` (below). Enrichment runs on a worker pool (`Config.enrichment_workers`, default 4) rather than one word at a time — this stage's per-book cost is both the dominant and most variable part of `ingest` (measured up to 310s/426 words serial on a real book) — and MW's own on-disk cache/quota lock guards only the local bookkeeping, never the network call itself, so concurrent MW lookups genuinely overlap instead of queuing behind each other's retry chains. Every HTTP call retries with exponential backoff; `Retry-After` is honoured on a real 429 (capped at 10s) but not on a bare 5xx — measured live, a single transient 502 carrying a `Retry-After: 60` header cost 60s of one word's lookup in an otherwise ~1s batch, confirming the header means nothing coming from a gateway hiccup the way it does from an actual rate limiter.
- **review** — prune too-common/easy terms afterward in the [web app](#web-app-webapp) (a soft delete — `word.active = false` — nothing is destroyed)
- nothing is ever silently dropped — every cut is logged to `rejected_word`, one row per (book, lemma)

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm
```

Requires a live `DATABASE_URL` (env or a git-ignored `.env`):

```
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

The validity gate and enrichment both check the local `vocab.wiktionary` dump
(~500k terms) first, before any of the offline/network fallbacks. It's the
cheapest and, since that dump carries no "Proper noun" POS category at all,
the cleanest authority available: unlike SymSpell/WordNet/wordfreq (all
frequency-derived from general web text, so polluted by real names with any
web footprint), membership alone means "this isn't a name."

## Ingest a book (`ingest`)

```bash
concordance ingest "some book.epub"
concordance ingest "some book.epub" --schema concordance
```

Runs the full extract → filter → judge → enrich pipeline and writes straight
into Postgres — no CSV, no hand-edit, no promotion step. Kept words upsert
into `word`/`word_book`; everything the pipeline dropped goes into
**`rejected_word`** — one row per **(book, lemma)**, deliberately *not*
deduped across books the way `word` is, since the same lemma can be rejected
for a different reason (or recurrence count) in a different book. Nothing is
silently lost; you query the DB instead of opening a CSV. Idempotent —
re-running the same book updates both tables in place rather than duplicating
rows, and never clobbers a field (definition, IPA, etymology, ...) that
already has content with a blank value from a re-run.

Review happens afterward in the **[web app](#web-app-webapp)**: its
**Accepted** tab lets you prune too-common/easy terms (a one-click soft
delete — `active = false`); its **Rejected** tab lets you browse what the
pipeline dropped and "Add" one back if it dropped something worth keeping
(flagged `rescued_from_reject` so the rescue stays traceable).

A word already marked pruned (`active = false`) via the web app, or
judge-rejected in an earlier book, is recognized before it ever reaches the
floor/validity gate/judge — see the cross-book verdict cache above — so
review decisions are never silently re-litigated or wasted as repeat LLM
calls.

**Batch mode — process everything in `incoming/`:**

```bash
concordance ingest              # every .epub/.pdf/.txt in incoming/
```

Name files `[Title] -- [Author].epub` (e.g. `Ulysses -- Joyce, James.txt`) and
the title/author populate `book.title`/`book.author` directly — no delimiter
found just uses the whole filename as the title with a blank author, rather
than erroring out. Each file is moved into `archive/` after processing
(`--no-archive` to leave them in place); explicit single-file mode
(`concordance ingest some/book.epub`) still archives next to the source file
rather than to the top-level `archive/`. The judge model, spaCy, and the
validity gate's dictionaries are loaded **once** for the whole batch (not
once per book), so a long batch pays that cost a single time.

Flags: `--min-zipf` (frequency floor; higher = rarer only), `--limit`,
`--no-lookup`, `--model`, `--stub`, `--schema`, `--database-url`,
`--no-archive` (batch mode only).

## Post-ingest maintenance (`maintain`)

Everything below this point — backfilling, enrichment/scoring, pronunciation
prep, and embeddings — runs in one dependency-ordered pass instead of twelve
commands to remember and re-order by hand:

```bash
concordance maintain   # fill-definitions -> classify -> normalize-pos -> ngram
                        # -> archaic -> difficulty -> quizdef -> quizzable
                        # -> calibrate-difficulty -> wordnik-pron -> ipa
                        # -> embed -> backfill-analogies
```

`book-similarity`/`book-clustering`/`author-similarity`/`author-clustering`
are NOT part of this chain (see "Vocabulary relatedness" below) — same
reasoning that already kept `book-fame`/`author-fame` standalone: those four
are full-corpus recomputes with no only-missing gate (corpus-wide IDF
weights shift whenever any book's vocabulary membership changes), so folding
one into a chain whose whole point is "catch up on just the new batch" would
mean `maintain` could never have a cheap mode.

Every remaining step runs incrementally (only-missing / blank-only /
not-yet-embedded), so a re-run after everything's caught up is fast — it
only touches the newest batch's words. The **first** run against a corpus
with a real backlog is not:
`classify`, `quizdef`, and `backfill-analogies` all load a local LLM and call
it per word (or per candidate relation), so catching up a few thousand words
is likely to take hours. That cost is paid once. Use `--skip-fill-definitions`,
`--skip-classify`, `--skip-quizdef`, `--skip-analogies`, etc.
(one flag per step) to defer the slow ones to run separately/overnight
instead of blocking on them inline; `--limit` caps words processed per step,
useful for chunking a large backlog into resumable pieces
(`compute_quiz_definitions`, for one, only commits at the very end of a run —
an unlimited invocation against tens of thousands of words risks losing hours
of LLM work to a single interruption; looping `--limit 3000` and relying on
only-missing to pick up where the last chunk left off is the safer shape for
a big catch-up). `fill-definitions` also honors `--recheck-after-days`
(default 14): a word whose last resolution attempt failed recently is
skipped rather than re-ground through Wordnik/web-search again on every
single `maintain` run. `load-taxonomy` and `train-fasttext` are deliberately
excluded — both are one-time/occasional setup, not per-batch maintenance —
and so are the Commons/Azure audio steps, since Commons rate-limits hard and
is meant to run for hours unattended on its own (see "Pronunciation audio"
below).

**Definitions can change after these have already run** — the same lemma
reappearing in a later book can resolve to a different dictionary sense, and
`sync_book_results`/`sync_master` will overwrite an existing definition with
the new one. Whenever that happens, whatever was computed from the *old* text
and only ever revisited via an only-missing check — `quiz_definition`, USAS
categories, the definition embedding — gets invalidated (cleared) right there
in the same upsert, so the next `maintain` run regenerates it from the
current text instead of silently going stale. `archaic`/`difficulty`/
`quizzable` don't need this: all three always recompute every row
unconditionally, so they self-correct on the next run with no help.

## Backfilling definitions (`refill`, `deepen`)

A word can be *kept* (a real word, worth learning) without ever getting a
definition — `ingest`'s enrichment sources sometimes miss genuinely rare or
archaic vocabulary. Rather than silently sitting blank forever, every such
word is durably marked `word.flagged_undefined` (+ `_at`) the moment it's
accepted with no definition — **sticky by design**: the marker is never
cleared, even once a definition is later found, because the point is a
permanent "this one needed a second look" audit trail for your own manual
validity review, not a live status flag.

`maintain`'s `fill-definitions` step runs both of these in one pass per word
(cheap sources first, falling through to the deeper ones without ever
re-entering the cascade from scratch). `refill`/`deepen` below remain as
separate, independent commands — useful for running just the cheap pass, or
re-running the deep pass on demand outside a full `maintain`.

```bash
concordance refill              # cheap sources, same ones ingest already tried
concordance deepen              # + Wordnik/yourdictionary/web-search + a validity estimate
concordance deepen --no-web     # skip the web-search/LLM tier (faster, no model load)
```

- **`refill`** re-tries the local Wiktionary dump and the free online
  dictionaries (Free Dictionary API, Wiktionary) for every word whose
  `definition` is still blank — useful when the miss was transient (a rate
  limit, a network blip) rather than the word genuinely being undefinable.
- **`deepen`** runs after `refill` and reaches further: **the OED reference
  dataset** (`--oed-schema`, default `oed`; no-op if `oed-ingest` was never
  run — a hit also flags the word for human review, see "dictionary lookup"
  above), **Merriam-Webster**, **Wordnik** (Century
  Dictionary + Webster's, which carry archaic vocabulary — needs a free
  `WORDNIK_API_KEY` in `.env`, falls back to yourdictionary-only without it),
  **yourdictionary.com**, and (default on — pass `--no-web` to skip) **web
  search + grounded local-LLM extraction** as the true last resort. This last
  tier does almost all of the real work by the time a word reaches it: every
  faster/cheaper source has already been tried and missed, so real-scale
  testing found nearly all of a deepen run's actual yield comes from here —
  it just costs a local 14B model load and is far slower per word than the
  rest. Whatever *still* can't be defined gets a
  deterministic, explainable **validity estimate** written to `word.validity_label`
  (`likely-valid` / `uncertain` / `likely-artifact`), `validity_score` (0–1),
  `validity_notes`, and `suggested_correction` — signals are Google Books
  Ngram, wordfreq, WordNet/NLTK wordlists, morphology, and a SymSpell
  near-neighbour check, the same scoring used for the CSV-era `<book>.undefined.csv`
  report. **In practice, most currently-flagged words score `likely-artifact`**
  — OCR misreads, archaic-spelling variants no modern dictionary carries as a
  headword, and foreign-language fragments that slipped past the keep-biased
  validity gate on some other authority's say-so. Cross-reference
  `flagged_undefined = true AND validity_label = 'likely-artifact'` for your
  prune review queue.
- Neither command ever overwrites an existing definition — both only touch
  rows where `definition` is still blank.

**Why a word count doesn't match `attempted`**: `fill-definitions`
(`maintain`'s step, backed by `db.fill_definitions`) does NOT candidate every
blank-definition word on every run — it also requires
`validity_checked_at IS NULL OR validity_checked_at < now() - recheck_after_days`
(default 14 days, `maintain --recheck-after-days N` to change it). A word
already run through the full cascade recently (including web-search) and
still blank got a fresh `validity_checked_at` stamp from that failed
attempt, so it's skipped for the rest of the cooldown window rather than
re-ground through Wordnik/web-search again immediately — without this,
every `maintain` run would re-attempt the entire permanently-undefined tail
from scratch, forever, once web-search became the default (see below). If
`SELECT count(*) FROM word WHERE definition = ''` is bigger than what a run
just reported as `attempted`, this cooldown is why — check
`validity_checked_at` on the difference. **The standalone `concordance
deepen` bypasses the cooldown entirely** (`recheck_after_days=0`) — it's the
explicit, deliberate "retry the undefined tail right now regardless of when
it was last checked" command; the cooldown only throttles `maintain`'s
*automatic* re-grinding, not a one-off human-invoked run.

A **separate** human-review flag — `word.variant_flag_reason`/
`variant_flag_note`/`variant_flagged_at`, written by every one of `ingest`/
`refill`/`deepen`/`fill-definitions` — marks a word that a source
successfully defined but that looks like a foreign word or an archaic/OCR
spelling of a common modern word (e.g. `acte`, an archaic-spelling
`assunder`). This is deliberately NOT an auto-reject: a real-scale sweep of
the existing vocabulary found the detector's false-positive rate too high to
trust unattended (real words like `haft`, `glaive`, `thurible` got flagged
too, and even naturalized English loanwords like `dénouement`/`matinée`/
`séance` — spelled with their original accent, same category as café/résumé
— got caught by the foreign-language check). The word stays fully active and
defined either way, just marked for a person to glance at:

```sql
SELECT lemma, variant_flag_reason, variant_flag_note
FROM word WHERE variant_flag_reason IS NOT NULL ORDER BY lemma;
```

`scripts/sweep_variant_rejects.py` (dry-run by default, `--apply` to write
the flags) runs the same check retroactively against words already active
before the flag existed — cross-checking each flagged word against the same
curated authorities (local Wiktionary dump, WordNet, NLTK's words corpus)
`validity.py`'s own ingest-time gate uses is a cheap, deterministic way to
clear most false positives before a human ever needs to look (verified: this
cleared ~95% of a 6,499-word retroactive flag pass on its own). What's left
after that genuinely needs a human read — a word both edit-distance-close to
a common word AND cross-language-frequency-close to another language isn't
reliably resolvable by any signal this project has; see git history around
the Phase 5 commits for the full false-positive analysis if extending this.

Both commands accept `--schema`, `--limit`, `--database-url`.

## Merriam-Webster word lookup (`scripts/lookup_mw.py`)

A stand-alone, network-only CLI — like `lookup_word.py`, but a single source
(Merriam-Webster) instead of the shared cascade, and prints full entries
(POS, pronunciation + audio, definitions, etymology, first known use) to
stdout rather than feeding the ingest pipeline:

```bash
python scripts/lookup_mw.py concordance
python scripts/lookup_mw.py concordance run --headless
python scripts/lookup_mw.py cangue --no-fallback
```

Two tiers, in order:

1. **The official Collegiate Dictionary API** (`concordance/mw.py`) — needs a
   free `MW_DICTIONARY_API_KEY` in `.env`. The free tier caps out at 1000
   queries/day, so every response (a real hit **and** a confirmed "no exact
   match") is cached on disk keyed by word — looking up the same word again,
   even in a later run, never costs another call. A daily usage counter
   refuses new API calls once the count reaches the cap rather than erroring
   partway through a batch. Only a genuine 200-with-parseable-JSON response
   gets cached; a bad key, a 5xx, or a dead network is never cached, so a
   transient problem can't permanently mark a word "unresolvable."
2. **A Playwright scrape of the live site** (`concordance/mw_scrape.py`) —
   tried only when the API comes back empty (a real miss, or the daily cap),
   for words the site has that the Collegiate API doesn't. The site sits
   behind Cloudflare's managed bot challenge (confirmed: a plain
   `requests`/curl GET gets a 403 regardless of headers — active bot
   mitigation, not a `robots.txt` restriction, which does allow
   `/dictionary/*`). A real browser clears the challenge on its own after a
   few seconds; this uses a **persistent** browser profile
   (`.cache/mw_browser_profile/`) so the cleared cookie survives across runs
   — the challenge only ever needs solving again if it expires or the site
   re-flags the profile. Defaults to headed (`--headless` to force
   headless, which is more likely to get challenged); on a brand-new profile
   you may need to solve an interactive checkbox once, by hand, the first
   time. One browser is opened lazily and reused across an entire word list
   rather than relaunched per word. Requires `pip install concordance[scrape]`
   + `playwright install chromium`.

### Bulk MW backfill (`mw-backfill`)

A DB-writing batch job built on the same two-tier lookup above, for exactly
the words the *existing* definition cascade (Free Dictionary/Wiktionary/
Wordnik/yourdictionary/web-search — see "Backfilling definitions" above)
couldn't resolve, where MW's own Collegiate coverage sometimes succeeds
anyway:

```bash
concordance mw-backfill                    # API + scrape fallback, 1000/day cap
concordance mw-backfill --no-scrape        # API tier only
concordance mw-backfill --limit 50
```

- **Candidates**: active, still undefined, and not already scored
  `likely-artifact` (null/`uncertain`/`likely-valid` only — a word already
  written off as an artifact isn't worth spending API quota to double-check).
- **`word.mw_checked_at`** is a permanent "already attempted" marker (hit OR
  miss, same sticky convention as `flagged_undefined`) — a repeated daily run
  just keeps working through whatever's left, never re-spending quota on a
  word already tried. `word.first_known_use` is new alongside it (MW's own
  field; nothing else here has a use for it).
- Only fills `definition`/`part_of_speech`/`etymology`/`definition_source`/
  `first_known_use`, never overwriting a non-blank existing value — and
  **never `word.ipa`**: MW's pronunciation is its own proprietary respelling,
  not true IPA (no `ahd.py`-style converter exists for it), and `word.ipa` is
  trusted elsewhere (Azure TTS synthesis) to actually contain IPA.
- **Exact-match guard**: MW's API does fuzzy full-text search and will
  happily return a same-ballpark idiom for a query that isn't a real headword
  at all — confirmed on live data, querying "atune" matched the idiom "sing a
  different tune", "aglance" matched "at a glance". `mw.exact_matches` only
  accepts an entry whose own headword literally is the queried word before
  anything gets written — fine for a human browsing `lookup_mw.py`'s fuller
  results, but this automated writer needs it so it never records an
  unrelated idiom's definition as if it were the word's own.
- **Extends the proper-noun/foreign-language cast-out gate** for MW's own
  category vocabulary: `model.JUNK_POS_REASON` now also recognizes MW's
  "biographical name"/"geographical name"/"trademark" labels (real leaked
  proper nouns caught live — a former Canadian PM, a French colonial
  territory, a drug trademark, none of which the existing symbol/proper-noun
  check knew about), and `mw.is_foreign_pos` catches MW's capitalized
  "`<Language>` noun/verb/adjective/adverb" loanword tag (e.g. "Swahili
  noun") — checked against the raw string before `normalize_pos` lowercases
  away the exact signal that makes it recognizable.
- Stops the **whole run** (not just the API tier) once the 1000/day cap is
  hit, leaving the remainder for tomorrow rather than falling through to an
  unbounded scrape-only tail. `--scrape-timeout-ms` (default 10s, half
  `lookup_mw.py`'s own 20s) trims the fallback's per-word wait, since a
  genuine miss on the live site always costs the full page-load timeout and
  most candidates reaching this tier already failed every other source too.

## Definition-quality cleanup (`dedupe-plurals`, `expand-synonyms`)

A dictionary source sometimes resolves a word to a bare cross-reference —
"warrs" → "plural of warr", "ephebus" → "Synonym of ephebe" — instead of real
content. Both commands find every live case and fix it, idempotently and
safely re-runnable (new cross-references introduced by future books get
picked up on the next run), but with **opposite** fixes, because a plural
and a synonym aren't the same kind of redundancy:

```bash
concordance dedupe-plurals      # consolidate a plural into its singular
concordance expand-synonyms     # give a synonym its own real definition
```

- **`dedupe-plurals`** — a plural form isn't separate vocabulary; it's the
  *same* word in a different grammatical form, so `quizdef.quizzable()`
  already excludes "plural of X" definitions from quizzing
  (`_VARIANT_RE`). The fix here is consolidation: resolve the singular X
  (reusing it if already active, creating and defining it via the same
  cascade every other definition path uses if not) and soft-delete the
  plural (`active = false`, reversible via the review webapp, never a hard
  delete — every removal in this codebase works this way). A singular that
  exists but is currently inactive is **always** left untouched, never
  reactivated — checked against real data before building this: every such
  case already had a real definition, meaning "inactive" is near-certain
  evidence of an earlier deliberate decision (a human prune, or a justified
  automated cast-out) that a plural merely existing isn't good reason to
  override.
- **`expand-synonyms`** — a synonym *is* separate vocabulary (two different
  surface words that happen to share a meaning), so unlike a plural it's
  never deleted. Unlike "plural of X", "Synonym of X" definitions were
  never excluded from quizzing either (`_VARIANT_RE` never had "synonym" in
  its word list) — a real data-quality gap, not just a missed quizzability
  case. The fix: replace the cross-reference with real content instead.
  Some sources already embed a real gloss right in the cross-reference
  ("Synonym of nithing (“a coward, a dastard; a wretch”)") — extracted
  directly, no lookup needed. Otherwise the target's own definition is
  reused (or freshly resolved, creating the target as its own word if it
  doesn't exist) — same conservative "never touch an inactive target" rule
  as `dedupe-plurals`, and never used to "upgrade" a definition if the
  target's own resolution turns out to be a symbol/proper-noun sense.

Both default to `--web` (full cascade depth, including web-search + local
LLM for anything that needs a fresh resolution — pass `--no-web` to stay on
the free/keyless tiers only) and accept `--schema`, `--limit`,
`--database-url`. Whenever a word's own definition text changes, its stale
`quiz_definition`/USAS categories/definition embedding are invalidated so
the next `maintain` run recomputes them from the new text — the same fix
`sync_book_results` already applies for a re-ingested word whose sense
changed, needed here too since this writes `word.definition` directly.

## Definition cross-linking (`link-definitions`)

```bash
concordance link-definitions    # full recompute every run, not incremental
```

Cross-links a word's definition to any OTHER **active** word it mentions
(lemma-aware, spaCy), so the review webapp can render a real clickable link
instead of inert prose — powers `LinkedDefinition.jsx` on the word-detail
page. Always a full recompute rather than an only-missing pass — cheap
enough that it's safe to re-run any time a definition changes (`refill`/
`deepen`/`mw-backfill`/`expand-synonyms` all rewrite `word.definition`, and
none of them know to invalidate this on their own). Written to
`word_definition_link`, joined back with an `active` guard on the target so
a link whose target was later pruned never renders as a dead link.
Standalone only — not part of `maintain`.

## Enrichment & scoring (`classify`, `archaic`, `ngram`, `difficulty`, `quizdef`, `quizzable`, `calibrate-difficulty`)

A further pass of DB-only commands (no book/model pipeline; each just reads
and updates rows in the schema `ingest` populated), meant to run in this
order after words exist:

```bash
concordance load-taxonomy   # once: load the USAS category tables
concordance classify        # tag every word with 1-3 USAS domain codes
concordance normalize-pos   # fold part_of_speech into one clean vocabulary
concordance ngram           # cache Google Books Ngram rarity/recency per word
concordance archaic         # set current/dated/archaic/obsolete + confidence
concordance difficulty      # 0-100 ex-ante difficulty scalar + factor breakdown
concordance quizdef         # quiz-safe definitions (rewrite ones that leak the word)
concordance quizzable       # flag variant/inferable-derivative words as unquizzable
concordance calibrate-difficulty  # per-user difficulty nudge from quiz response data
```

- **`classify`** — assigns each word 1-3 USAS category codes (word + POS +
  definition + sentence), using the WordNet-Domains mapping as a candidate
  hint the model prunes/confirms against context rather than a hard seed.
  `--only-missing` / `--batch` to backfill incrementally. Commits every
  `--commit-every` words (default 200), not once at the end — this is
  often the single longest-running `maintain` step (a full backlog is an
  hours-to-days local-LLM pass, one word at a time), so a crash mid-run
  loses at most one partial chunk instead of every word classified so far.
  Paired with `--only-missing`'s own re-select-what's-still-missing query,
  a killed and restarted run resumes close to where it left off rather
  than reclassifying from scratch.
- **`archaic`** — an ordinal (current < dated < archaic < obsolete) with a
  0-1 confidence: a register label in the definition or the Wiktionary dump is
  high-confidence; a Google-Books recency decline alone is real but noisy
  (can't distinguish "faded" from "always uncommon"), so it's low-confidence
  and queued for later review rather than trusted outright.
  Needs `ngram` to have run first.
- **`ngram`** — fetches + caches peak/recent frequency and recency ratio per
  word from Google Books Ngram; feeds both `archaic` and `difficulty`.
- **`difficulty`** — blends rarity (dominant), archaic confidence, USAS domain
  specificity, and morphological transparency into a single 0-100 scalar,
  storing the factor breakdown alongside it: a principled ex-ante estimate,
  shared across every user. `calibrate-difficulty` (below) is the
  once-quiz-data-exists follow-on, but it deliberately stays a per-user
  overlay rather than folding back into this column.
- **`quizdef`** — ~37% of definitions leak the target word's root ("audaciously"
  → "in an audacious manner"), making recall quizzing trivial; this builds a
  separate `quiz_definition` per word — passed through as-is if already clean,
  LLM-paraphrased (then machine-verified leak-free) if not, redacted as a last
  resort.
- **`quizzable`** — flags a word unquizzable for any of three reasons: its
  only difference from an already-known base form is grammatical (plurals,
  inflections); it's a transparently inferable derivative of a common root
  ("reveller" ← "revel"); or the definition actually served (`quiz_definition`
  if set, else the raw definition) still leaks a near-identical word even
  after `quizdef`'s own pass — a stem match, a known-prefix derivative, or a
  close spelling variant like "codpieced" defined via "codpiece" — so quizzing
  never hands the answer away or wastes a card on something not actually new.
  Always recomputes every word on every run (no `--only-missing`), which
  matters here specifically: it's what lets a new detection rule apply
  retroactively to the whole corpus for free, without re-running `quizdef`'s
  costly LLM rewrite pass.
- **`calibrate-difficulty`** — a per-`(user, word)` nudge to the ex-ante
  `difficulty` score, from that user's own **first** quiz exposure to the
  word (`concordance/calibration.py`): a fixed-guessing-floor Rasch-style
  update (`b_new = b0 - eta*(y - P0)`), using the question's actual assembled
  guessing floor (option/set count, not nominal request config) so an easy
  4-choice MC miss moves the needle more than a coin-flip true/false miss.
  Deliberately **not** a population-level IRT calibration and never written
  back into the shared `word_difficulty.difficulty` column — with one
  dominant rater (this is a single-user deployment), response data only
  ever reveals that person's own relative gaps, never "true" item
  difficulty, no matter how much of it accumulates. Stored in
  `word_personal_difficulty` and consumed only by quiz word **selection**
  (`difficulty_min`/`difficulty_max` prefer it over the shared score via
  `COALESCE`) — every other consumer of `word_difficulty.difficulty` is
  unaffected. `eta`/scale are hand-tuned via `app_settings`
  (`calibration_eta`/`calibration_scale`), not fit, for the same
  single-rater reason. Only a word's first-ever quiz exposure counts; a
  later re-exposure is evidence the person is *learning* it, not
  independent evidence about a fixed item's difficulty.

`normalize-pos`/`archaic`/`difficulty`/`quizzable` all accept `--limit` for
chunking a large backlog, but — unlike `refill`/`fill-definitions`'
only-missing gating — they always **recompute every row in scope**, capped
by the limit, not just rows missing a value: all four read mutable upstream
columns (definition text, ngram trend, USAS domain, quiz_definition) with no
separate signal to gate a re-check on, so an only-missing pass here would
silently freeze a word's score the first time it computed and never notice
if the underlying data later changed. They're cheap, pure-local computation
(string/regex/wordfreq, no network/LLM), so recomputing everything on every
`maintain` run is a fast, not-batched-for-performance choice — `--limit`
exists purely for interface consistency with the slower steps.

### Pronunciation audio (`wordnik-pron`, `ipa`, `oed-ipa`, `commons-search`, `commons-download`, `audio`, `audio-guess`)

Real human recordings where they exist, IPA-guided synthesis otherwise —
never a blind spelling-to-speech guess unless nothing else is available:

```bash
concordance wordnik-pron      # fetch raw Wordnik transcriptions (ARPAbet/AHD-5/IPA)
concordance ipa               # backfill+validate word.ipa from kaikki, then Wordnik, then local Wiktionary
concordance oed-ipa           # backfill word.ipa from the oed schema's verified pronunciations
concordance commons-search    # find real Commons recordings kaikki's dump missed
concordance commons-download  # download the recordings commons-search confirmed
concordance audio             # Commons recording if present, else Azure IPA-guided TTS
concordance audio-guess       # last resort: Azure guesses from spelling alone
```

`wordnik-pron` and `ipa` are both part of the `maintain` chain above — rerun
`maintain` (or just `ipa`) before `audio`, since synthesis quality depends on
the transcription it's given. `wordnik-pron` is rate-limited (~1 word/several
seconds on the free tier) and `ipa`'s primary source is a 2.7GB dump scan, so
both stay batch passes rather than per-word ingest-time lookups.
`commons-search`/`commons-download`/`audio` are deliberately separate
commands rather than folded into `maintain`: Commons rate-limits hard and is
meant to run for hours unattended, which would starve every other step if
interleaved. `audio-guess` results are tagged `source='azure_guess'` (vs.
`'azure'` for IPA-guided) so the app can flag them as unverified.

`oed-ipa` is the one place the OED reference dictionary (below) feeds back
into the main vocabulary pipeline — it draws on the oed schema's
double-pass-verified `pronunciation_ipa`, only filling a word's `ipa` when
it's currently empty/invalid and every oed entry for that headword agrees
(a homograph like "bay"/"fleet"/"back" can have several entries; a genuine
disagreement is skipped rather than guessed at, since oed's `part_of_speech`
field is unparsed OCR abbreviation soup, not clean enough to pick the "right"
one). Standalone like `wordnik-pron`/`commons-download` — OED coverage grows
with each future volume ingested, so this is meant to be re-run periodically,
not folded into `maintain`. No local LLM or embedding model, so it's safe to
run alongside GPU-bound steps like `author-fame`/`book-fame`/`classify`.

Every word backfilled from `oed-ipa` is tagged `ipa_source='oed'`, which
`audio` uses to pick a matching British voice (`en-GB-SoniaNeural`) instead
of the default US one (`en-US-AvaNeural`) — OED is a British dictionary, and
every other IPA source here is deliberately US-biased (kaikki's lookup
prefers US-tagged entries, local Wiktionary's column is literally
`us_pronunciation`, Wordnik's ARPAbet/AHD-5 converters are both US phoneme
systems), so feeding OED's RP transcriptions to the US voice would produce
wrong/unnatural audio. The dialect switch also flips how an optional/
dialectal sound in parentheses is handled: kept for the US voice (post-vocalic
r is always pronounced), dropped for the UK voice (OED's `(r)` marks an RP
linking r that's pronounced only in connected speech, not citation form —
keeping it would be a rhotic mispronunciation backwards from what the
notation means).

## Semantic distance (`train-fasttext`, `embed`)

Two independent per-word vectors, for visualizing word relationships and — a
later feature this doesn't build — generating quiz distractors. Neither is a
precomputed all-pairs distance matrix (already 200M+ pairs at 20k+ words, and
only growing); both are queried on demand via a pgvector HNSW index, so
finding a word's nearest neighbors is O(log N), not O(N²).

```bash
concordance train-fasttext          # once (or after a big ingest batch): train on archive/
concordance embed                   # definition_vector for words missing one
concordance embed --signal fasttext # fasttext_vector instead (needs the trained model)
concordance embed --signal both     # both in one pass
```

- **Definition embedding** (`sentence-transformers`, `BAAI/bge-small-en-v1.5`)
  embeds each word's dictionary *gloss* — modern English text — rather than
  the rare headword itself, falling back to `synonyms` then the book
  example `sentence` when a definition is missing. This is what makes rare/
  archaic vocabulary tractable at all: the word is rare, its definition
  usually isn't. In practice this reaches ~100% coverage (the `sentence`
  fallback catches almost everything a real definition doesn't).
- **FastText subword vectors** are trained from scratch on this project's own
  `archive/` corpus via `train-fasttext` (not a generic pretrained binary),
  so its character n-grams are learned from the same archaic/literary
  English these words come from. Because it composes a vector from
  subwords, it produces one for *every* lemma regardless of definition
  coverage — genuinely 100%, including words no dictionary could define.
- These capture different things on purpose — meaning vs. spelling — and are
  stored/queried independently (`word_embedding.definition_vector` /
  `.fasttext_vector`), not fused into one score. `train-fasttext` is a
  holistic pass (must see the whole corpus at once — rerun periodically as
  the archive grows); `embed` is the familiar incremental maintenance pass.

The review webapp's backend exposes this as `/api/words/search` (word
picker) and `/api/words/{id}/neighbors` (`signal=definition|fasttext`, with
optional POS/quizzable/difficulty-band/USAS-domain filters and synonym
exclusion) — query infrastructure for a future visualization UI and future
distractor generation, not those features themselves.

## Vocabulary relatedness (`book-similarity`, `book-clustering`, `author-similarity`, `author-clustering`)

Which books and authors are related to each other by **literal shared
vocabulary** — a different axis from the definition/FastText embeddings
above, which measure *meaning* similarity between individual words. This is
lexical bag-of-words cosine similarity, IDF-weighted so common words don't
dominate the score the way raw Jaccard overlap would (a bug already on
record in `browse.py`'s own comments: an earlier unweighted join made
"every Shakespeare play pull in nearly the entire corpus as co-authors").

```bash
concordance book-similarity      # each book's top-k most vocabulary-related books
concordance book-clustering      # hierarchical clustering + 2D map over the top-N books
concordance author-similarity    # same, one level up, for authors
concordance author-clustering    # hierarchical clustering + 2D map over the top-N authors

concordance relate               # convenience wrapper: runs all four above with default
                                  # options, in the order listed (--skip-book-similarity /
                                  # --skip-book-clustering / --skip-author-similarity /
                                  # --skip-author-clustering to omit one; --limit caps the
                                  # two similarity steps only). Use the standalone commands
                                  # directly for --top-k/--top-n/--n-clusters/--min-fame.
```

- **`book-similarity`** / **`author-similarity`** — `idf = ln(N / df)` (N =
  book/author count, df = how many books/authors use a word), cosine
  similarity over each entity's IDF-weighted word-presence vector, storing
  only each entity's **top-k** neighbors (default 12, `--top-k`) above a
  `--min-shared-words` floor (default 3) — never the full O(n²) matrix, so
  cost stays flat as the corpus grows. Words used by more than
  `max_df_fraction` (default 50%) of entities are excluded from the metric
  entirely, not just down-weighted — both a near-lossless approximation
  (their IDF is already near zero) and a performance guard against
  combinatorial cost from ubiquitous words. Both directions are stored as
  separate rows so "X's related items" is always a single indexed lookup,
  no UNION needed. Always recompute everything in scope on every run (no
  only-missing gate) — IDF weights are corpus-wide and shift whenever *any*
  entity's vocabulary changes, not just the one being looked at.
  `author-similarity` excludes `PLACEHOLDER_AUTHORS` (`Various`, `Unknown
  Author`, `Unknown`, `Anonymous`) entirely — aggregation labels, not real
  authors; treating one as a real author produces a spuriously
  well-connected phantom node (`Various` alone has 1,193 books, dwarfing
  every real author).
- **`author-clustering`** — hierarchical clustering (`ward` linkage,
  `scipy.cluster.hierarchy`, `optimal_ordering=True` for matrix seriation)
  plus a 2D projection (classical/Torgerson MDS via a single
  `numpy.linalg.eigh` call — deterministic and fast, unlike sklearn's
  iterative SMACOF) over the top `--top-n` (default 200) authors by book
  count, same corpus-wide IDF weighting as `author-similarity` so the two
  stay consistent. Distance is `sqrt(2 * (1 - cosine))`, a proper Euclidean
  distance for L2-normalized vectors — raw `1 - cosine` fails the triangle
  inequality and produces a non-PSD Gram matrix, forcing lossy eigenvalue
  clipping in the MDS step; this one distance definition feeds both the
  linkage (which assumes Euclidean input) and the projection, not decided
  independently in two places. `fcluster(..., criterion='maxclust')`
  (default `--n-clusters 12`) assigns cluster membership; eigenvector sign
  is pinned deterministically (forcing each axis's highest-magnitude
  component positive) so the map doesn't mirror-flip between runs. Writes
  `author_cluster` (one row per author: cluster id, x, y) and the singleton
  `author_cluster_run` (the full pairwise grid + the dendrogram tree, as
  one blob) together in a single transaction — never partially, since the
  map, matrix, and dendrogram views must always agree with each other.
- **`book-clustering`** — the same technique one level down: top `--top-n`
  (default 200) books by (extracted-vocabulary) word count, same
  corpus-wide IDF weighting `book-similarity` uses, writing `book_cluster`/
  `book_cluster_run` in the identical shape to the author tables above.
  The one real difference: an author's name alone is both display label
  and navigation key, but a book needs id + title + author together (a
  title alone isn't a stable identity — two books can share one, and it's
  not enough to build a link), so `book_cluster`/`book_cluster_run.leaf_order`
  and every dendrogram leaf carry all three rather than a single string.

All four are standalone commands, deliberately NOT part of `maintain`
(full-corpus recomputes, no only-missing gate — see "Post-ingest
maintenance" above) — run them on their own schedule when you want fresher
cross-references, or `concordance relate` to run all four at once. Reachable
from the review webapp's **Visualizations** page — see below.

## Cultural & historical importance (`author-fame`, `book-fame`)

A different axis from vocabulary overlap above: how *important* a book or
author is, not how related it is to anything else. An ABSOLUTE 1–10 score,
LLM-judged against a fixed external rubric anchored on real reference
figures (Shakespeare = 10) — deliberately **not** a corpus-relative
percentile, so a score stays meaningful in isolation and doesn't drift every
time the corpus grows, at the cost of the resulting spread reflecting how
skewed real fame actually is (this corpus is mostly public-domain/pulp-
heavy, so score mass sits low, not spread evenly 1–10 — see
`concordance/fame.py`'s module docstring for the full tradeoff). Evidence
gathered per entity: Google Books Ngram phrase frequency of the name/title,
Wikidata sitelinks count (the standard academic proxy for lasting cultural
importance — Wikipedia presence across many language editions, unlike
recency-biased pageviews), and web search snippets. Every gathering function
records an explicit failure/skip marker rather than silently omitting a
signal — a run scoring an entity blind would look identical to "the LLM
weighed this evidence and found it weak," which is exactly the failure mode
this module is built to avoid.

```bash
concordance author-fame              # score every real author, 1-10
concordance book-fame                # score every book, one level down
concordance author-fame --stub       # dry run: gather + print evidence, no LLM call, no write
```

Run `author-fame` before `book-fame` — book scoring uses the author's
already-computed score as weak context only, never a floor or cap (a famous
author's forgotten minor book still scores low on its own; `book-fame`
tolerates an author with no fame row yet). Excludes `PLACEHOLDER_AUTHORS`.
`book.author` is stored library-catalog style ("Last, First M. (Full Given
Name)") and normalized to a natural name before every external query —
confirmed live that querying Wikidata/Ngram/web search with the raw string
measurably degrades results. A genuinely expensive job (several network
round-trips plus one real LLM generation per entity, realistically 5–15s
each, so a full ~4,000-author corpus is on the order of a day) — deliberately
**not** part of `maintain`; run it as its own occasional pass.
`--stale-days` rescores entities checked more than N days ago (default: only
never-scored ones).

Surfaced throughout the web app once scored: a ⭐ score with a filter range
on the Books/Authors browse pages, the score plus the model's own reasoning
on each book's/author's detail page, and the Home page's "most acclaimed
book" / "most acclaimed author" stat. Coverage fills in gradually, not all
at once — as an expensive occasional pass rather than something `ingest`
triggers automatically, `author-fame` runs well ahead of `book-fame` in
practice, so expect the ⭐ tag to show up on more authors than books until
both passes catch up.

## Analogy relations (`backfill-analogies`)

A word-relation graph built for one specific purpose — the quiz's "*A is to
B as C is to ___*" question type (see "Quizzing" below), not a general
knowledge-graph feature. Two sources feed candidate edges, both required to
pass local-LLM verification against both terms' real definitions before
either is usable — mandatory, not optional: live checks found roughly
60–80% of raw WordNet "hypernym" edges for rare/abstract words are actually
disguised near-synonyms, unfit for a question whose whole point is "these
are different but related."

- **WordNet** relations — hypernym, holonym (part/member/substance),
  antonym, similar-to, derivationally-related, attribute.
- **Definition-pattern parsing** — a spaCy `Matcher` over `word.definition`
  recovers relations for words WordNet has no useful synset for, straight
  from prose that states one outright ("a wooden collar worn by prisoners"
  → *kind of* collar) — exactly the rare/archaic vocabulary this project
  exists to surface, and where WordNet's own coverage is thinnest.

```bash
concordance backfill-analogies         # extract + verify, resumable
```

Resumable at two grains, the same way `classify` is: a term already scanned
is skipped on a later run, and within one run each `--chunk-size` (default
200) batch of terms is extracted, verified, and committed as its own unit —
a kill mid-run loses at most one in-flight chunk of already-GPU-verified
work, not the whole run. Loads the same local judge model `classify`/
`quizdef` do, so — like those — it's slow against a real backlog; it runs as
the last step of `maintain` by default (`--skip-analogies` to defer it).

## OED reference dictionary (`oed-ingest`)

A separate, standalone ingestion pipeline over scanned OED volume PDFs —
not the vocabulary-extraction pipeline above, and not merged into it. Lives
in its own `oed` Postgres schema (`volume`/`entry`/`definition`/`quotation`
tables, no foreign keys into `concordance.word`) with its own admin-only
browsing UI (see "Web app" below) — a reference dictionary to browse and
cross-check against, not a source of quizzable vocabulary.

```bash
concordance oed-ingest                       # every .pdf in dictionaries/
concordance oed-ingest dictionaries/09.pdf   # one volume
concordance oed-ingest --page-limit 20 dictionaries/09.pdf   # smoke-test
```

Per-volume pipeline: hash-check (`file_hash`, so re-running is idempotent —
a volume already ingested is skipped, one that errored partway resumes from
where it stopped) → extract → headword segmentation → stopword filter →
definition-entry filter → parse (headword/POS/etymology/senses/quotations)
→ pronunciation → write. One volume's failure doesn't abort the rest of a
batch run.

- **Headword detection** (`concordance/oed/segment.py`) uses two OCR-text
  heuristics rather than a text-layer bold/italic signal — the baked-in OCR
  text layer from these scans doesn't preserve font weight, so a
  look-for-bold approach (viable on a born-digital PDF) isn't available
  here. A font-size-ratio filter plus a left-margin-alignment check
  (catching small-caps cross-references mid-paragraph that are
  headword-sized but don't start flush at the column margin) together cut
  real-page false positives substantially, without fully eliminating the
  ambiguity between a genuine headword and same-size OCR noise.
- **Pronunciation** is read straight off a cropped image region by a local
  vision-LLM (Qwen2.5-VL via `llama-cpp-python`), never trusted from the OCR
  text layer — falls back to leaving `pronunciation_needs_review = true`
  with no transcription if no vision model is configured, rather than
  guessing from spelling the way `audio-guess` (above) explicitly flags as
  a last resort elsewhere in this project.
- **Known gaps**: an entry that continues onto the next page is truncated at
  the page boundary; every detected headword is currently written as a
  plain `entry_type='main'` (run-on/compound sub-entries and repeated
  homographs like lead¹/lead² aren't distinguished yet); sense/quotation
  splitting is a first-pass regex parser, not yet calibrated against broad
  output the way the pronunciation pipeline was.

Not part of `maintain` — own schema, own concern, run on demand as volumes
are acquired. One exception to "not merged into" the vocabulary pipeline
above: `oed-ipa` (see "Pronunciation audio") reads this schema's verified
pronunciations back into `word.ipa` for words that still lack one.

## Running the local model (RTX 3060, 12 GB)

The judge talks to `llama.cpp` through the `llama-cpp-python` bindings — no
separate server.

**1. Install the bindings with CUDA.** Easiest is a prebuilt CUDA wheel:

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

If no wheel matches your setup, build it (needs the CUDA toolkit on your WSL/Linux):

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-binary llama-cpp-python
```

**2. Get a model.** A Qwen2.5-14B-Instruct GGUF at Q4_K_M is ~9 GB and fits your
12 GB VRAM with context to spare:

```bash
pip install huggingface-hub
huggingface-cli download bartowski/Qwen2.5-14B-Instruct-GGUF \
  Qwen2.5-14B-Instruct-Q4_K_M.gguf --local-dir models
```

Want it snappier for big books? Swap in `Qwen2.5-7B-Instruct` or
`Llama-3.1-8B-Instruct` at Q5_K_M.

**3. Run it.**

```bash
concordance ingest "some book.epub" --model models/Qwen2.5-14B-Instruct-Q4_K_M.gguf
```

The judge defaults to the 14B; if that file is missing it falls back to the
stub judge automatically (which keeps every survivor, letting the pipeline
run end-to-end without a model), and `--stub` forces the stub even when the
model is present. `Config.n_gpu_layers = -1` offloads as many layers to the
GPU as the VRAM allows; drop it if you hit out-of-memory.

## Web app (`webapp/`)

The first slice of a browser-based quizzing/viz/user-management app: a review
table for pruning too-common/easy terms out of the vocab bank. Deleting a term
doesn't hard-delete it — it flips `word.active` to false, so every downstream
feature (quizzing, stats) just needs to filter on `active = true`, and history
(audio, ngram data, etc.) stays intact.

- **Home page** (`/app`, linked from both header logos) — the app's landing
  page: a large logo; four headline counts — active words, books, authors,
  and the fixed 21 top-level USAS categories (the only one of the four
  that isn't a growing corpus measurement); a few "impressive" bonus stats
  (most-acclaimed book/author by fame score — see "Cultural & historical
  importance" above, the single hardest word by difficulty score, total
  quiz questions answered); and a table-of-contents linking every
  top-level section.
- **Accepted tab** — term/POS/definition/difficulty/validity-score/
  validity-label/validity-notes table, filterable by POS and by validity
  label, sortable by validity score (score/label/notes are `validity_score.py`'s
  own real-word-vs-artifact estimate — see "Backfilling definitions" above),
  one-click delete (no confirm) that sets `active = false`; whole-row hover
  highlight so a delete click can't land on the wrong term.
- **Rejected tab** — browse `rejected_word`, filterable by book and by reason
  (both multi-select), with an "Add" button that rescues a word back in (live
  dictionary lookup, since rejects were never enriched) and flags it
  `rescued_from_reject` for after-the-fact tracking.
- **OED entries tab** (admin-only) — read-only browsing of the separate
  `oed` schema `oed-ingest` populates (see above): search/A–Z letter jump/
  volume filter/needs-review filter over the entry list, each row showing a
  definition preview; the detail view shows the full entry — pronunciation
  (flagged if it still needs a human's review), etymology, and every sense
  with its supporting quotations. A QA/browsing tool over raw ingest output,
  not a curation queue — no edit actions, just visibility into data that
  previously required direct SQL to inspect.

Every book's and author's own detail page additionally shows: its ⭐ fame
score with the model's own reasoning (see "Cultural & historical importance"
above); a domain-distribution chart and a difficulty histogram over its own
words that cross-filter each other on click (pick a USAS domain segment to
narrow the histogram to just that domain, or a difficulty bucket to narrow
the chart to just that range — "uncategorized"/"not yet scored" are
clickable segments in their own right, not just the populated categories);
and, in its word list, each definition auto-linked to any other active vocab
word it mentions (`LinkedDefinition.jsx`, powered by `link-definitions` —
see above).

```bash
# first time only
pip install -e ".[web]"
(cd webapp/frontend && npm install)

# every time — runs backend + frontend together, http://localhost:5173
./webapp/dev.sh
```

### User accounts (`create-admin`, invites, `/app`)

The curation UI above (Accepted/Rejected/Graph) is admin-only, enforced by
the app itself (see below) rather than an edge gate. Separately, the app has
its own login — independent accounts for browsing/studying the vocab bank,
gated by one-time invite links rather than open signup:

- `concordance create-admin <username>` seeds the first admin-flagged
  account (prompts for a password via `getpass`; run once per deployment —
  needs to exist and be confirmed working before anything else touches
  who's allowed in, so app-layer auth is never the only thing standing
  between a restart and a lockout).
- A logged-in admin gets a **"+ Generate Invite Link"** button in the
  Accepted tab, which mints a one-time `/register?token=...` link
  (`POST /api/admin/invites`, default 7-day expiry). Whoever opens it sets a
  username/password and lands on `/app` — a non-admin browse page (word
  search → full word detail, including its similarity graph) with no access
  to the curation API.
- The login page also has a **"Request an invite"** link for someone without
  an account yet — a plain `mailto:` link (an optional email field
  interpolated into the pre-filled subject/body), not a form submission or a
  new backend endpoint. The app has no email-sending infrastructure, so this
  hands off to the visitor's own mail client rather than building a first
  real SMTP path just for this one page.
- Sessions are an httpOnly, Secure cookie (`concordance_session`, 30-day
  expiry) backed by a `sessions` table — not JWTs — so a session can be
  revoked server-side (`/api/auth/logout`) rather than just expiring.
  Passwords are hashed with Argon2 (`argon2-cffi`).
- Every route is one of `require_admin` (curation API) or `require_viewer`
  (word search/detail/audio/graph) — this is the sole, load-bearing gate;
  see "Public access" below for why an edge layer (Cloudflare Access) turned
  out to be structurally incompatible with a fetch()-driven SPA and got
  dropped rather than layered on top. `auth.py` still has a
  `verify_cf_access` JWT-verification helper and both dependencies still
  call it, but with no Access application configured it's simply dead code
  — harmless, since it fails closed (returns `None`) when unconfigured.

`dev.sh` also sets `WATCHFILES_FORCE_POLLING=true` for uvicorn (Vite's own
polling is configured in `frontend/vite.config.js`). Both are needed because
this repo lives on `/mnt/c` — a Windows drive mounted into WSL — where native
fs-change notifications don't reliably reach either dev server's watcher, so
edits silently fail to hot-reload without polling.

### Quizzing (`/app/quiz`)

Any logged-in account (admin or invited viewer) can take a quiz — configure
it at `/app/quiz`, driven entirely by `webapp/backend/quiz.py` and
`concordance/distractors.py`:

- **Four question types, blendable in one session**: multiple choice,
  true/false, matching (a set of word↔definition pairs), and **analogy**
  ("*A is to B as C is to ___*", drawn only from words with at least one
  LLM-verified relation edge — see "Analogy relations" above). Pick one type
  for a single-type test or several to mix them. A matching set counts as
  one question toward the test length no matter how many pairs it holds, and
  scores with **per-pair credit** — 3 of 4 correct pairs contributes 0.75 to
  that slot, not a binary pass/fail. Analogy questions have their own
  options-per-question setting (`analogy_choice_count`, 2–8, default 4),
  separate from multiple choice's option count, and their own distractor
  strategy (`select_analogy_distractors`) — a plausible wrong completion has
  to respect the relation itself (an "analogy_trap" option drawn from the
  same relation family), not just general word similarity, so it isn't part
  of the orthographic/semantic/domain/random blend the other question types
  share below.
- **Direction** — "show the definition, pick the word" or "show the word,
  pick the definition." Matching is direction-agnostic (always shows both).
- **Filters**: length (5/10/20/custom), difficulty range, POS, and domain
  (the same 6 USAS buckets `/api/graph/legend` uses, not raw category codes).
- **Distractors** are POS-matched (never negotiable) and drawn from a
  weighted blend of orthographic lookalikes (`pg_trgm` on the lemma),
  near-miss semantic proximity (embedding cosine-distance band — close
  enough to be a plausible mix-up, far enough not to be a true synonym),
  domain/theme similarity (shared USAS category), and random — ratios are
  configurable per quiz, with a smart-vs-random split on top. A target's own
  `synonyms` are always excluded from every strategy — a distractor that's
  actually a valid synonym is a second correct answer, not a wrong one.
  Antonyms are a reserved-but-unimplemented strategy slot (no antonym data
  exists anywhere in this pipeline yet).
- **"None of the above"** (multiple choice only) can be toggled on, with a
  configurable rate (default 15%) at which it's actually the correct answer
  rather than always a decoy.
- **Feedback timing** (reveal correct/incorrect immediately after each
  question, or only at the end) is a single **admin-controlled global
  setting** — Settings tab in the curation UI, backed by a generic
  `app_settings` key/value table — not a per-quiz-taker choice. A session
  snapshots whichever mode was active when it started, so changing the
  setting never affects an in-progress quiz.
- Every quiz-taking route requires only a logged-in session
  (`require_user`) — no admin flag needed, matching/true-false/multiple-choice
  are all available to invited non-admin accounts.
- Every answer is written with the question's actual assembled guessing floor
  (`guessing_floor` — see `calibrate-difficulty` above), which `concordance
  calibrate-difficulty`/`maintain` later folds into a per-user difficulty
  overlay from that word's first exposure; **difficulty range filters**
  above prefer a user's own calibrated value over the shared ex-ante score
  once one exists.

Spaced repetition (re-surfacing missed words sooner, `concordance/
spaced_repetition.py` + `word_review_schedule`) is live as an opt-in
per-session selection bias (`spaced_repetition_enabled`/`_frequency`). Not
yet built: any mastery-tracking dashboard — the schema captures enough
(question type, choice count, NOTA presence, per-answer correctness,
timestamps) to add one later without a backfill.

### Visualizations (`/app/visualizations`)

Every relatedness view in one place, linked from the main Browse page:

- **Books/authors by vocabulary overlap** — type-ahead search jumps
  straight into that book's or author's own ego-anchored relatedness graph
  (that entity plus its top-k related neighbors, `react-force-graph-2d`).
  These are real graphs, not stars: neighbors are cross-linked to each
  other too, wherever that link already happens to be stored (each also
  being the other's own top-k match) — rendered dashed to visually
  distinguish "why you're looking at this" edges from peer relationships.
- **Words by meaning** — the same definition/FastText embedding graph the
  word-detail page uses, embedded here with search enabled.
- **The what, not just the why** — every relatedness view (a ranked list
  row, an ego-graph edge, a matrix cell) can open a **shared-vocabulary
  panel**: the actual overlapping rare words between two books or authors,
  with definitions, rarest first — computed on demand and bounded to
  exactly the pair being compared. The precomputed top-k tables above only
  ever answer *how* related two things are; this is the only view that
  answers *in what words*.
- **Top authors/books at once** (`/app/authors/relatedness`,
  `/app/books/relatedness`) — four tabs over one shared clustering run
  (`author-clustering` / `book-clustering` respectively): a **cluster map**
  (default — position and color are principled, derived from real
  MDS/clustering over the similarity structure, not a physics simulation's
  arbitrary compromise layout), a **seriated similarity matrix** (a canvas
  heatmap, entities ordered so related ones form visible blocks along the
  diagonal; click a cell to open the shared-vocabulary panel for that
  pair), a **dendrogram** (hand-rolled SVG tree, leaf dots colored to match
  the map's clusters so the two views agree), and the original
  **force-directed graph** (kept as a fourth, lower-priority option, not
  deleted — its global response has no real "center," just peers, so it's
  its own response type rather than the ego-graph's shape reused with a
  fake one). Each covers only the top-N entities from its own clustering
  run (200 by default, ranked by book count for authors / word count for
  books) — an unbounded 11k-book or 3.5k-author force-directed layout would
  be an unreadable hairball regardless of compute cost, so a bounded,
  principled subset stands in for "everything," same reasoning either way.
  The page header shows the real, live count (`Top {N} authors`/`books`),
  not a hardcoded "all," and the highlight search box only offers entities
  actually in that run — picking from the full corpus meant most picks
  landed on "isn't in this view," a control that mostly didn't work.
- **Categories drilldown** (`/app/categories`) — each level's sibling
  categories (the 21 top-level USAS discourse fields, then their
  subcategories on drill-in) as a force-directed graph: node radius by word
  count (sqrt-scaled, since counts span orders of magnitude within one
  level), edge width by pairwise Jaccard overlap of the categories' word
  sets. Clicking an edge opens the actual word-set **intersection** between
  those two categories — a new `all_top_code` AND-semantics filter, distinct
  from the OR-semantics category filter used everywhere else.
- **Discipline-category relationship map** (`/app/visualizations/domain-map`)
  — a different axis entirely from every view above: instead of
  shared-vocabulary overlap, this positions books (≥50 words) and authors
  (≥100 words) by how their vocabulary *distributes* across the 21 USAS
  discourse fields (science, law, arts, ...), so two works can sit close
  together with zero words in common. Position is PCA over each entity's
  L1-then-L2-normalized category-distribution vector — the same embedding
  classical MDS produces, computed directly over the 21 dense category
  columns instead of an O(n²) distance matrix, so it runs live per request
  with no precomputed table. Dot color is the category an entity leans on
  **more than the corpus typically does** (lift: its share ÷ the corpus's
  own average share for that category), not its single largest share — a
  raw-share "dominant" color came back "GENERAL & ABSTRACT TERMS" for 82%
  of books in a live check, since that field is corpus-wide dominant
  everywhere and a color that uniform tells you nothing. A spread slider
  (real uniform zoom around the layout's own center, not a per-point
  radial power transform — an earlier attempt at the latter distorted a
  dense cluster into a ring instead of separating it) and click-to-toggle
  legend filtering by domain bucket round it out.

### Public access — `vocab.brfinnegan.org`

The app is exposed to that domain via a **Cloudflare Tunnel** running on this
WSL machine (no port-forwarding/firewall changes) — DNS-only, no Cloudflare
Access application in front of it. Access was tried and removed: it gates by
redirecting unauthenticated requests to a `cloudflareaccess.com` login page,
which only works for a full page navigation. Every API call this SPA makes
is a background `fetch()`, and a fetch that gets redirected cross-origin
fails the browser's CORS check outright (a bare "Failed to fetch", no status
code) — so Access couldn't gate a single `/api/*` route without breaking the
page that calls it, and it can't gate `/register` at all without defeating
the entire point of invite links. There was nothing left for it to
usefully protect.

[User accounts](#user-accounts-create-admin-invites-app) are the actual,
sole gate now — `require_admin`/`require_viewer` in
`webapp/backend/main.py`, fail-closed, verified end-to-end by
`tests/test_auth.py`'s HTTP round-trip test. Removing Access doesn't expose
anything Access was reliably protecting before this: same test suite covered
the app-layer boundary the whole time, Access was always documented as
"redundant, not load-bearing" on top of it (see `webapp/backend/auth.py`).

Setup order still matters on any fresh deployment: the admin account
(`concordance create-admin`) has to exist and be confirmed working before
anything else changes, since `require_admin` fails closed — no admin row
means no one, including Brian, can reach the curation API at all.

Both pieces run as **systemd --user services** (survive reboot/logout via
`loginctl enable-linger brian`, already enabled):

- `concordance-web.service` — runs the backend directly against whatever's
  already built in `webapp/frontend/dist`. It deliberately does **not**
  rebuild the frontend itself — this unit's PATH doesn't include nvm's Node
  (only an interactive shell profile sets that up), so a build attempted here
  silently uses the system's older Node and breaks.
- `concordance-tunnel.service` — runs `cloudflared tunnel run concordance-vocab`
  (config at `~/.cloudflared/config.yml`, tunnel id in that file, credentials
  JSON alongside it — none of this lives in the repo).

To ship a frontend change to the public site: rebuild, then bounce the
service so it picks up the new `dist/`:

```bash
cd webapp/frontend && npm run build
systemctl --user restart concordance-web.service
```

Useful commands: `systemctl --user status concordance-web concordance-tunnel`,
`journalctl --user -u concordance-web -u concordance-tunnel -f`.

**If `systemctl --user` fails with "Failed to connect to bus":** WSLg mounts
its own tmpfs (for `wayland-0`/`pulse` sharing) directly on top of
`/run/user/1000` at some point during/after boot, hiding the real per-user
runtime dir that already has the D-Bus session socket bound into it — the
socket is still alive at the kernel level, just unreachable by path. A
system-level watcher unit, `fix-run-user-runtime-dir.service` (installed at
`/etc/systemd/system/`, script at `/usr/local/sbin/fix-run-user-runtime-dir.sh`,
enabled for boot), polls every 15s and unmounts just the `/run/user/1000`
shadow layer whenever it appears — `/mnt/wslg/run/user/1000` is left alone so
WSLg itself keeps working. Neither file lives in this repo (machine-specific,
like the tunnel config). Check it's running with `systemctl status
fix-run-user-runtime-dir.service`.

## Corpus housekeeping (`book-merge`, `incoming-merge`, `archive-metadata`, `refresh-rejected-index`, `import-defined`)

Occasional/one-time commands, none part of `maintain` — each addresses a
specific corpus-hygiene problem rather than routine per-batch upkeep:

- **`book-merge`** — Project Gutenberg splits some works into multiple
  files ("Vol. I"/"Vol. II", "Part 1", ...). Detects already-ingested books
  sharing a title (once the part label is stripped) and author, compiles
  them into one `"{title} (Complete) -- {author}.txt"` in `archive/`, and
  folds the corresponding `book` rows together in Postgres — idempotent and
  resumable via a `book_merge_group` manifest; original per-volume files are
  never deleted. `--dry-run` reports every eligible group and why a
  near-match was excluded (duplicate number, a gap, an existing sibling
  conflict) with zero writes; `--compile-only` stops before the DB merge.
  Re-run `book-similarity`/`book-clustering`/`book-fame` for touched books
  afterward — their precomputed rows are cleared, not recomputed inline.
- **`incoming-merge`** — the same split-volume detection, run BEFORE
  ingestion instead of after: scans `incoming/` (`.txt` only) and compiles
  matching part-files together prior to `concordance ingest` ever seeing
  them. No DB work (nothing has a `book` row yet), but the original
  per-part files ARE deleted once compiled — destructive; run `--dry-run`
  first.
- **`archive-metadata`** — backfills `word_count`/
  `distinct_nonstop_word_count`/`archive_path` (and, best-effort,
  `publication_year`/`publication_era` for `.txt`/Gutenberg-style books) for
  books whose full text sits in `archive/`. `ingest` already computes the
  same fields inline for new books via the same shared
  `archive_metadata.compute_book_metadata`; this exists for historical
  backlog or re-running after an interrupted/`--no-archive` run.
  Only-missing by default; the Gutenberg year lookup is slow (thousands of
  requests, hours-scale), which is why this stays a standalone pass rather
  than folding into `maintain`.
- **`refresh-rejected-index`** — refreshes `rejected_lemma_index`, the
  precomputed distinct-lemma view behind the Rejected tab's search box and
  A–Z letter jump (`rejected_word` itself is tens of millions of rows,
  hundreds of thousands of distinct lemmas — too big to search directly).
  Cheap (~15–20s); meant to run on its own daily cron/systemd schedule,
  since curation search can tolerate being briefly stale in a way
  `maintain`'s enrichment steps can't.
- **`import-defined`** — one-time bootstrap: imports genuinely-new terms
  from a legacy `vocab.defined` table (term/POS/definition, no book source)
  into `word` as book-less words, skipping phrases, already-flagged-bad
  rows, terms already in `word`, and anything ever rejected in any book.
  Not part of `maintain` — run once, then run `maintain` to classify/score
  whatever it added.

## Status

Every stage is real and runs end-to-end; the LLM judge is live, running the
14B model against every ingest (not the no-model stub — see the fallback
note above). Beyond the base extract → judge → enrich pipeline:
a cross-book verdict cache, a public review webapp, USAS domain tagging, an
ex-ante difficulty scalar with a per-user calibration overlay from quiz
response data (`calibrate-difficulty` — see "Enrichment & scoring" above),
quiz-safe definitions + a quizzable flag, a
pronunciation-audio pipeline (real recordings first, IPA-guided synthesis
otherwise), a unified definition-lookup cascade (one `resolve.py` cascade
behind `refill`/`deepen`/`fill-definitions`/`lookup_word.py`, with
web-search + grounded local-LLM extraction as the default last resort — real-
scale testing found it's where nearly all of a `deepen` run's actual yield
comes from; `ingest` runs the same LOCAL→FREE tiers inline plus Merriam-
Webster as a third, faster fallback — see "Pipeline" above) plus a
human-review flag (not an auto-reject — see "Backfilling definitions" above)
for words that look foreign or like an archaic spelling of a common word, a
separate Merriam-Webster lookup source (`lookup_mw.py` for one-off checks,
`mw-backfill` as a batch pass over whatever `deepen`'s cascade still
couldn't resolve — see "Merriam-Webster word lookup" above) for the words
nothing else in the cascade catches, definition cross-linking so a
definition's mention of another active word renders as a real link
(`link-definitions` — see above), and semantic-distance vectors
(definition-embedding + corpus-trained FastText, queried via a pgvector HNSW
index — infrastructure for future visualization and quiz-distractor
generation, not those features themselves yet) are all in place. Also live:
an absolute, LLM-judged 1–10 cultural/historical importance score for every
author and book (`author-fame`/`book-fame` — see "Cultural & historical
importance" above), a verified WordNet + definition-pattern relation graph
powering a fourth "A is to B as C is to ___" quiz question type
(`backfill-analogies` — see "Analogy relations" above), and a wholly separate
OED reference-dictionary ingestion pipeline over scanned volume PDFs, into
its own schema with its own admin-only browsing UI (`oed-ingest` — see "OED
reference dictionary" above), which now also feeds back into the main
pipeline as an additional pronunciation source: `oed-ipa` backfills
`word.ipa` from OED's verified transcriptions, and `audio` synthesizes those
with a matching British voice instead of the US one every other source uses
(see "Pronunciation audio" above). Vocabulary-relatedness visualization is
also live: book/author lexical-overlap similarity, real (cross-linked, not
star-shaped) ego-graphs, a shared-vocabulary comparison view, a
category-overlap force-directed graph in the Categories drilldown, and —
for both books and authors — hierarchical clustering surfaced as a cluster
map, a seriated similarity matrix, and a dendrogram, plus a discipline-
category relationship map positioning books/authors by USAS-field
vocabulary distribution rather than shared words (see "Vocabulary
relatedness" and "Visualizations" above). A logo-bearing Home page ties all
of the above together with headline corpus stats and links to every
top-level section (see "Web app" above). Deferred by choice: other languages, Anki
export, scanned-PDF OCR, a curated names/gazetteer list to close the one
known gap in proper-noun filtering (every validity authority is itself
somewhat name-polluted; deliberately not started — see
`concordance/validity.py`'s module docstring for the shape of the gap if
picking this up), and a genuinely full-corpus (or near-it) relatedness
view — the cluster map/matrix/dendrogram pages are honestly labeled "top
N," not "all," because both the clustering math (an n×n distance matrix
plus its `eigh` call) and the rendering itself (an 11k-leaf dendrogram or
an 11k×11k matrix is illegible regardless of how fast it loads) stop being
tractable well before real corpus scale. Two separate pieces, neither
started: (1) a standalone `concordance full-relatedness`-style maintenance
command — deliberately NOT folded into `maintain` any more than
`commons-download`/`audio` are, since a real full-corpus run (as opposed
to today's top-200) is a heavy, occasional, run-it-yourself pass, not
routine per-batch upkeep — that computes the complete all-by-all
similarity/clustering rather than a top-N subset; (2) a new frontend view
actually capable of showing that many entities usefully (a
searchable/filterable list, level-of-detail zoom), since the current
map/matrix/dendrogram components would stay illegible at that scale no
matter how the data behind them is computed.

CSV-based ingestion (`run` → hand-edit → `finalize` → `sync-db`) still works
but is no longer the primary workflow — `ingest` writing straight to Postgres,
reviewed in the web app, is. The CSV commands (`run`, `finalize`, `sync-db`,
`define`, and the standalone `python -m concordance.refill`) remain in the
codebase for anyone with an existing CSV-based project, but aren't documented
here; see git history for their usage if needed.
