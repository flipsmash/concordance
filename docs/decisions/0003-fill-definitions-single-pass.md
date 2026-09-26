# fill-definitions is one pass, reaches MW, and always tries WEB

Where: concordance/db/definitions.py (`fill_definitions`)

Moved here verbatim from the code comment it explained; the code keeps
the current rule and a one-line reason.

The single definition-acquisition pass for words whose definition is
still blank: one candidate SELECT, one lexicon build, one per-row trip
through resolve.resolve_definition at whatever depth `use_web` allows
(YOURDICT without it, WEB with it) -- replaces what used to be two
separate passes (refill_definitions then deepen_definitions) each
re-entering the cascade at Tier LOCAL, the second one's local/free
attempts always redundant with the first's on the same lemma.

Tier.MW (mw_api_key auto-discovered from MW_DICTIONARY_API_KEY, same as
Wordnik) is included automatically -- this is the first `maintain` step
to try it; previously only ingest-time enrichment and the standalone
`mw-backfill` command ever reached MW, so a word needing MW specifically
(not LOCAL/FREE) sat undefined through every `maintain` run until someone
remembered to run `mw-backfill` by hand.

WEB (when use_web) is tried for EVERY word nothing else defined, regardless
of its validity estimate -- there used to be a pre-gate skipping WEB for anything already scored
likely-artifact, on the theory that a web search for OCR noise was
wasted effort; dropped because that same "probably not a real word"
signal is exactly the rare/archaic vocabulary this project's judge
rubric exists to prize, and a word simply not matching any of the
dictionaries checked earlier is not strong enough evidence to skip the
one source most likely to catch what they all missed.
