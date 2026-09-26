# Merriam-Webster lookup lock covers bookkeeping, not the network call

Where: concordance/mw.py (`_LOCK`)

Moved here verbatim from the code comment it explained; the code keeps
the current rule and a one-line reason.

Guards ONLY the on-disk cache/usage-counter read-modify-write in
lookup_api's uncached branch -- NOT the network call itself. Originally
wrapped the whole branch (cache re-check, quota check, the `_get` call
with its own internal retry/backoff, cache write), which meant two worker
threads that both needed a fresh (uncached) MW lookup in the same book
could never issue their HTTP requests concurrently: the second thread
queued behind the first's ENTIRE retry chain (dictionary._get's 4-try
exponential backoff, worst case tens of seconds to ~2 minutes on a
throttled/erroring response) before it could even start its own. With
ingest's enrichment batch waiting for every worker to finish (see
pipeline.process's ThreadPoolExecutor loop), two or three such words in
one book's shortlist reproduced live as exactly "the last couple of
definitions take forever" -- confirmed by inspection, not just theory:
this lock previously spanned mw.py:163's `_get(...)` call directly.

Narrowing it to just the cache/quota bookkeeping (each a fast local
read/write, not a network round trip) lets concurrent MW lookups actually
run in parallel, which is the whole point of ingest's worker pool. The
tradeoff, accepted deliberately: two threads racing the SAME uncached
word can both slip past the quota check before either increments the
counter, so usage can overshoot _DAILY_QUOTA by up to a few requests
(bounded by enrichment_workers, 4 by default) -- utterly preferable to
serializing every network call for a quota that has 1000 requests/day of
headroom. The cache-save race this lock was ALSO guarding against (two
threads resolving two different new words each loading the cache before
either saves, second save clobbering the first thread's new entry) is
still fully prevented: the cache is re-loaded fresh immediately before
each locked save below, so a save that lost the race for lock acquisition
still merges against whatever the winner just wrote, never clobbering it.

Thread-scoped only, not cross-process: this does not protect against a
concurrent `ingest` and `mw-backfill` invocation racing the same on-disk
cache/usage files. Don't run them at the same time.
