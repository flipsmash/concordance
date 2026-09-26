# Classification commits in chunks and re-checks each chunk's words

Where: concordance/classify.py (`classify_and_store`)

Moved here verbatim from the code comment it explained; the code keeps
the current rule and a one-line reason.

Commits every `commit_every` words, not once at the end -- this is by
far the longest-running maintain step (an hours-to-days local-LLM pass
over the whole backlog when there's one), so a crash partway through
used to lose every word classified so far, no matter how close to
done it was: the old code held the whole run's results in memory and
issued exactly one conn.commit() after the entire batch finished.
Found live: a run crashed ~3 hours in with zero word_category rows
written. Chunking is safe to do here without changing what gets
classified -- Classifier.classify() already just loops over its own
internal self.batch-sized (default 15) LLM calls with no state shared
across them, so calling it once per outer chunk instead of once on
the full item list produces identical per-word results. Paired with
only_missing's own re-select-what's-still-missing query, a killed and
restarted run resumes close to where it left off instead of
re-classifying from scratch.

The word list itself is one big snapshot SELECT taken up front, before
the (possibly hours-long) chunk loop below even starts -- on a large
backlog a word from that snapshot can be deleted out from under this
run by the time its chunk is reached (a concurrent prune via the web
app, or any other admin cleanup touching `word` directly). Found live:
a maintain run crashed on a ForeignKeyViolation inserting word_category
for a word deleted mid-run by an unrelated cleanup. Each chunk re-checks
which of its own word ids still exist immediately before use, dropping
any that don't, rather than trusting the stale snapshot -- cheap (one
indexed SELECT per chunk) and avoids wasting an LLM call on a word
about to fail to insert anyway.
