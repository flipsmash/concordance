# Decisions

Longer histories behind rules in the code: what was tried, what broke, and
the evidence. Code comments state the current rule and why in a line or two
and link here for the rest.

- [0001](0001-mw-lock-scope.md) Merriam-Webster lookup lock covers bookkeeping, not the network call
- [0002](0002-classify-chunked-commits.md) Classification commits in chunks and re-checks each chunk's words
- [0003](0003-fill-definitions-single-pass.md) fill-definitions is one pass, reaches MW, and always tries WEB
- [0004](0004-junk-pos-recheck-on-cached-words.md) Junk-POS resolutions are re-checked on every re-encounter
- [0005](0005-variant-detector-flags-only.md) The foreign/spelling-variant detector flags for review, never casts out
- [0006](0006-classifier-batch-and-sentence.md) Classifier: two steps (field, then code), one word per call; sentence only without a definition
