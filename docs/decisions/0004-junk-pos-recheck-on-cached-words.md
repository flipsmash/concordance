# Junk-POS resolutions are re-checked on every re-encounter

Where: concordance/pipeline.py (`process`)

Moved here verbatim from the code comment it explained; the code keeps
the current rule and a one-line reason.

Applies to cache-sourced candidates too (already `known` — an
established KEEP from an earlier book): enrichment re-runs on them
since it isn't cached, and a junk-POS resolution is a structural
signal, not enrichment's own non-determinism — every other place in
this codebase treats it as authoritative wherever it's seen, and a
word's first-ever lookup happening to land on a different sense
before the junk one ever surfaced is exactly why this needs to keep
checking on every re-encounter, not just the first. (Confirmed in
the wild: taxonomic Latin genus names — linnaea, olor, hircus —
sat active and defined for weeks because this check used to skip
them once cached, even as later books' lookups kept correctly
resolving "proper noun" and being ignored.) sync_book_results casts
the word out (active=false) if it already exists, same as
refill/deepen do for their own junk-POS resolutions.
