"""OED PDF extraction — a separate `oed` Postgres schema (same database as
`concordance`) populated from scanned OED volume PDFs dropped in dictionaries/.

Kept as its own subpackage rather than folded into the main ingest pipeline:
different source format (scanned dictionary volumes, not books), different
schema, and a pronunciation-extraction step (pronunciation.py) with real
accuracy stakes that the main pipeline has no equivalent of. See
concordance/oed/pipeline.py for the entry point (`concordance oed-ingest`).
"""
