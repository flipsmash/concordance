# The foreign/spelling-variant detector flags for review, never casts out

Where: concordance/pipeline.py (`process`), concordance/validity_score.py (`variant_reject_reason`)

Moved here verbatim from the code comment it explained; the code keeps
the current rule and a one-line reason.

validity_score.variant_reject_reason (foreign-word / archaic-
spelling-variant detection) is NOT wired in as a hard cast-out
here: real-scale testing (a 31k-word dry-run sweep) found it flags
~21% of the live vocabulary, and a sample of the flagged words was
mostly genuine rare vocabulary (haft, glaive, thurible, discomfit,
kickshaw, outlawry) rather than the foreign/misspelling junk it
was built to catch — edit-distance similarity doesn't imply a real
spelling-variant relationship, and cross-language zipf can't
separate a foreign word from an English word that's ALSO a word
in that language (haft, argent, rood are all real English).
Instead it's a human-review flag: the word is kept/defined
normally, and Candidate.variant_flag_reason/_note (picked up by
sync_book_results) mark it for a person to glance at and manually
prune via the review webapp if it really is junk.
