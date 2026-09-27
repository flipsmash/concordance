# Classifier: two steps (field, then code), one word per call; the sentence only without a definition

Where: concordance/classify.py (`Classifier._classify_one`, `_FIELD_SYSTEM`, `_code_system`, `_prompt_items`)

Measured 2026-09-26 on a hand-labelled set of 150 active words, 30 per stratum:
single-sense, multi-sense, cross-reference, blank definition, and "sentence"
that is really a glossary/index line. Each word's acceptable primary codes
were written down before any run. Cells are primary-code correct / top-level
field correct / scored (143; 7 blank-definition words had no defensible label).

| variant (Qwen2.5-14B unless noted) | primary | field | omitted | primary flips vs. re-run |
|---|---|---|---|---|
| A: def + sentence, batch 15 | 60-64 | 105-106 | 7-8 | 77/150 |
| B: sentence only if no def, batch 15 | 61 | 100-111 | 1-17 | 94/150 |
| C: sentence if no def or multi-sense, batch 15 | 60-75 | 97-111 | 3-19 | 81/150 |
| D: B at batch 1 | 62-63 | 104-105 | 1 | **12/150** |
| E: B, 3-run vote | 68 | 111 | 0 | n/a (3x cost) |
| F: two-stage, fields listed by name only | 66-68 | 95-101 | 0 | - |
| **F2: two-stage, each field listed with the categories it covers** | **78** | **107-108** | 0-1 | **0/150** |
| Qwen3-30B-A3B, C | 58-65 | 97 | 15 | 89/150 |

Findings:
- The sentence makes no measurable difference beside a definition, and it
  can't be dropped outright: ~3,200 active words have no definition and the
  sentence is all they have. Kept only for those.
- Batching was the source of the churn: which other words shared a batch
  changed over half the primary codes at temperature 0. One word per call is
  as accurate, no slower, and nearly deterministic.
- Accuracy (~45% exact code, ~75% field against this strict gold) did not move
  with the sentence policy, voting, or the larger local model. The model does understand the words (asked plainly it calls a
  boomslang an animal); it is picking from the ~230-code list that fails.
  The chat template was verified correct (Qwen ChatML from the GGUF).
- Two steps DO help once the field step shows what each field covers (F2,
  Brian's suggestion): exact codes 63 -> 78 of 143 (paired: 32 words right
  only with two steps, 17 only with one), field unchanged, and fully
  repeatable (0 of 150 primaries changed on a re-run). Cost: ~2.4 s/word
  instead of ~0.9. The shipped Classifier reproduces F2 exactly (78/108).
- Against the categories actually live at the time (one run of the old
  batch-15 setup, a good draw: 72 exact / 114 field), two-step is a wash:
  78 / 108; paired, +6 exact and -6 field, neither significant. So the
  corpus was NOT reclassified (~47 GPU hours for no measurable gain). Two-step
  stays the classifier for new words and re-runs because it is repeatable:
  a word's categories change only when its definition does. The weaker half
  is the field step -- the place to work if accuracy matters later.
- Other local models that fit 12 GB, shipped two-step classifier, same set
  (2026-09-27): Qwen2.5-14B (current) 78 / 108; Gemma 3 12B 76 / 106 (paired
  27 vs 29 -- a tie; ~20% faster); Qwen3-14B, non-thinking mode, 60 / 97
  (paired 16 vs 34 -- clearly worse). Kept Qwen2.5-14B. The eval set and
  scripts live in the git-ignored data/classify_eval/ (book sentences).
