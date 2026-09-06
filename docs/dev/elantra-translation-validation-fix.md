# Numeric compounds and contextual translation validation

Reviewed on 2026-09-05 against the user-supplied Elantra development triplet.
This is a local bugfix checkpoint, not a full translation-pipeline replacement.

## Changes

- Numeric modifiers such as `six-speed` now receive the same boundary protection
  as `six speed`. ASCII and Unicode hyphens are supported; sentence dashes remain
  excluded. Existing rule weights and normalization acceptance thresholds stay
  unchanged.
- The suffix-based adjective heuristic no longer calls the next clause a missing
  noun when punctuation, a subordinating conjunction, an explicit subject, and a
  finite auxiliary establish a new clause. Parenthetical adjective phrases and
  incomplete evidence retain the previous risk. This lets the normalizer keep a
  complete numeric modifier and head noun together, then split at the comma.
- Chinese localization of `DCT` is accepted only when the same source cue clearly
  identifies a vehicle or transmission. It does not exempt the mathematical
  abbreviation or borrow evidence from adjacent cues. The new acceptance path
  also requires exact source gear counts in Arabic or Chinese numerals, since
  the legacy general token extractor skips single-digit values.
- `normalize_boundaries` checks for any translated input before every rebuilding
  pass. The old check ran after compact merging and could return rebuilt cues
  with their translations already lost. Partially translated input is protected
  as a whole.

No prompt, provider, online retry, or native-reasoning policy changed. No fixed
historical translation was added. The four changes can be reverted independently;
existing translation and recovery orchestration remains in place.

## Evidence

The supplied machine and reference files each contain 589 cues with identical
indices, source text, and timing; 164 target edits are differences, not error
counts. The original word-level file contains 5,942 cues.

Offline replay covers this triplet and three existing, hash-validated development
triplets: 1,415 machine cues, both machine/reference preservation checks, and
2,822 adjacent source boundaries. The new detector identifies one previously
missed boundary in the supplied sample; the acronym fix removes exactly two
reference-side false positives. All other frozen decisions match, including all
three other development videos. No holdout was inspected.

Reconstruction using the real word timestamps moves the affected break one word
later, preserving every word, order, timestamp, and speaker field. The result is
idempotent. Baseline and candidate snapshots each repeat byte-for-byte. These
fixtures are explicitly `srt_reconstruction`, not captured API responses.

Twelve selected regression checks fail on the original implementation and pass
after the fixes. The full non-integration suite passes 2,652 tests, with 35
integration tests deselected; 72 focused checks, Ruff, Pyright, and the diff check
also pass. Original triplet hashes remain unchanged.

Local evidence, frozen policies, failed intermediate tests, audit matrix, and
adoption decisions are under:

```text
artifacts/translation-quality/runs/20260905-elantra/
```

No external model calls were made during that offline checkpoint.
Source-aligned translation, partial English residue, idioms, ASR ambiguities,
RPM localization, and broader naturalness issues remain separate work. This
checkpoint does not establish live-generation quality, cost savings, independent
blind preference, or improvement on real mixed-language videos.

## Subsequent authorized GLM validation

The user then requested real GLM validation. One full pre-fix run and one frozen
four-fix run used `glm-5.3-flash`, concurrency 20, batch size 20, and reflective
translation on the same supplied development sample. Both completed: 590 versus
582 cues, 259 versus 309 API attempts, 588,463 versus 681,886 tokens, and 423.126
versus 502.692 seconds. The candidate required document-level recovery after a
single-item price-expression validation failure, then passed normal completion
checks and saved a processed result.

The final candidate keeps the numeric modifier with its head noun at the actual
affected timestamp. Full source/target review of all 1,172 output cues nevertheless
found unresolved partial English residue, cross-cue semantic ownership, ASR
interpretation, referent, and idiom errors. Static hard checks reported zero for
both runs and missed those problems. A standalone missing-number signal on a
localized mid-thirties-thousand price is a false positive, not lost numeric value.

The prospectively frozen schema-2 bugfix policy returned `observe`: token usage
rose 15.88%, above its 15% allowance; wall time remained within its 20% plus ten
seconds allowance. Historical blanket 5% gates in the comparison output are
informational. One stochastic development pair does not establish causal cost
growth or general quality preference. The locally proved fixes remain in the
working candidate; no full-pipeline acceptance or packaged rollout is claimed.

Both runs captured 30 main-translation batches, each replayed twice offline with
identical within-process results. Candidate capture matching was 30/30; baseline
was 29/30. The remaining baseline outcome matched, but an unordered missing-token
set reversed two feedback items, changing later request hashes. Permutation/hash
evidence proves this limitation; the mismatch was retained rather than relabeled
as a pass. These captures exclude source splitting and final document repair.

All captured pretranslation source mappings match their respective final source
cues, original triplet hashes remain valid, and all four production modules still
match their frozen candidate hashes. No production source was edited during the
paired calls. Detailed local evidence, actual subtitles, and the Chinese report
are under `artifacts/translation-quality/runs/20260905-elantra-glm/`.
