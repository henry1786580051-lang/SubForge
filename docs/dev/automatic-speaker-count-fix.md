# Automatic Speaker Count: Single-Narrator Regression

Date: 2026-08-31

## Root Cause and Correction

The automatic branch in `subforge/core/asr/transcribe.py` used a speaker-count
range of 2-10. Community-1's VBx clustering can discover one voice, but its
subsequent count constraint forces K-Means re-clustering when that result is below
the requested minimum. Consequently a solo narrator could be split into two
artificial voice classes before word assignment or translation began.

Automatic mode now permits 1-10 speakers. This is a lower bound, not a forced
one-speaker result. Explicit two-person mode remains 2-4 and fixed-count mode
continues to pass the exact requested count with no automatic bounds. A saved
fixed count is ignored in automatic mode.

The frontend states the automatic range. Offline benchmark defaults and usage
instructions match production. Historical benchmark scores have not been altered;
use `--min-speakers 2` to reproduce their original constraint.

No embedding model, clustering threshold, acoustic verification gate, overlap
handling, denoising policy, word timestamp, or translation prompt was changed.
The existing cache key includes all count parameters, so the new automatic mode
cannot read a result produced with the old minimum. No global cache purge is needed.

## Real-Audio Verification

Local Community-1 inference ran on Apple MPS with existing ECAPA512-LM verification.
Only speaker analysis and assignment were rerun; existing ASR words and timestamps
were held constant. No LLM API was used, and original user files were not overwritten.

| Sample | Audio duration | Before | After | Word/timing preservation |
| --- | --- | --- | --- | --- |
| Why Dubai's New Giant Faces a Race Against Time | 16:43.594 | 2 classes, 29 word-label switches | 1 class, 0 switches | All 2,736 words unchanged |
| Why Japan Builds Like Nowhere Else | 89:41.194 | 10 classes, 182 word-label switches | Identical | All 13,389 words unchanged |

For Dubai, the original 2-10 replay reproduced all 124 cached exclusive turns
exactly. The corrected production speaker path completed in 43.84 seconds. The
opening `And having been on top of Merdeka 118 myself` now belongs to one speaker
and remains one input group in the existing subtitle splitter. Processed cues 4-6
no longer have the reported ownership disagreement.

For the long multilingual sample, both full-audio runs produced identical regular
and exclusive turns, overlap regions, and every word label. Five already-unassigned
words remain unassigned in both runs; the correction does not introduce new gaps.
Measured speaker analysis plus verification took 230.21 seconds with the old bound
and 224.67 seconds with the new bound. These single runs do not establish a speedup.

These are product regressions, not human-annotated DER/WDER measurements. Ten voice
classes do not establish ten distinct real people. The earlier two-person and
five-person interview files were no longer present in the available YouTube folder,
so the existing long multilingual video was used instead. Two- and five-speaker
mode contracts also have deterministic regression coverage, not new real-audio
accuracy measurements for those earlier interviews.

Local evidence is under `build/diagnostics/dubai-speakers-20260831/`:
`report.json` records the initial controlled experiment; `regression.json` records
the production speaker-path check and the long-video A/B comparison. Temporary
decoded audio was cleaned automatically. The corrected word-level diagnostic SRT
is separate from the user's source and translated files.

## Regression Coverage

- Automatic bounds allow one speaker while retaining the existing upper limit.
- One, two, and five detected speakers preserve all source words and timestamps.
- Real detected turn boundaries remain hard boundaries for subtitle grouping;
  a continuous single-speaker prefix is not split by an invented speaker change.
- A stale saved fixed count does not influence automatic mode.
- Explicit two-person and fixed-five-person configurations retain their bounds.
- Old/new automatic, explicit two-person, and fixed-count cache keys are distinct.
- Benchmark CLI defaults match the application and allow historical overrides.
- Existing translation-policy tests retain monologue prompts for one unique label
  and dialogue prompts for multiple labels.

Validation: 2,421 non-integration Python tests passed (35 external integration tests
excluded); 26 frontend tests passed. Ruff, TypeScript, ESLint, and the frontend
production build passed. Targeted Pyright reported zero errors and one pre-existing
warning in the benchmark's scoring helper.

## Scope Limits

This does not repair already exported bilingual subtitles in place or regenerate
translations. Re-run speaker analysis and segmentation to replace stale labels.
No claim is made that all remaining ASR proper-name or translation errors are fixed.
No production grammar-based speaker merging was enabled. The installed `.app`
must be rebuilt separately before this source change affects packaged users.
