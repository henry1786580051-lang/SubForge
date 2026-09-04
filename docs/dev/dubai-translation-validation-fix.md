# Dubai Subtitle Validation Fixes

Date: 2026-09-01. Scope: deterministic validation correctness, not a new pipeline.

## Adopted Changes

1. `quality/numbers.py` normalizes only complete, valid thousands-grouped numbers.
   Preserved-number validation and Chinese magnitude ownership share it. The old
   suffix-only rule treated a year followed by a three-digit quantity as a single
   number, rejecting correct translations. Real thousands groups, decimals, and
   ASR-inserted horizontal spaces remain supported. Ambiguous numeric enumerations
   are not resolved by guessing.
2. All four document-level `_translate_alignment_item` callers pass `source_key`.
   Previously the default `"1"` could validate a later cue against the first cue's
   source. No ownership validator or speaker boundary is relaxed.

## Evidence

- Fixed-response replay of 14 real baseline batches: 13 unchanged; the numeric
  false-positive changes from failure after three responses to acceptance of its
  first complete response. Replay made no API calls.
- Numeric tests: 29 cases including missing facts and magnitude injection.
- Source identity: 8 cases covering real indices, single/multiple speaker labels,
  first-cue name contamination, and explicit keys at every production call.
- Offline suite: 2491 passed, 35 integration tests deselected. Changed modules pass
  Ruff and Pyright; diff whitespace checks pass.
- Four complete production-worker runs used only GLM-5.3-Flash, concurrency 20,
  batch size 20. The reference was local evaluation data, never prompt input.

| Run | Seconds | Tokens | Attempts | Input Cache Hit Rate |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 332.765 | 389851 | 173 | 17.24% |
| Number fix | 261.101 | 313932 | 132 | 12.49% |
| Number fix + isolated context prompt | 272.743 | 340678 | 152 | 20.90% |
| Number fix + source identity (adopted) | 223.450 | 302576 | 132 | 11.28% |

These are single-run observations, not a statistically established speedup.
The baseline and final run each retried one transient provider error. Final output
has 263 cues with no detected empty targets, placeholders, reasoning leaks, or
timestamp overlap. Whole-document review still finds awkward boundaries, literal
phrasing, and technical sense errors. Static success is not literary quality.

## Not Adopted

`context-role-contract` remains an explicitly opt-in harness experiment, never
imported by the application. Separating ASR correction and translation roles in
the context prompt did not establish a consistent quality gain on this sample.
Do not enable it in production based only on fewer bilingual glosses.

## Limits and Next Work

- This is a reviewed development sample, not a blind quality test.
- Audio timing and acoustic speaker identity were not revalidated.
- Tests protect shared single/dialogue behavior; one online sample cannot prove
  every domain or speaker mode improved.
- Prioritize source-backed boundary changes and trusted localized entity identity.
  Do not expand case-specific replacements or enable reasoning everywhere.
- No installer, application bundle, or GitHub operation was requested here.

Local evidence, not committed and containing user subtitles:
`artifacts/translation-quality/runs/20260901-dubai/结果与采用决定.md`.
