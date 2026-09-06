# Local validation efficiency checkpoint

2026-09-05. Four narrow changes remain in the working tree: exact RPM
localization, stable missing-token order, identical successful fidelity-check
reuse, and conservative weak-boundary exclusions. The experimental midpoint-price
extension was withdrawn after GLM exposed an incorrect acceptance. No prompt text
or new model-review pass was added. This extends the existing four-fix Elantra
checkpoint; it does not replace the full pipeline or certify its overall quality.
At experiment close the subset had not been packaged; the later user-authorized
local build is recorded at the end of this document.

## Retained implementation

- `localized_quantities.py`: an explicit source RPM quantity permits the same
  adjacent number followed by 转/轉, including Chinese continuation. Changed
  values and lexical 转账/转弯 uses do not establish the rotational unit. The
  integration is restricted to Chinese target languages. The previous bare-RPM
  shortcut could attach another number's unit and rejected Chinese continuation.
- `preservation.py`: important tokens are deduplicated in first-source-occurrence
  order. Missing-token feedback is stable across Python hash seeds, with the
  existing diagnostic vocabulary and recovery prompt unchanged.
- `verdict_cache.py`: a 256-entry task-local cache reuses only successful,
  identical window-fidelity checks and shares concurrent identical requests.
  Identity covers actual messages, model/client, language, custom prompt,
  translation context, and source/speaker/language/gap maps. Changed input,
  rejected/malformed verdicts, and new documents cannot reuse an old acceptance.
  Generation boundaries protect against old in-flight completion; cancelled
  translators do not bypass cancellation on a cache hit.
- `closed_boundary.py`: excludes only the two weak function-word/pronoun signals
  when an English sentence is closed, source-boundary risk is zero, and both
  Chinese sides match narrowly complete structures. Strong signals, long display
  gaps, source-open phrases, ambiguous nominal continuations, and absent evidence
  retain the old path. Each addition can be reverted independently.

Source splitting, provider settings, retry caps, and reasoning budgets were not
changed by this checkpoint. Prior numeric-compound/DCT/source-integrity fixes are
preserved. No historical full translation was hard-coded.

## Per-component evidence and remaining questions

This is a record of what remains in code, not a retrospective declaration that
every component passed an independent quality or efficiency adoption gate.

| Component | Evidence supporting retention | Unverified outcome |
| --- | --- | --- |
| Exact source-backed RPM localization | Fixed-input and negative cases cover same quantity/unit, Chinese continuation, changed values and lexical collisions; one reference false positive disappears, and two historical batches each use one fewer saved response | General semantic safety outside tested forms and fresh whole-task token savings |
| Stable diagnostic order | First-source-occurrence order is stable across tested hash seeds; existing diagnostic wording is retained | Better model repair quality or lower live retry cost |
| Identical successful fidelity-check reuse | Bounded task-local identity, changed-input invalidation, rejection retry, cancellation and concurrent-generation behavior are tested | Real workload benefit: all measured runs had zero hits/shared requests; no measured saving is attributed to this component |
| Narrow closed-boundary exclusions | Fixed positive/negative and call-path tests retain strong, open, ambiguous and long-gap cases; one final C2 boundary qualifies for exclusion | Independent semantic non-inferiority, general false-positive reduction, and actual live requests avoided |

In particular, the boundary exclusion changes accepted diagnostic outcomes. Its
deterministic implementation and local tests do not independently establish
semantic non-inferiority. The cache's reuse behavior is tested, but an efficiency
adoption claim requires observed hits and appropriate quality/cost evidence.
The composite experiment did not isolate live benefit per component. Future
claims must follow the component/combination protocol in `优化指南.md` 2.2;
this documentation update does not retroactively supply the missing evidence.

Rollback for these additions is by scoped code reversion/rebuild, or restoration
of the archived previous app as a whole. This checkpoint does not provide four
independent user-facing runtime switches.

## Offline verification of the retained subset

The full non-integration suite passes 2,690 tests (35 deselected), including 38
new cases. Seventeen selected positive regression checks fail on the frozen
pre-change working baseline. Ruff, Pyright, and the diff check pass.

Four hash-validated development triplets cover 1,415 machine cues and their
reference translations. One reference-side RPM false positive disappears; no
new preservation or Chinese-boundary diagnostic appears. The other three videos
retain their scanned decisions. Sixty historical main-agent captures replay twice
with deterministic results on both baseline and retained code. Two RPM batches
consume one fewer saved response each; these fixed-response replays are not
estimates of fresh model token usage. No holdout was inspected.

## GLM experiment and withdrawal

All runs use GLM 5.3 Flash, concurrency 20, batch size 20, reflection enabled,
explicit client isolation without local disk response caching, and hash-frozen
production modules. Both baselines use the working four-fix implementation from
before this follow-up, not historical Git HEAD.

The initial pilot C1 exposed missing whitespace normalization on `3 万 5 左右`
and an unintended veto of earlier accepted price forms. A prospective amendment
recorded the functional correction before C1 completed or any v2 call began.
C1 (663,231 tokens) remains visible and excluded from formal v2 repetitions.
Order was B1, C1 pilot, C2, C3, B2. Token allowance stayed at 5%; wall-time
allowance stayed at 20% plus ten seconds on the aggregate.

| Run | Role | Cues | Requests | Tokens | Seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| B1 | First baseline | 590 | 286 | 679,411 | 290.711 |
| C2 | First v2 candidate | 589 | 278 | 656,621 | 260.400 |
| C3 | Second v2 candidate | 588 | 250 | 604,456 | 229.724 |
| B2 | Second baseline | 594 | 262 | 612,553 | 241.503 |

V2 aggregate tokens fell from 1,291,964 to 1,261,077 (-2.39%), attempts from 548
to 528, and wall time from 532.214 to 490.124 seconds. Individual token changes
were -3.35% and -1.32%. No API attempt failed. Static hard checks passed; all 120
formal main-agent captures match frozen-code replays repeated twice. Captured
pretranslation source maps exactly match each run's final source cues. Original
triplet hashes remain unchanged.

Full non-blind source/target review of all 2,361 formal output cues found mixed
quality. Candidate examples include an off-state assertion mistranslated as a
week of driving, reversed rev-hang meaning, cross-cue ownership errors, and
English residue. Both baselines also contain material mistakes. Most generation
differences cannot be causally assigned to these code changes.

One rejection *is* directly relevant: C3's first accepted main-agent response
invented `三十五六万……不对，三万多美元` for the price cue. The new equivalence path
accepted this incorrect self-correction. The entire midpoint-price extension was
therefore removed from default code, rather than adding another case-specific
word filter. The prototype, its tests, captures, and cost results remain frozen
under ignored artifacts; a regression verifies that the retained implementation
does not grant this new acceptance. Existing legacy price tolerances remain;
this withdrawal is not a general invented-number detector.

The successful-verdict cache had zero measured hits/shared requests in these
runs; no measured saving is attributed to it. The final C2 output contains one
source-backed weak-boundary exclusion at cue 142, but this post-run observation
does not establish an exact number of avoided live requests.

The v2 static budget screen returned `review`, while semantic review rejected
whole-candidate adoption. The final retained subset differs from measured v2 by
removing price equivalence and was rechecked offline, not rerun through a fresh
paid full-pipeline pair. Consequently **-2.39% is an experimental result, not a
cost claim for the final retained version**. General cost stability, independent
blind preference, multi-speaker/mixed-language live coverage, and public release
remain unclaimed. Local packaging was subsequently completed as recorded below.

All five authorized full runs have finished. The Chinese report, policies,
protocol amendment, frozen code, raw outputs, semantic notes, telemetry, replay
results, and final integrity evidence are under:

```text
artifacts/translation-quality/runs/20260905-low-token/
```

## Subsequent local app delivery — 2026-09-05

After experiment close, the user explicitly requested updating the app. The
retained subset and preceding fixes were built into `dist/SubForge.app` 1.2.1,
which was opened locally. The supported build verified signing and resources;
all eight changed/new algorithm modules matched current source in the package.
The packaged smoke checks passed for backend HTTP, FFmpeg/ffprobe, MLX Metal,
PyTorch MPS, alignment imports, and speaker-diarization imports/basic assignment.
These checks are not a full transcription or translation quality benchmark.

Delivery evidence is in `artifacts/desktop-update-20260905/result.json`, with
build/smoke logs and `packaged-code-verification.json` alongside it. The previous
app's signature was verified before saving `SubForge.previous.tar` there.
This was a local app update only; no public release or additional GLM run took
place. The final-subset live-cost gap and full-pipeline quality limitations above
remain unchanged by installation.
