# Translation Quality Refactor Plan

Status: Phases 0-7 complete; Phase 8 experimentation and Phase 9 blind holdout complete; all behavioral candidates rejected for rollout and production remains legacy

Admission update (2026-09-01): future work follows the three-track protocol in
`优化指南.md`, sections 10-12. Bug fixes, quality improvements, and efficiency
changes have different evidence requirements and prospectively frozen budgets.
The historical 5% trials below retain their original decisions; this update
neither reinstates candidates nor changes production translation behavior.

Last reviewed: 2026-08-30

Execution guide: `docs/dev/translation-quality-refactor-execution-guide.md`

This document defines the target architecture for refactoring SubForge's subtitle
splitting, translation validation, and repair pipeline. Execution is governed by
the guide linked above. Corpus inventory, deterministic evaluation, immutable
session data, typed diagnostic adapters, and source/timeline write-back guards
are now in place. The legacy production path remains the default; broader
behavior changes and production replacement still require the evaluation gates
described below.

## Why This Work Was Deferred

The current pipeline is functionally mature and covered by a large regression
suite, but its central translation and boundary modules have exceeded a reliably
maintainable size. Replacing them without an independent quality baseline would
risk trading known edge cases for less visible regressions.

The safety constraints remain:

- keep the existing production path;
- do not add more full-sentence, case-specific replacements;
- collect representative gold samples before changing output behavior;
- separate structural refactoring from prompt or quality-policy changes.

## Audit Snapshot

The review found the following structural risks:

- `subforge/core/translate/llm_translator.py` is approximately 8,871 lines and
  combines provider calls, prompts, task state, validation, retries, reasoning,
  deterministic repair, domain terminology, and finalization.
- `_validate_preserved_tokens` is approximately 903 lines with cyclomatic
  complexity around 155.
- `_chinese_boundary_signal` is approximately 584 lines with complexity around
  85.
- `_validate_cross_key_boundaries` is approximately 462 lines with complexity
  around 75.
- `_validate_natural_chinese_contextual_constructions` is approximately 497
  lines with complexity around 73.
- `assess_english_boundary` in `subforge/core/split/boundary.py` still spans
  1,109 lines after behavior-preserving family extraction and is shared by
  splitting, translation audit, and recovery paths.
- Some production repair branches recognize exact source passages and write
  fixed Chinese sentences. These protect known examples but can overwrite a
  correct translation in an unseen, superficially similar context.
- Validation returns human-readable strings that are later parsed by the
  finalizer to choose repair behavior. Message wording is therefore coupled to
  control flow.
- Many tests call private translator methods and assert exact historical fixes.
  This protects legacy behavior but does not independently establish quality on
  unseen videos.

## Current Evaluation Inventory

Phase 0 discovered and hash-validated 13 complete local triplets:

- 56,188 source or word-level cues;
- 5,288 machine-result cues;
- 5,289 human-edited gold cues;
- 12 machine/gold pairs with identical keys, timing, and source text;
- one airport sample explicitly marked `requires_alignment` because the human
  result contains timing edits;
- seven development samples, one validation sample, and five initial holdout
  samples;
- explicit speaker metadata covering two-speaker dialogue and three-to-eight
  speaker material;
- one approximately 89-minute English/Japanese mixed-language sample;
- automotive and infrastructure domains.

The corrected deterministic static baseline found no placeholder, empty, or
reasoning-leak entries in the stored machine results and one source-copy risk.
An earlier evaluator draft incorrectly matched the character `略` inside normal
words such as `略便宜` and `战略`; a regression test now prevents that false
positive. Human editors changed 4,068 of the structurally comparable machine
cues. One human file uses UTF-16LE and is intentionally read without rewriting
the source artifact.

Important limitations remain:

- most historical model and algorithm-version metadata is unknown;
- samples without explicit speaker labels remain `speaker_mode: unknown` until
  their provenance is confirmed;
- the initial holdout is rich in dialogue and multi-speaker infrastructure
  material but does not yet contain an uncontaminated automotive, mixed-language,
  or over-one-hour sample;
- detailed holdout cue indices are redacted from generated reports;
- the corpus authorizes evaluation and behavior-preserving refactoring, not a
  production algorithm replacement.

## Required Evaluation Corpus

Before changing production output behavior, collect at least 12 to 15 independent
source/machine/human triplets and 2,500 human-reviewed subtitle cues:

- four single-speaker videos from more than one subject domain;
- three two-speaker conversations;
- two videos with three to five speakers;
- two mixed-language videos;
- two videos longer than one hour;
- representative recovery cases containing empty translations, misplaced
  clauses, repeated translations, entity errors, and incomplete speech timing.

Reserve three to five complete videos as a holdout set. They must never be used
to design rules or prompts.

Each sample should have a manifest recording source media, source subtitle,
machine result, human result, speaker mode, languages, domain, model, algorithm
version, configuration, removed advertisement ranges, and known limitations.

## Target Architecture

### Immutable Task Context

Introduce a `TranslationSession` containing read-only source text, speakers,
languages, timing gaps, and task configuration. Precompute a `CueFeatures` object
for every cue so numbers, entities, scripts, tokens, and boundary features are
not repeatedly extracted by separate validators.

### Typed Diagnostics

Replace `(bool, message)` validation responses with structured records:

- `rule_id`;
- category and severity;
- confidence;
- affected cue keys;
- evidence;
- recommended repair strategy.

A `ValidationReport` should return all detected issues in one pass. Human-readable
messages must never be parsed to control program behavior.

### Modular Quality Rules

Move validators into focused modules:

```text
subforge/core/translate/quality/
  schema.py
  language.py
  numbers.py
  entities.py
  english_boundary.py
  chinese_boundary.py
  dialogue.py
  fluency.py
  registry.py
```

Rules should be separated into:

- hard invariants for keys, timing, empty output, numbers, and entities;
- general linguistic detectors for subjects, clauses, modifiers, and cue
  ownership;
- optional domain guidance packs that provide terminology and context without
  writing fixed full-sentence translations.

### Repair Planner

Aggregate diagnostics by contextual window and choose the least expensive safe
repair:

- local deterministic repair for formatting, punctuation, and exact numeric
  equivalence;
- non-reasoning LLM repair for ordinary fluency defects;
- reasoning only for confirmed semantic displacement, complex Chinese word
  order, entity ambiguity, or cross-cue ownership;
- accept a candidate only when every hard invariant passes and its quality score
  improves;
- impose per-task and per-rule repair budgets and require repairs to be
  idempotent.

### Explicit Mode Policies

Share hard invariants while keeping behavioral policy separate:

- `MonologuePolicy` may use wider same-speaker context and allow an omitted
  subject only when its referent is unambiguous;
- `DialoguePolicy` treats speaker boundaries as hard by default and exposes
  adjacent turns as read-only context;
- `MixedLanguagePolicy` selects language-specific alignment, terminology, and
  boundary rules without changing the user's explicit source-language choice.

### Boundary Rule Registry

Keep the existing dynamic-programming segmentation strategy, but replace the
large boundary condition chain with declarative rules containing stable IDs,
weights, applicability conditions, and exceptions. Extract boundary features
once and make every score contribution observable in diagnostics.

## Migration Sequence

1. Build the corpus manifest, evaluator, and baseline metrics without changing
   production output.
2. Add immutable session and precomputed cue features behind the legacy path.
3. Introduce typed diagnostics and adapt legacy validators without changing
   their decisions.
4. Migrate schema, empty-output, number, and entity rules first.
5. Add the repair planner and compare it with the legacy retry loop in shadow
   mode.
6. Separate monologue, dialogue, and mixed-language policies.
7. Migrate English and Chinese boundary rules last.
8. Remove exact case-specific production replacements only after the holdout
   corpus proves their general replacement is at least as safe.
9. Delete legacy code only after several complete releases can fall back to the
   new path without quality regression.

Use a feature flag such as `quality_pipeline_v2` during migration. A task must be
able to fall back to the legacy pipeline until final acceptance.

## Acceptance Gates

The new path may replace the legacy path only when all of the following hold:

- subtitle keys and timing integrity: 100%;
- empty translations and placeholder text: zero;
- severe semantic regressions on the holdout set: zero;
- no measurable regression in speaker ownership or mixed-language preservation;
- pairwise human preference is no worse than the legacy result;
- costs and latency meet the prospectively frozen track-specific budget, with
  baseline variability and observed quality benefit reviewed separately;
- reasoning is restricted to demonstrably high-risk windows;
- no production rule contains a complete fixed translation for one historical
  source passage;
- orchestration modules remain small enough to review, with ordinary rule
  functions targeting cyclomatic complexity no greater than 15.

## Next Safe Step

Keep production on legacy until the relevant admission route is reviewed.
Reproducible local validator defects may be isolated as bugfix revisions without
bundling them with speculative LLM repair behavior. Quality candidates need
independent, mode-scoped semantic review; efficiency candidates need quality
non-inferiority and repeatable resource gains. Use the schema-2 policy screening
in `scripts/translation_quality/admission.py` and record the policy hash before
testing. Screening never authorizes rollout.

Historical rejection is evidence, not a permanent ban on a problem category.
Reconsideration needs a new mechanism or new causal evidence and a new frozen
revision; do not silently restore old candidates or tune against the completed
holdout. A budget overrun or non-activation can justify observation rather than
deleting useful isolated code and regression tests. Full pipeline replacement
still requires blind evaluation and gradual rollout.

Phase 8 isolation plumbing now uses the default-off
`SUBFORGE_QUALITY_PIPELINE_V2` flag and requires an explicit
`SUBFORGE_QUALITY_PIPELINE_REVISION` whenever it is enabled. Legacy runs retain
their exact `_processed` and `_recovery` names and existing cache keys. Candidate
runs include the normalized revision and task ID in both output names, refuse to
overwrite an existing candidate artifact, identify the pipeline in task results,
and append a dedicated namespace to split, optimization, context, LLM, and
translator-result caches. No UI control exposes the flag.

The static comparison command writes text-free legacy/candidate JSON and Markdown
reports, rejects mismatched sample order, and verifies a source/gold comparison
identity that intentionally excludes the machine-output path. Hard gates cover
structure, empty or placeholder output, reasoning residue, source copies,
untranslated text, and adjacent duplication. A self-comparison of the frozen
13-sample report passes every gate with zero metric delta. The aggregate report,
English boundary snapshot, and Chinese boundary snapshot remain byte-identical to
their frozen baselines.

Phase 8 efficiency telemetry is now task-scoped and attached only to an explicit
LLM client. Candidate runs write a separate, non-overwriting `.telemetry.json`
sidecar containing request attempts, provider successes and failures, API and
wall duration, retry waits, token families, reasoning modes, cache state, model
and stage totals, pipeline revision, and a source-file workload hash. It stores
no prompts, responses, subtitle text, credentials, URLs, or cue keys. Aggregation
rejects duplicate tasks, mixed pipeline identities, mixed cache states, missing
workload identity, and mixed repair-shadow availability. Comparison hard gates
limit successful calls, attempts, tokens, and wall time to 105% of legacy and
forbid any increase in reasoning-enabled requests.

The bounded repair recorder now also emits an immutable, text-free summary of
planner dispositions, strategies, reasoning modes, diagnostic rule counts, and
planner-versus-legacy routes. Unique plan and comparison storage are capped.
Candidate admission fails when no repair plan was exercised, comparison coverage
is incomplete, or any observation was dropped. Existing legacy actions are
observed at main-batch retry, alignment repair, locked-batch recovery, single-item
fallback, and Chinese fluency-repair exits without executing the planner or
adding model calls. At that structural checkpoint, the full non-integration
suite passed 2,273 tests; Ruff and Pyright passed; all frozen static and boundary
artifacts remained byte-identical.
The unchanged `phase8-shadow-baseline` run then froze seven development samples
at DeepSeek V4 Flash concurrency 20 and batch size 20. It completed six files and
one recovery file, used 2,436 requests and 5,453,029 tokens in 1,118,935 ms, and
recorded 137 fully compared repair observations with none dropped. The largest
fully observed mismatch was `local_rewrite/disabled -> retry/disabled`.

The first behavioral revision, `phase8-local-preservation-repair-v1`, replaced
only that route with one non-reasoning item repair and retains the exact legacy
batch retry when local validation fails. On the identical workload it again
completed six files and one recovery file, introduced no new hard static failure,
used 2,403 requests and 5,268,567 tokens in 1,097,242 ms, and reduced reasoning
tokens from 356,804 to 304,275. Forty-three observed plans executed the intended
local route, all 193 comparisons were retained, and no observation was dropped.
The automated development quality and efficiency gates accepted the revision,
so it advanced to one blind holdout evaluation under its frozen candidate
revision. It was never enabled in production.

This admission is not a production replacement decision. Fresh model runs may
produce different subtitle segmentation, so machine-to-gold cue-change counts
are not comparable across these two runs even though source/gold workload hashes
match. Both runs retained the same Bentley recovery failure. Inspection showed
that valid localized quantities and source-supported ASR canonical corrections
can still be rejected by literal token ownership checks. Those validator families
must be evaluated separately rather than broadening the admitted repair.

An attempted follow-up based on explicit context mappings was rejected before a
full corpus run. Focused tests passed, but the real Bentley task still rejected
`Naim` because that run supplied no mapping that activated the candidate's
assumed contract. The run was stopped after Bentley and the behavior code was
removed. This is evidence that context production and context consumption must
be measured before another ownership rule is designed.

The next observation-only checkpoint now measures that contract without storing
subtitle text, terminology text, cue keys, names, prompts, responses, URLs, or
credentials. It records only bounded integer counts and runs only for isolated
candidate tasks. A fresh Bentley run under the development-admitted revision at
that checkpoint, `phase8-local-preservation-repair-v1`, produced 48 formatted terminology
lines: five were ASR-labelled, one was parseable by the current canonical-name
consumer, that mapping matched one source cue, and document-support validation
rejected it. The run completed rather than producing a recovery file, but used
244 requests, 631,932 tokens, and 87,449 reasoning tokens. Model and segmentation
variance make that completion unsuitable as quality or efficiency admission
evidence. The valid conclusion is narrower: at least one real context correction
reaches the consumer and is then lost at the ownership gate, while four other
ASR-labelled lines are not Latin canonical mappings under the current parser.
No ownership behavior has been changed.

Four structured-ASR ownership revisions were then evaluated only on Bentley and
rejected before a seven-sample expansion. The first run parsed five mappings but
supported only one and failed after correct canonical names were rejected by
downstream ownership checks. A second run emitted no usable mappings. A third
run classified all eight source matches as supported and corrected one target
name, but still ended in recovery. The fourth run completed, yet classified zero
of three source matches as supported and omitted the same target name; it also
used 262 requests, 676,870 tokens, and 91,881 reasoning tokens. This run-to-run
instability fails the route-activation and efficiency gates even though focused
tests passed. All structured schema, entity-slot ownership, and target-script
exception behavior was removed. The text-free evidence collector remains, the
admitted `phase8-local-preservation-repair-v1` revision remains isolated, and no
holdout or production path was changed.

A separate `phase8-local-preservation-numeric-equivalence-v2` revision then
fixed one deterministic validator defect: formatting an integral ten-thousands
value with `rstrip("0")` turned `30` and `40` into `3` and `4`, so valid `30万`
and `40万` translations were rejected for source values `300000` and `400000`.
Offline replay proved that the candidate removed only those two Bentley false
positives, and a Bentley-only live run reduced the recovery diagnostics from
three to two. The full seven-sample development run completed every sample and
improved hard static failures from eight to six, with empty and placeholder
outputs both improving from one to zero. It nevertheless used 1.0526 times the
tokens, 1.0720 times the wall duration, and 1.0504 times the reasoning-enabled
requests of the frozen admitted run. Only total requests, at 1.0350 times the
baseline, passed the five-percent efficiency gate. The revision therefore was
rejected, its behavior and tests were removed, and its isolated artifacts were
retained. Do not admit the rule by weakening the frozen gate; a future numeric
candidate needs lower-variance causal evidence or a cheaper activation path.

The next candidate attempted to replace a full-batch retry only when an otherwise
valid response omitted exactly one key. Historical v1 telemetry contained two
`schema.missing_key` observations across two samples, but the isolated live run
on the smaller affected canal sample produced no missing-key diagnostic. It
completed with 157 requests, 308,025 tokens, and 107,582 ms wall duration, but
none of that workload exercised the candidate route. Expansion stopped at one
sample, all missing-key behavior and tests were removed, and the isolated run
was retained only as non-activation evidence. Sparse provider-format variance is
not a suitable next behavior target without a deterministic replay harness.

A bounded untranslated-output candidate was then selected because Bentley and
the multi-speaker Topher sample repeatedly emitted that diagnostic across fresh
runs. The Bentley live run activated the route, but its local rewrite remained
Latin-only and was rejected again. Inspection showed the affected cue was a
short list of vehicle model names, and the human-edited gold also retained those
names in Latin script. The diagnostic was therefore a model-caption false
positive rather than omitted translation. The run remained a three-item recovery
and expansion stopped. The candidate behavior and tests were removed. At that
checkpoint only, a behavior-preserving helper shared by the then-admitted local
preservation route was retained; the final Phase 9 rollback later removed it
after it had no remaining caller. A future target-script candidate must first distinguish neutral
identifier lists from ordinary untranslated prose with strict negative tests.

That classifier was subsequently implemented as the isolated
`phase8-identifier-caption-exemption-v1` revision and tested against the same
seven development samples. It removed the Bentley false positive and reduced
hard static failures from eight to six, including the one empty and one
placeholder result. However, only one of 3,680 inspected cues exercised the
exemption, while the full run introduced one real adjacent duplicate in the
mixed Japanese/English sample. Compared with the admitted v1 run, total tokens
rose 5.27%, wall time rose 11.51%, and reasoning-enabled requests rose 12.23%; all
three exceed the frozen gate. The candidate was rejected and its feature flag,
runtime behavior, text-free activation telemetry, evaluator exception, and tests
were removed. Its isolated run and comparison reports remain as evidence. The
then-admitted local-preservation candidate and production legacy path were
unchanged at that checkpoint.

Phase 9 subsequently ran all five frozen holdout samples twice with DeepSeek V4
Flash, concurrency 20, batch size 20, reflection enabled, and identical explicit-
client no-disk-cache conditions. The runner required an explicit blind-holdout
flag, forbade sample selection, suppressed task logs at the file-descriptor
level, and reported aggregate-only results. The unchanged behavior used 1,141
requests, 2,387,732 tokens, 31 reasoning-enabled requests, and 693,624 ms. The
frozen local-preservation candidate used 1,251 requests, 2,578,472 tokens, 38
reasoning-enabled requests, and 731,839 ms. Both had zero empty targets,
placeholders, reasoning leaks, source copies, untranslated targets, and adjacent
duplicate risks, with no hard-quality delta. Candidate requests rose 9.64%,
tokens 7.99%, wall time 5.51%, and reasoning-enabled requests 22.58%; every
efficiency gate failed. The candidate was rejected without threshold tuning and
all candidate-only runtime behavior and tests were removed. The blind reports
remain isolated evidence and must not be opened for rule design.

Final post-rejection verification passes all 2,289 selected non-integration tests
with 35 external integration tests deselected. Ruff, Pyright, and `git diff
--check` pass, and no local-preservation revision switch, factory argument,
runtime repair function, or candidate-only test remains. Phase 10 was not entered
because the Phase 9 acceptance gate failed. Production therefore remains on the
unchanged legacy behavior; a future behavioral attempt must start as a new
development-only candidate backed by new evidence.

Phase 4 moved the large
numeric/entity equivalence implementation and its shared alias tables out of
`LLMTranslator` into `quality/preservation.py`; the old method is now a thin
compatibility adapter. Schema, key completeness, empty output, placeholders,
target script, reasoning residue, source-copy metrics, typed numeric/entity
findings, source/timeline write-back integrity, and deterministic Chinese
punctuation finalization all have focused entry points.

The Phase 4 aggregate static metrics match the Phase 3 baseline exactly across
all 13 local samples. Phase 5 records bounded, text-free plans
for typed hard failures, including strategy, affected keys, context radius,
reasoning eligibility, attempt limit, required hard rules, and legacy fallback.
It uses immutable session mode and repeated-failure history, then aggregates
planner-versus-legacy decisions. It neither executes repairs nor issues model
requests.

Phase 6 now composes immutable monologue/dialogue and unknown/single/mixed-language
dimensions into an explicit mode policy. Existing prompt-family selection and
metadata guidance delegate to that policy without changing their text, retries,
reasoning, token use, or output. The current non-integration suite passes 1,907
tests; Pyright and Ruff pass; all 13 local samples remain byte-identical to the
Phase 3 aggregate baseline.

The initial Phase 7 detector-extraction checkpoint registered 74 translation
boundary diagnostics and all 149 English score rules across 151 call sites. A
static source auditor reported zero anonymous legacy score sites, zero unpaired
score/reason branches, and zero unresolved rule IDs. Immutable English boundary
features and focused numeric, comparison, entity, coordination, discourse,
predicate, and grammar detectors captured every English condition and first
reduced `assess_english_boundary` from 1,925 to 1,109 lines. At that checkpoint,
a text-free snapshot covered 5,275 adjacent boundaries in all 13 samples: all
1,938 active risk points were attributed, unregistered active risk was zero, and
the final decision hash remained unchanged. The non-integration suite passed
2,075 tests; Pyright reported zero errors; Ruff passed; the aggregate static
baseline remained byte-identical; and no additional model request was issued.

The 151 registered applications are now executed by six explicitly ordered
stages: foundation, relations, completions, discourse, clause ownership, and
dependencies. The source auditor follows that declared plan, rejects registered
calls outside it, and freezes the exact stage order. `assess_english_boundary`
is now a 28-line coordinator rather than a scoring monolith. After this change,
2,076 non-integration tests pass; Pyright reports zero errors; Ruff passes; both
the aggregate static baseline and the 5,275-boundary snapshot remain byte-identical;
and the frozen decision hash remains
`51479e015de80013d81e89d29e74b4d4caeb300dbf7d14e0f07c557c057fdf1d`.
No model request was issued.

Chinese boundary migration remains part of Phase 7. Phase 8 must not start until
the complete Phase 7 exit gate is met.

The Chinese migration inventory now covers 74 registered source, target, and
display signals, 104 literal signal branches, and all 9 direct signal call sites.
There are no unknown emitted messages, unreferenced definitions, or calls outside
the approved quality-flow closure. The original position-coupled inventory
fingerprint was
`830103a49d098d7ec2b6c799082022511fc9b869c59267cacea639ee48abeaa7`.
The migration-safe semantic inventory hash, recomputed against the pre-migration
source and preserved after extraction, is
`ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`;
the current physical layout hash is
`d58eb0d3a11e91af294bc5ca7edbc0fe1b58d2b717cb59c3415311b69f05140c`;
and the call-flow hash remains
`a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`.
A text-free snapshot separately covers 5,275 machine boundaries and 5,276 human
gold boundaries across all 13 samples. Machine output emits 345 target candidates;
gold output emits 288; neither side contains an unregistered source or target
signal. Repeated generation is byte-identical. After adding these gates, 2,079
non-integration tests pass, Pyright reports zero errors, Ruff passes, and the
aggregate static quality baseline remains byte-identical. No model request or
production-path behavior changed.

The first Chinese production migration checkpoint introduces immutable
`ChineseBoundaryFeatures` and uses it for both the main syntax signal adapter and
the visible-pause path. The two visible-pause rules now live in a pure detector
module and resolve stable registry IDs while `_long_gap_chinese_boundary_signal`
remains a thin compatibility adapter. First-match behavior, messages, thresholds,
candidate selection, repair policy, and provider behavior are unchanged. The
migration-safe semantic inventory hash and call-flow hash above remain identical;
the physical layout hash is tracked separately so moving code cannot masquerade
as a semantic change. The 5,275-machine/5,276-gold Chinese boundary snapshot and
the 13-sample aggregate baseline are byte-identical to their frozen inputs.
After this checkpoint, 2,085 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The next production checkpoint moves the first five contiguous syntax branches
into a pure foundation detector: standalone connectives, demonstrative subjects,
sentence adverbs, subject-plus-adverb tails, and coordinated subjects stranded
from their predicates. The legacy function calls that detector once at the same
position and returns the same registered message. Its semantic inventory hash
remains
`ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
its call-flow hash remains
`a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
and its new physical layout hash is
`a99fc8712ad8cd7bd709212259e4faccd98af7830a6fac68a69f6edd06e5f7d7`.
The machine/gold boundary snapshot and aggregate baseline remain byte-identical.
After this checkpoint, 2,096 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The following nominal-attachment checkpoint moves nine more contiguous branches:
relative clauses and their head nouns, demonstrative relatives, comparative
objects, comparison frames, vehicle model modifiers, reporting complements,
classifiers, demonstrative modifiers, and contextual count classifiers. It stops
before `把` and disposal constructions. The semantic inventory and call-flow
hashes remain unchanged; the new physical layout hash is
`436abe430ddbda377d1ec1024821b06e9d9ee3c794dd5bbaa9773678675f01f7`.
The machine/gold boundary snapshot and 13-sample aggregate remain byte-identical.
After this checkpoint, 2,106 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The governing-attachment checkpoint moves five more contiguous branches:
`把` constructions, general disposal constructions, required predicate
complements, locative phrases, and standalone temporal phrases. Existing
predicate-present and complete-passive exceptions remain in the same order. The
semantic inventory and call-flow hashes remain unchanged; the new physical
layout hash is
`22d910b161709554b52b3a41b3ffa13f22ac0cd8b7b16f4f0e2c515ee0f2af96`.
The machine/gold boundary snapshot and 13-sample aggregate remain byte-identical.
After this checkpoint, 2,114 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The surface-fluency checkpoint moves eleven more contiguous branches: explicit
boundary duplication, superlative attachment, literal Japanese difficulty
phrasing, duplicated construction nominalization, stacked connectives, canonical
boundary overlap, bigram/sequence similarity, repeated short/meaning units,
repeated predicates, and repeated locatives. The similarity thresholds and
first-match order are unchanged. The semantic inventory and call-flow hashes
remain unchanged; the new physical layout hash is
`c87c6ee41b443253b77e1900763c34a44aae3add830997f25671a2f6fe8aedb0`.
The machine/gold boundary snapshot and 13-sample aggregate remain byte-identical.
After this checkpoint, 2,126 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The subject-attachment checkpoint moves three more contiguous branches: material
subjects followed by predicate continuations, existential people subjects, and
coordinated modifiers. The object-role exclusion and terminal-punctuation guard
remain unchanged. The semantic inventory and call-flow hashes remain unchanged;
the new physical layout hash is
`8b5fdcd6b3c62fc53da297d023f5d4c620483d10693da0c35de026258e38d561`.
The machine/gold boundary snapshot and 13-sample aggregate remain byte-identical.
After this checkpoint, 2,132 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The discourse-bridge checkpoint moves three more contiguous branches: a
connective stranded at the preceding subtitle end, an `在于` bridge separated
from its complement, and a topic noun separated from a following `是` clause.
The exact first-match order and punctuation guards remain unchanged. Across the
main syntax path, 36 rules now live in pure detectors, in addition to the two
visible-pause rules. The semantic inventory and call-flow hashes remain
unchanged; the new physical layout hash is
`5545b371f6c9ce926f4d61629e74464d44f99ffc651f95544ecec2f91420847a`.
The machine/gold boundary snapshot and 13-sample aggregate remain byte-identical.
After this checkpoint, 2,137 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The unfinished-frame checkpoint moves five more contiguous branches: special
unfinished grammatical tails, reporting frames, embedded locative frames,
pronoun-plus-`把` frames, and copular frames separated from their result. Across
the main syntax path, 41 rules now live in pure detectors, in addition to the two
visible-pause rules. Exact regular expressions, first-match order, messages, and
fall-through behavior remain unchanged. The semantic inventory and call-flow
hashes remain unchanged; the new physical layout hash is
`8b63337d27e5e63ecf990c0c1f621a3ebc794619c202e8c611bdcfcb1375c341`.
The machine/gold boundary snapshot and 13-sample aggregate remain byte-identical.
After this checkpoint, 2,143 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The reason-construction checkpoint moves the single immediately following
`之所以……原因` branch into a pure detector. The earlier copular-bridge match for
an `原因 / 是我们……` boundary remains explicitly protected by a precedence test.
Across the main syntax path, 42 rules now live in pure detectors, in addition to
the two visible-pause rules. The semantic inventory and call-flow hashes remain
unchanged; the new physical layout hash is
`2876b7839958fad8adccf989c8a0db7eeae4f84d3b979fce3e7608833f258eae`.
The machine/gold boundary snapshot and 13-sample aggregate remain byte-identical.
After this checkpoint, 2,148 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The completion-frame checkpoint moves four more contiguous branches:
percentage-use predicates, resultative predicates, classifier phrases, and
comparison examples. The complete `的车/车辆/东西` noun-phrase exclusion remains
unchanged and has a dedicated negative test. Across the main syntax path, 46
rules now live in pure detectors, in addition to the two visible-pause rules.
The semantic inventory and call-flow hashes remain unchanged; the new physical
layout hash is
`df84678f7bdd891ca5168b828703ca95e7c823d8721eb4e0bfce867a34f32d36`.
The machine/gold boundary snapshot and 13-sample aggregate remain byte-identical.
After this checkpoint, 2,154 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The numeric-completion checkpoint moves two more contiguous branches: split
numeric ranges and stranded numeric complements. Both left/right conditions,
Arabic and Chinese numeral sets, classifier scope, and continuation words remain
unchanged. Across the main syntax path, 48 rules now live in pure detectors, in
addition to the two visible-pause rules. The semantic inventory and call-flow
hashes remain unchanged; the new physical layout hash is
`c74ce1450c522e6ced99b50122f2af47bb5545ad4623d97662a4a6b1bc29ffb9`.
The machine/gold boundary snapshot and 13-sample aggregate remain byte-identical.
After this checkpoint, 2,160 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The consequence-predicate checkpoint moves the single immediately following
missing-consequence branch into a pure detector. Its trigger and complete-
predicate exclusion remain colocated, preserving the distinction between a bare
`因此两个成果` result and a complete `因此产生了两个成果` clause. Across the main
syntax path, 49 rules now live in pure detectors, in addition to the two
visible-pause rules. The semantic inventory and call-flow hashes remain
unchanged; the new physical layout hash is
`c0c5838e0843fb46afae5c7dcd8d616f78e65d41435d95a5b4f0537a8e63e87a`.
The machine/gold boundary snapshot and 13-sample aggregate remain byte-identical.
After this checkpoint, 2,165 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The semantic-attachment checkpoint moves three more contiguous branches:
incomplete semantic frames, reporting frames, and stranded nominal modifiers.
Exact keyword sets, character spans, messages, and first-match order remain
unchanged. Across the main syntax path, 52 rules now live in pure detectors, in
addition to the two visible-pause rules. The semantic inventory and call-flow
hashes remain unchanged; the new physical layout hash is
`facd6630d7a26fa46bc0977693efa32f1eaa8672a32b8cf4458d09a6df166ba4`.
The machine/gold boundary snapshot and 13-sample aggregate remain byte-identical.
After this checkpoint, 2,171 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The unfinished-predicate checkpoint moves the single immediately following
generic unfinished-predicate branch into a pure detector. Its trigger set and
nominal-attempt exclusion remain colocated, preserving the distinction between
an incomplete `希望能够/旨在/尝试` predicate and a complete noun phrase such as
`一项大胆尝试`. Across the main syntax path, 53 rules now live in pure detectors,
in addition to the two visible-pause rules. The semantic inventory and call-flow
hashes remain unchanged; the new physical layout hash is
`2aa15b3fd73b76be114f8d47f186ecdbad98fabf57adc943ae8377bf9f870d88`.
The machine/gold boundary snapshot and 13-sample aggregate remain byte-identical.
After this checkpoint, 2,176 non-integration tests pass, Pyright reports zero
errors and zero warnings, Ruff passes, and no model request was issued.

The Phase 7 completion sweep moves every remaining Chinese target-boundary branch
into ten focused detector families while preserving the original first-match
order. The checkpoints completed after unfinished predicates were:

| Checkpoint | Sites moved | Layout hash | Passing tests |
| --- | ---: | --- | ---: |
| Predicate completion | 2 | `b189d65d12659cd59bc8b5404b48b8ee6fc84e81a9a32d620da6422546985592` | 2,180 |
| Temporal/locative attachment | 4 | `afc7c95be5981f0fe2f68c2e9ac9636e2f14cbb70f6f30624253115f19e84252` | 2,186 |
| Incomplete nominal frames | 4 | `7482820ba23a2e50bfe9763219e8596073e9a4da61df9b6ba85ec0af10b32472` | 2,192 |
| Semantic completion | 5 | `a1df5a565735cebf55388fe105b561b40b2be5a30128691e3ec48935578670eb` | 2,199 |
| Clause attachment | 5 | `b27b08046679caa549d44df99fa3ff304ab89b5c23fc85dcceea4dc11e4ba5b5` | 2,206 |
| Subject/nominal completion | 3 | `f97da7c1d101def4aa0ff59227f0c6acd2ffdb969fc182fbb123678571ce64f5` | 2,211 |
| Structural tails | 5 | `292f4bed8649a09d40d1b43134671a3631e1ca46de705d6f793c4f3e55599def` | 2,218 |
| Late structural frames | 3 | `2bf7d48aa5a0e12a20d2938791f92fada56f1cce27a5f07668a5908dd440477c` | 2,223 |
| Adverb/pronoun attachment | 6 | `c5fe25f7bc9c4e0cdb384bb518606ff9ae8d6c33b2f685367257c0eacecb13bd` | 2,230 |
| Terminal tokens | 4 | `60ffbb2846333f2b755fdb2999db13d22822c7e4a17b02fd7c3604bb961c1a1e` | 2,235 |

All 94 main-path message sites now live in pure detectors, in addition to the two
visible-pause sites. An immutable ordered registry replaces 23 repeated adapter
branches, removes the complexity waiver from `_chinese_boundary_signal`, and adds
one explicit precedence-contract test. The final suite passes 2,236
non-integration tests. The semantic inventory hash remains
`ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
the call-flow hash remains
`a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
and the final physical layout hash is
`60ffbb2846333f2b755fdb2999db13d22822c7e4a17b02fd7c3604bb961c1a1e`.
The frozen machine/gold boundary snapshot and all 13 static samples remain
byte-identical. Pyright reports zero errors and warnings, Ruff passes, and no
model request was issued.
