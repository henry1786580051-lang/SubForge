# Translation Quality Refactor Execution Guide

Status: Phases 0-7 complete; historical Phase 8/9 candidates rejected; legacy main pipeline with subsequent local fixes. Local app updated on 2026-09-05; full-pipeline quality and final-subset efficiency acceptance remain unproven.

Last validated: 2026-09-05

Development bugfix checkpoint (2026-09-05): see
[`elantra-translation-validation-fix.md`](elantra-translation-validation-fix.md).
The new development triplet led to narrow fixes for hyphenated numeric modifiers,
suffix-based adjective false positives at explicit clauses, source-local DCT
localization, and translated-input guard ordering. Four hash-validated development
videos were replayed offline; the full suite passes 2,652 non-integration tests.
The subsequent user-authorized GLM-5.3-Flash pair completed both full runs. The
numeric-modifier split improved, but tokens rose 15.88% and the frozen bugfix
screening returned `observe`. Full output review still found English residue,
source/target ownership errors, and semantic mistakes. This is not an overall
quality or efficiency acceptance. At that checkpoint no holdout inspection or
packaged rollout had been performed; the later local app update is recorded below.
See the checkpoint document for live evidence and limitations.

Local validation efficiency follow-up (2026-09-05): see
[`low-token-validation-fixes.md`](low-token-validation-fixes.md). Exact RPM
equivalence, stable missing-token order, identical-success reuse, and narrow
closed-boundary exclusions remain, with 2,690 non-integration tests passing.
Two formal GLM v2 pairs reduced aggregate tokens by 2.39%, but semantic review
failed and a new midpoint-price acceptance was withdrawn. That cost result does
not certify the final reduced subset, which was rechecked offline. All five
full runs (including one preserved pilot) completed. At experiment close there
was no packaged rollout or whole-pipeline quality acceptance. The subsequent
local build does not change the experiment's semantic rejection.

Local delivery follow-up (2026-09-05): the user subsequently requested an app
update. `dist/SubForge.app` 1.2.1 was rebuilt and opened; signing, matching all
eight changed/new algorithm modules to source, and packaged backend/media/
Apple Silicon runtime checks passed. The previous app is archived for rollback.
Evidence: `artifacts/desktop-update-20260905/result.json`. This is a local
installation, not a public release, new GLM evaluation, or quality acceptance.

Documentation amendment (2026-09-05): `优化指南.md` version 2.2 is the current
operating contract. Sections 8.1, Step 7/8.1, 10.2.1, and 12-15 add exact
version/evidence binding, component and combination checks, two-sided evidence
for validator relaxations, staged evaluation, and separate delivery records.
These are prospective reporting and execution requirements, not a new automated
admission schema; historical policies and decisions remain frozen.

Prospective admission amendment: 2026-09-01. `优化指南.md` sections 10-12 are the
current executable admission contract. They supersede blanket 5% caps and
automatic candidate deletion for future experiments, not the historical trial
records below. The three routes are bugfix, quality, and efficiency. Static
screening uses a frozen schema-2 policy; its review/observe/blocked result is not
semantic acceptance. No production behavior or historical rollout decision is
changed by this amendment.

Development bugfix checkpoint: 2026-09-03. Two newly supplied development
triplets were frozen under
`artifacts/translation-quality/runs/20260903-megastructure-airport-triplets/manifest.json`
before inspection. The review admitted only source-anchored, reusable defects:

- an English noun separated from a following postpositive participial modifier;
- a scalar degree modifier separated from its comparative complement;
- non-agentive natural-event causation mistranslated with an intentional attack verb;
- Chinese causative frames and coordinating conjunctions stranded at cue endings;
- equivalent discourse connectors duplicated across a boundary;
- negative-auxiliary ellipsis rendered as an incomplete Chinese predicate.

No sample title, project-specific proper-name map, or reference-only stylistic
preference was added. The airport reference contained 13 source corrections and
the Gemini reference included unsupported embellishments, so neither was treated
as literal truth. Focused tests include terminal-punctuation, intentional-agent,
completed-idiom, and unrelated-context exclusions.

The authorized GLM-5.3-Flash development run used concurrency 20 and batch size
20. Both full samples completed without empty targets, placeholders, reasoning
leaks, untranslated targets, task failures, or rate-limit retries. English
boundary risk fell to zero. A second full run of the affected multi-speaker
sample reduced detected Chinese target-boundary signals from 16 to 11; the
reference has 5 after the same false-positive cleanup. Selective native reasoning
remained sparse (8 enabled requests out of 173 total requests). The remaining
soft-boundary and stylistic differences were not converted into deterministic
rules. This checkpoint follows the narrow bugfix route and does not reopen or
reuse the completed blind holdout.

Primary plan: `docs/dev/translation-quality-refactor-plan.md`

## 1. Purpose

This guide is the operational manual for executing SubForge's translation-quality
refactor. It converts the archived architecture plan into a staged, measurable,
reversible implementation process.

The executor must use this guide before editing the production translation,
splitting, validation, or repair pipeline. The guide exists to prevent three
recurring failure modes:

1. changing prompts, orchestration, and rule semantics in the same patch;
2. optimizing against one historical subtitle while silently degrading unseen
   material;
3. deleting legacy behavior before a replacement has passed an independent
   baseline and holdout evaluation.

The immediate objective is not to produce a new translation algorithm. The
immediate objective is to make every future quality change observable, testable,
and reversible.

## 2. Current Starting Point

At the start of the refactor:

- `subforge/core/translate/llm_translator.py` contains about 9,385 lines;
- `subforge/core/split/boundary.py` contains about 2,913 lines;
- the local evaluation collection contains 13 complete three-file groups;
- each group contains a source or word-level SRT, a machine result, and a
  ChatGPT-edited result;
- the human-edited results contain about 5,289 subtitle cues in total;
- the numerical corpus threshold in the architecture plan has been reached;
- corpus provenance, mode coverage, advertisement removal, timing edits, and
  holdout assignment had not yet been formalized in a manifest;
- `TranslationSession`, `CueFeatures`, and typed diagnostics had not yet been
  introduced;
- the working tree contains unrelated user changes and must be treated as
  valuable state, not reset or overwritten.

The current implementation has since completed the Phase 0-3 foundations:

- an ignored, hash-validated local manifest classifies 13 complete triplets and
  freezes development, validation, and holdout assignments;
- the deterministic evaluator records structural, boundary, gold-comparison,
  and efficiency evidence without rewriting source artifacts;
- immutable `TranslationSession` and `CueFeatures` snapshots are available to
  new quality code;
- typed boundary diagnostics run beside the legacy message adapters, with parity
  tests covering the complete registered message surface;
- the legacy production path remains the default and no full pipeline
  replacement has been authorized.

Two narrowly scoped boundary defects discovered by the authorized full-model
validation were fixed independently of the structural migration: English
determiner/degree-modifier splits and source-supported Chinese nominal modifiers
stranded at a cue ending. Both changes have focused regression tests and preserve
keys, timing, source text, and existing fallback behavior.

Phase 7 is complete. English score migration and the complete Chinese source,
target, and display-signal migration have closed static inventories, deterministic
call-flow audits, and frozen machine/gold corpus snapshots. Phase 8 isolation,
task-scoped efficiency telemetry, candidate experiments, and the first Phase 9
blind holdout are also complete. Every historical Phase 8/9 behavioral candidate
was rejected and its runtime behavior removed. The legacy main path now includes
later local fixes; it is not identical to the historical baseline. Full-pipeline
Phase 10 has not started. The local app delivery above is tracked separately.
Any future candidate must return to development-only evidence
collection under a new isolated revision; the completed holdout cannot be used
for tuning.

## 3. Non-Negotiable Engineering Rules

### 3.1 Preserve the current production path

- The legacy translation path remains the default until every acceptance gate is
  satisfied.
- New architecture must first run as an adapter, recorder, validator, or shadow
  path.
- A new code path must be independently disableable.
- No migration step may make fallback impossible.
- A confirmed harmful candidate is rejected. Insufficient evidence, non-activation,
  suspected false positives, and marginal cost overruns remain isolated for
  review. Do not relax a frozen threshold retrospectively or delete useful
  regression tests just because a stochastic full run is inconclusive.

### 3.2 Separate structural and behavioral work

A structural patch may:

- move code without changing decisions;
- introduce immutable data types;
- precompute existing features;
- return typed diagnostics alongside legacy messages;
- add adapters, registries, metrics, and reports;
- add tests that capture current behavior.

A structural patch must not:

- change prompts;
- change translation wording;
- change boundary scores or thresholds;
- change retry behavior;
- add new domain-specific corrections;
- remove existing fallback branches;
- change which candidate is accepted.

A behavioral patch may change one explicitly named policy only after the
structural baseline is stable. Its evaluation report must isolate that policy's
effect.

### 3.2.1 Bind evidence to components and final code

- Freeze the working baseline and each component's file/hash manifest; Git HEAD
  alone does not identify a dirty working tree. Record corpus and configuration
  identities alongside every report.
- Prove individual decisions with fixed inputs, relevant counterexamples and
  call-path checks before a separately named combination experiment. Record the
  combination's component revisions and interactions; do not require a paid
  full-video run for every deterministic component.
- Removing a component creates a new retained revision. Explain which earlier
  evidence remains applicable and recheck affected paths. Old aggregate tokens,
  latency, or quality conclusions cannot be inherited by the reduced subset.
  Mark unmeasured outcomes explicitly instead of starting unplanned paid runs.
- Record activation, correct repairs, cache hits, and demonstrably avoided
  requests separately. A zero-hit cache has no measured saving in that run;
  combination savings do not establish each component's benefit.
- Validator relaxation requires both source-supported correct acceptance and
  rejection of relevant near misses, including extra incorrect facts beside a
  correct match. Deterministic implementation alone does not prove semantic
  equivalence; probabilistic tolerance belongs on the quality track.

Use fixed-input checks, offline replay, necessary bounded live probes, then
prospectively planned full pairs. Reuse applicable evidence with its identity;
do not mechanically repeat every stage. Each escalation needs a remaining
question, evidence to proceed, and a stopping condition. Preserve pilot failures
and cost, introduce fixes under a new identity, and do not exclude inconvenient
runs after seeing their completed cost results.

### 3.3 Do not optimize against the holdout set

- Holdout videos are selected before general rule development begins.
- Their subtitle text must not be used to invent prompts, rules, exceptions, or
  terminology mappings.
- During ordinary development, report only aggregate holdout results.
- If detailed holdout inspection becomes necessary, move that video to the
  development set, record the contamination, and replace it with a previously
  unseen holdout before continuing.
- Never repeatedly tune a threshold until one fixed holdout sample passes.

### 3.4 Protect user data and credentials

- Do not copy videos or complete external subtitle corpora into Git.
- Do not store API keys, Base URLs containing credentials, cookies, or keyring
  contents in manifests or reports.
- Use paths relative to a configurable local corpus root.
- Store hashes and metadata in the repository; keep source media and full SRT
  files in the user's external data directory.
- Reports committed to Git must not contain complete copyrighted subtitle text.
- Small excerpts may appear only in local ignored reports when needed for
  debugging.

### 3.5 Preserve unrelated worktree changes

Before every phase:

1. inspect `git status --short`;
2. record the current revision and changed paths in the phase report;
3. identify which existing changes belong to this refactor;
4. do not reset, revert, overwrite, or stage unrelated files;
5. stop if an unrelated edit changes the same function in a way that prevents a
   safe merge.

The executor must work with existing modifications. Destructive Git operations
are prohibited unless the user explicitly requests them.

### 3.6 No hidden paid evaluation

- Corpus validation, deterministic metrics, and offline replay may run without
  additional approval.
- Sending subtitle text to an external LLM requires the user's authorization for
  that provider and task.
- Shadow mode must not silently double paid API calls.
- Online shadow validation should reuse existing responses whenever possible.
- Token, call, retry, cache, and latency measurements must distinguish primary
  work from evaluation overhead.

## 4. Required Repository and Local Artifacts

The target implementation should use the following separation.

### 4.1 Repository-tracked artifacts

```text
docs/dev/
  translation-quality-refactor-plan.md
  translation-quality-refactor-execution-guide.md

scripts/translation_quality/
  __init__.py
  manifest.py
  srt.py
  align.py
  metrics.py
  compare.py
  report.py

scripts/evaluate_translation_quality.py

tests/test_scripts/
  test_translation_quality_manifest.py
  test_translation_quality_srt.py
  test_translation_quality_metrics.py
  test_translation_quality_compare.py

subforge/core/translate/quality/
  __init__.py
  schema.py
  session.py
  features.py
  language.py
  numbers.py
  entities.py
  english_boundary.py
  chinese_boundary.py
  dialogue.py
  fluency.py
  registry.py
  planner.py
  policies.py
```

Only create production modules when their migration phase begins. Do not create
empty placeholder modules in advance.

### 4.2 Local ignored artifacts

```text
artifacts/translation-quality/
  corpus.local.yaml
  baselines/
  candidate-runs/
  holdout-runs/
  reports/
  logs/
```

The existing `.gitignore` already excludes `artifacts/`. Corpus paths, detailed
reports, and generated subtitle comparisons belong there.

### 4.3 Environment variables

Use explicit local configuration rather than hard-coded absolute paths:

```text
SUBFORGE_TRANSLATION_CORPUS_ROOT
SUBFORGE_TRANSLATION_CORPUS_MANIFEST
SUBFORGE_QUALITY_PIPELINE_V2
SUBFORGE_QUALITY_SHADOW_MODE
```

The first two variables are evaluation-only. The feature flags are introduced
only when their corresponding production code exists.

## 5. Corpus Manifest Specification

### 5.1 Manifest goals

The manifest must make every sample reproducible without embedding the sample in
the repository. It must answer:

- where the three files came from;
- whether they belong together;
- which file is source, machine output, and human gold;
- whether cue counts and timing are directly comparable;
- which mode, language, provider, and algorithm produced the machine result;
- whether the human version removed advertisements or changed timing;
- whether the sample is development, validation, or holdout data;
- what limitations prevent a metric from being interpreted literally.

### 5.2 Required corpus-level fields

```yaml
schema_version: 1
corpus_id: subforge-local-translation-quality
created_at: ISO-8601 timestamp
data_root_env: SUBFORGE_TRANSLATION_CORPUS_ROOT
samples: []
```

### 5.3 Required sample fields

Every sample record must include:

```yaml
id: stable-lowercase-id
title: Human-readable title
split: development | validation | holdout
domain: automotive | architecture | infrastructure | interview | news | other
speaker_mode: monologue | dialogue | multi_speaker | unknown
speaker_count: integer | null
source_languages: [en]
target_language: zh-CN
duration_ms: integer | null
source_srt: relative/path.srt
machine_srt: relative/path_processed.srt
gold_srt: relative/path_chatgpt_edited.srt
source_media: relative/path.mp4 | null
machine_model: provider/model | unknown
algorithm_version: version-or-commit | unknown
configuration:
  batch_size: integer | null
  concurrency: integer | null
  reflection: boolean | null
  multi_speaker: boolean | null
provenance:
  source_kind: word_level_asr | sentence_srt | unknown
  machine_kind: subforge | external | unknown
  gold_editor: chatgpt_pro | human | mixed | unknown
  verified_by_user: boolean
alignment:
  cue_structure: exact | requires_alignment | incompatible
  timing_changed: boolean
  advertisements_removed: boolean
  removed_ranges: []
known_issues: []
notes: string
hashes:
  source_sha256: hex
  machine_sha256: hex
  gold_sha256: hex
```

Unknown information must be recorded as `unknown` or `null`. It must not be
guessed from a filename.

### 5.4 Manifest validation rules

The validator must fail clearly when:

- an ID is duplicated;
- a required file is missing;
- a hash does not match;
- a split value is invalid;
- the same sample appears in both development and holdout sets;
- a holdout sample is marked as having been used for rule design;
- a machine/gold pair is declared exact but cue counts, keys, timing, or source
  text do not match;
- an advertisement-edited result is treated as an exact structural reference;
- an absolute path is about to be committed to a tracked manifest.

### 5.5 Holdout selection

Select three to five complete videos after metadata classification. Prefer:

- at least one single-speaker sample;
- at least one two-speaker or multi-speaker sample;
- at least one mixed-language or structurally difficult sample;
- more than one content domain;
- samples not previously used to formulate exact production rules.

Record the holdout assignment and hashes before writing new general rules.

## 6. Evaluation Model

### 6.1 Evaluation levels

Use six levels of evidence. No single metric is sufficient.

1. **Structural invariants**: keys, count, timing, source text, encoding.
2. **Hard semantic invariants**: empty output, placeholders, numbers, entities,
   negation, obvious duplication, missing keys.
3. **Boundary diagnostics**: subject-predicate, modifier-head, number-unit,
   cross-cue ownership, speaker boundaries.
4. **Human-gold comparison**: edit coverage, changed windows, terminology and
   sentence-boundary differences.
5. **Efficiency telemetry**: calls, tokens, reasoning use, retries, cache hits,
   latency, peak memory.
6. **Human pairwise preference**: legacy versus candidate without exposing which
   output is new.

### 6.2 Structural metrics

Every run must report:

- cue count equality;
- key set equality;
- key order equality;
- timestamp equality;
- source text equality;
- empty translation count;
- placeholder count;
- source-copy count;
- reasoning or note leakage count;
- invalid SRT block count;
- duplicate key count;
- encoding and newline format.

Any structural regression is a hard failure.

### 6.3 Semantic and boundary metrics

Report counts and affected windows for:

- preserved-number failures;
- preserved-entity failures;
- negation scope risks;
- repeated adjacent meaning;
- translation shifted forward or backward;
- stranded subject or conjunction;
- predicate-complement split;
- modifier-head split;
- number-unit split;
- speaker ownership conflict;
- mixed-language loss;
- suspicious untranslated text;
- unnatural Chinese construction signals.

Diagnostics must include stable rule IDs. Reports must not depend on parsing
English error sentences.

### 6.4 Gold comparison metrics

Automated comparison with a human-edited file is evidence, not an absolute
quality score. Report:

- cue-level exact match rate;
- cue-level changed rate;
- character edit distance distribution;
- windows where machine and gold allocate meaning differently;
- numbers and named entities changed by the editor;
- repeated phrases removed by the editor;
- empty or missing content restored by the editor;
- boundary-risk signals resolved by the editor;
- changes that cannot be scored because timing or advertisements differ.

Do not use BLEU-like similarity alone to accept or reject a translation. A valid
Chinese translation can differ substantially from the gold wording.

### 6.5 Efficiency metrics

For LLM-backed runs record:

- provider and model;
- request count by purpose;
- prompt tokens;
- completion tokens;
- reasoning tokens when exposed;
- cache-read and cache-write counts;
- retry count by status code or failure class;
- rate-limit wait time;
- total wall time;
- translation time;
- validation time;
- repair time;
- deterministic finalization time;
- peak resident memory when available.

Cached and uncached runs must not be mixed in one performance comparison.

### 6.6 Scoring and acceptance

Use safety-first, track-specific admission:

1. investigate observed hard signals and block rollout until integrity is verified;
2. reject confirmed candidate-caused severe semantic or speaker/language regressions;
3. separate inherited defects, actual repairs, heuristic false positives, and uncertain causality;
4. review meaning and style alongside costs, even when a risk or budget signal rises;
5. for bugfixes, require fixed-input proof, negative cases, and affected-path regression;
6. for quality, require independent benefit; for efficiency, require quality non-inferiority;
7. use prospectively frozen budgets and paired-repeat evidence, not a single noisy run;
8. keep useful inconclusive candidates isolated; never auto-adopt from static gates.

`compare --policy ... --policy-sha256 ...` adds schema-2 screening. Omitting the
policy preserves historical v1 behavior for reproducibility only. The legacy
`accepted` field is not the v2 verdict. See the main guide for the JSON contract,
exit codes, manual reviews, and mode-scoped rollout requirements.

Do not collapse all quality dimensions into one weighted score and allow a high
fluency score to hide a factual failure.

## 7. Baseline Capture Procedure

### 7.1 Preflight

Before running a baseline:

- record the current Git revision;
- record `git status --short` without altering it;
- record Python, uv, operating system, architecture, and dependency versions;
- record the SubForge version;
- validate corpus hashes;
- confirm which samples are development and holdout;
- confirm whether external API use is authorized;
- disable unrelated experimental feature flags;
- decide whether caches are enabled and record that decision.

### 7.2 Baseline types

Capture three separate baselines:

1. **Static corpus baseline**: compare existing machine and gold SRT files with no
   API calls.
2. **Legacy replay baseline**: run current validators and finalizers over stored
   machine results without retranslating.
3. **Full legacy pipeline baseline**: rerun translation only when explicitly
   authorized, using a pinned model and recorded configuration.

The static baseline is mandatory. Full API replay is optional and must not block
structural refactoring.

### 7.3 Baseline artifacts

Each run creates a uniquely named directory:

```text
artifacts/translation-quality/baselines/
  YYYYMMDD-HHMMSS_<commit>_<run-id>/
    run.json
    corpus-summary.json
    sample-metrics.jsonl
    aggregate.json
    report.md
    environment.json
```

Do not overwrite a baseline. Record the corpus manifest hash in `run.json`.

### 7.4 Baseline exit gate

Phase 1 cannot finish until:

- all manifest records validate;
- all exact machine/gold pairs are structurally aligned;
- incompatible pairs are explicitly excluded from exact metrics;
- repeated runs produce deterministic static metrics;
- reports contain no secrets or unintended full subtitle dumps;
- unit tests cover malformed SRT and malformed manifest cases;
- the baseline report can identify the known defect classes in the corpus.

## 8. Migration Phases

Each phase below is a separate reviewable change. Do not combine phases merely
because their code touches the same large file.

### Phase 0: Freeze scope and classify the corpus

**Objective**

Establish trustworthy inputs before editing production architecture.

**Allowed work**

- create the local manifest;
- compute hashes and cue statistics;
- classify mode, language, domain, provenance, timing changes, and advertisement
  removal;
- select and freeze holdout samples;
- update the archived plan's inventory and status.

**Prohibited work**

- changing production prompts or rules;
- editing human gold subtitles;
- choosing holdouts after seeing candidate failures;
- copying the full corpus into Git.

**Required output**

- validated local manifest;
- corpus inventory report;
- holdout assignment record;
- list of missing metadata and incompatible sample pairs.

**Exit gate**

The numerical and category coverage is known, not assumed.

### Phase 1: Build the evaluator and capture the baseline

**Objective**

Measure current behavior without changing it.

**Allowed work**

- add manifest parsing and validation;
- add robust SRT parsing for evaluation;
- add deterministic metrics;
- add report generation;
- add unit tests and static corpus reports.

**Prohibited work**

- importing evaluator decisions into production;
- calling paid APIs without authorization;
- using gold translations as runtime prompts;
- producing a single opaque quality score.

**Required output**

- evaluator CLI;
- unit tests;
- first immutable baseline report;
- documented command examples.

**Exit gate**

The same corpus and revision produce the same static report, and known structural
defects are observable.

### Phase 2: Introduce immutable session and cue features

**Objective**

Remove repeated extraction and implicit shared state without changing decisions.

**Target types**

- `TranslationSession`;
- `CueFeatures`;
- immutable task configuration snapshot;
- explicit source, speaker, language, timing-gap, and terminology views.

**Implementation rules**

- prefer frozen dataclasses or immutable Pydantic models;
- build the session once per task;
- build each cue's features once;
- pass session/features explicitly to new code;
- expose adapters for legacy methods;
- prohibit validators from mutating source cues or task configuration;
- preserve existing callback, stop, cache, and recovery behavior.

**Parity checks**

- old and adapter-backed paths return identical decisions;
- prompts are byte-for-byte unchanged;
- candidate acceptance is unchanged;
- API call order and count are unchanged;
- cache keys are unchanged;
- recovery file behavior is unchanged.

**Exit gate**

All non-integration tests pass and the corpus parity report shows no output
changes.

### Phase 3: Add typed diagnostics beside legacy messages

**Objective**

Decouple control flow from human-readable error strings.

**Target schema**

Each diagnostic includes:

- stable `rule_id`;
- category;
- severity;
- confidence;
- affected cue keys;
- evidence fields;
- repair strategy recommendation;
- machine-readable parameters;
- human-readable message for logs only.

**Implementation sequence**

1. define enums and immutable diagnostic records;
2. let legacy validators optionally emit diagnostics;
3. keep existing `(bool, message)` adapters;
4. compare legacy outcome and typed outcome in tests;
5. migrate consumers one at a time;
6. remove message parsing only after every consumer is typed.

**Stable rule ID format**

Use lowercase dotted identifiers, for example:

```text
schema.missing_key
translation.empty
number.value_changed
entity.identifier_changed
boundary.subject_predicate_split
dialogue.speaker_ownership_conflict
fluency.translationese_degree_structure
```

Rule IDs never contain cue numbers, source text, or translated text.

**Exit gate**

Typed and legacy decisions match across unit tests and the development corpus.

### Phase 4: Migrate hard invariant rules

**Objective**

Move the safest, deterministic checks into focused modules first.

**Migration order**

1. response schema and key completeness;
2. empty output and placeholder detection;
3. source-copy and reasoning-leak detection;
4. exact numeric equivalence;
5. product identifier and entity preservation;
6. timestamp and source-text integrity;
7. deterministic punctuation finalization.

**Rule requirements**

- pure input/output where practical;
- no network access;
- no complete historical sentence translations;
- explicit applicability conditions;
- stable diagnostics;
- focused unit tests for positive, negative, and ambiguous cases;
- idempotent behavior.

**Exit gate**

Legacy parity is exact for all hard invariants, or every intentional difference
has a separately approved behavior-change report.

**Current checkpoint (2026-08-29)**

- response container, exact key-set, untranslated-target, empty-output,
  placeholder, reflective-value, and reasoning-residue checks emit typed
  diagnostics from focused quality modules;
- source-copy and reasoning-residue detection now share one implementation with
  the static evaluator; the narrower reasoning rule intentionally stops treating
  normal narration such as `以下是结果` as private model reasoning;
- numeric and product/entity losses now produce stable `number.value_missing`
  and `entity.identifier_missing` diagnostics while the legacy provider feedback
  and acceptance policy remain unchanged;
- translation write-back now snapshots source text, cue timestamps, word timing,
  speaker, language, and alignment metadata, and refuses to save if translation
  mutates any source-owned field;
- Chinese punctuation finalization is a pure, idempotent function that retains
  decimals and ASCII identifiers while preserving the existing product output;
- the approximately 900-line numeric/entity equivalence implementation and its
  shared alias tables now live in `quality/preservation.py`; `LLMTranslator`
  retains only a thin compatibility adapter and its acceptance policy is
  unchanged;
- the full non-integration suite passes with 1,886 tests and 35 integration tests
  deselected;
- Ruff passes and Pyright reports zero errors and warnings.

The authorized development validation used DeepSeek `deepseek-v4-flash` with 20
concurrent requests and a batch size of 20. It produced 254 cues from 163
successful requests in 129.8 seconds, with zero empty targets, placeholders,
reasoning leaks, source-copy targets, untranslated targets, or adjacent duplicate
risks. Prompt-cache reuse was 60.59% and boundary Jaccard against the human-edited
reference was 0.9326. This is development evidence, not holdout acceptance. A
post-run targeted check confirmed the deterministic repair for a source-supported
Chinese nominal modifier without issuing another paid request.

After the hard-rule migration above, a fresh deterministic run over all 13 local
samples produced aggregate metrics identical to the Phase 3 report: 56,188 source
cues, 5,288 machine cues, 5,289 gold cues, zero hard failures, zero empty or
placeholder targets, zero reasoning leaks, one source-copy risk, and 4,068
human-changed cues. No additional paid request was issued for this parity check.

### Phase 5: Add the repair planner in shadow mode

**Objective**

Replace scattered retry decisions with an explicit plan while still returning
the legacy result.

**Planner inputs**

- immutable session;
- cue features;
- all diagnostics for the local window;
- provider capabilities;
- remaining task and rule budgets;
- previous repair attempts;
- stop state.

**Planner outputs**

- repair strategy;
- affected keys;
- context window;
- reasoning mode;
- maximum attempts;
- hard invariants required for acceptance;
- fallback strategy;
- machine-readable rationale based on diagnostics.

**Repair tiers**

1. deterministic format or exact-value repair;
2. non-reasoning LLM repair for ordinary fluency;
3. reasoning LLM repair for confirmed semantic displacement, complex Chinese
   order, entity ambiguity, or cross-cue ownership;
4. conservative legacy fallback;
5. recovery output when no candidate passes hard gates.

**Shadow-mode restrictions**

- return the legacy result to the user;
- do not issue an additional paid LLM request by default;
- evaluate planner decisions against stored or cached responses;
- record disagreements without changing output;
- make every planner decision reproducible from the report.

**Exit gate**

Planner decisions are stable, budgeted, and no less conservative than legacy
behavior on hard failures.

**Current checkpoint (2026-08-30)**

- a pure planner now emits immutable plans containing disposition, repair tier,
  affected keys, context radius, reasoning mode, maximum attempts, required rule
  IDs, fallback, and machine-readable rationale;
- provider reasoning capability, total/reasoning budgets, previous-attempt counts,
  and cancellation state are explicit inputs;
- schema, untranslated-target, reasoning-residue, placeholder, reflective-schema,
  number, and entity failures record shadow plans while returning exactly the
  legacy validation result;
- equal plans are aggregated in a thread-safe recorder capped at 128 unique
  entries; no subtitle text is stored and no per-item log is emitted;
- the planner itself performs no network operation and does not alter prompts,
  retries, token use, or output;
- 1,897 non-integration tests pass, Pyright and Ruff pass, and all 13 local static
  samples remain identical to the Phase 3 aggregate baseline.

The immutable task session and observed same-failure history now feed planning,
and planner-versus-legacy decisions are aggregated without executing them.
Phase 5 is complete for typed hard diagnostics. Boundary-policy diagnostics remain
scheduled for Phase 7.

### Phase 6: Separate monologue, dialogue, and mixed-language policies

**Objective**

Stop mode-specific behavior from being scattered through shared validators.

**Shared hard invariants**

- key and timing integrity;
- empty-output prohibition;
- numeric and entity preservation;
- no source leakage or reasoning leakage;
- no unsupported facts;
- speaker metadata integrity.

**MonologuePolicy**

- permits wider same-speaker context;
- allows an omitted Chinese subject only when unambiguous;
- emphasizes clause continuity, subject-predicate ownership, modifiers, and
  rhetorical flow;
- does not repeat a subject merely to make every cue independent.

**DialoguePolicy**

- treats speaker changes as hard boundaries by default;
- exposes adjacent turns as read-only context;
- preserves question-answer and stance ownership;
- limits cross-cue repair to the same speaker unless only non-material Chinese
  grammar is supplied;
- never emits speaker labels unless the product setting requires them.

**MixedLanguagePolicy**

- preserves the user's explicit source-language selection;
- uses per-cue detected language as metadata, not as permission to overwrite the
  source-language setting;
- chooses language-specific alignment and boundary rules;
- reconstructs a mixed-language utterance only within a verified same-speaker
  window;
- preserves names and code-switching function.

**Exit gate**

No mode regresses on its corresponding development set, and aggregate holdout
speaker/mixed-language metrics do not worsen.

**Completed checkpoint (2026-08-30)**

- immutable speaker and language policy dimensions now compose into one
  `TranslationModePolicy` selected from the task session;
- monologue, dialogue, unknown-language, single-language, and mixed-language
  modes expose explicit context and cross-speaker-repair permissions;
- prompt-family selection and metadata guidance read the policy while preserving
  the previous prompt names and guidance text byte for byte;
- the policy layer changes no retry budget, provider behavior, reasoning mode,
  model request, boundary score, subtitle timing, or output;
- focused tests cover every mode selection and the existing single-speaker,
  multi-speaker, Kimi K3, and Nemotron prompt families;
- 1,907 non-integration tests pass, Pyright and Ruff pass, and the aggregate for
  all 13 local samples is byte-identical to the frozen Phase 3 baseline.

### Phase 7: Migrate boundary rules into a registry

**Objective**

Replace long condition chains while preserving the dynamic-programming
segmentation strategy.

**Rule definition fields**

- stable rule ID;
- direction and weight;
- applicable source languages;
- applicable speaker modes;
- required features;
- exclusions and exceptions;
- severity;
- explanation template;
- tests covering activation and non-activation.

**Migration method**

1. instrument current score contributions;
2. snapshot decisions for representative boundaries;
3. migrate one rule family at a time;
4. compare every score contribution, not only final split output;
5. keep legacy and registry scoring selectable;
6. migrate English boundary rules before Chinese repair rules only if parity data
   is complete;
7. remove a legacy branch only after registry parity is proven.

**English registry and ordered score-plan checkpoint (2026-08-30)**

- 74 source, target, display, and fluency boundary diagnostics now live in a
  declarative registry with stable IDs, scope, kind, level, applicable language
  and speaker modes, required features, and exclusions;
- all 149 English scoring rules now expose stable IDs, direction, weight, severity,
  explanation templates, applicability, required features, and exceptions;
- all 151 scoring call sites preserve the exact legacy reason text and risk weight,
  including repeated contributions and the reason-free lowercase-continuation
  bonus;
- a static source auditor rejects unresolved rule IDs, unpaired score/reason
  branches, and any reintroduction of anonymous legacy score sites; the audited
  legacy score-site count is now zero;
- a text-free boundary snapshot command hashes every adjacent source pair and
  records total, registered, and unregistered risk without storing subtitle text;
- across 13 local samples and 5,275 adjacent boundaries, all 1,938 risk points
  are attributable to registered rules, unregistered active risk is zero, and
  the frozen decision hash is
  `51479e015de80013d81e89d29e74b4d4caeb300dbf7d14e0f07c557c057fdf1d`;
- immutable `EnglishBoundaryFeatures` now extracts shared lexical, semantic, and
  completion evidence once for every eligible source boundary;
- all English conditions have moved into focused numeric, comparison, entity,
  coordination, discourse, predicate, and grammar detector modules, each with
  activation and non-activation fixtures;
- the 151 registered score applications are grouped into six explicitly ordered
  stages: foundation, relations, completions, discourse, clause ownership, and
  dependencies;
- the source auditor resolves the declared stage plan, freezes its exact order,
  and rejects any registered score call left outside that execution closure;
- `assess_english_boundary` has fallen from 1,925 lines to a 28-line coordinator
  while retaining exact scoring order and repeated-contribution behavior;
- 2,076 non-integration tests pass, Pyright reports zero errors, Ruff passes, and
  aggregate static quality remains byte-identical to the Phase 3 baseline;
- the 5,275-boundary snapshot remains byte-identical after stage extraction and
  retains the frozen decision hash;
- no provider call, prompt, retry, reasoning mode, output text, or timing was
  changed by this migration;
- English migration is complete for this checkpoint. Chinese boundary behavior
  must now be inventoried, assigned stable rule IDs, and frozen at contribution
  level before its own migration begins; Phase 8 remains blocked until the full
  Phase 7 exit gate is met.

**Exit gate**

Boundary decisions and scores are explainable, stable, and no worse on corpus
metrics and pairwise review.

**Chinese boundary-inventory checkpoint (2026-08-30)**

- all 74 registered source, target, and display signals are emitted by the
  inventoried production flow; no unknown signal or unused definition remains;
- 104 literal signal sites are frozen: 94 in `_chinese_boundary_signal`, 2 in
  `_long_gap_chinese_boundary_signal`, 7 in `_source_boundary_signal`, and 1
  compatibility override in `_target_boundary_diagnostic`;
- all 9 direct calls to the three signal functions are inside the explicit
  audit, candidate-selection, confirmation, windowing, or reasoning-policy
  closure; no unscanned caller remains;
- the source signal has one intentional dynamic return that delegates to the
  already registered English contribution set; corpus attribution confirms zero
  unregistered source signals;
- the original position-coupled inventory fingerprint is
  `830103a49d098d7ec2b6c799082022511fc9b869c59267cacea639ee48abeaa7`;
  the migration-safe semantic inventory hash, recomputed against the frozen
  pre-migration source, is
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`, and the
  call-flow hash is
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`;
- a text-free machine/gold snapshot covers 5,275 machine boundaries and 5,276
  human-edited boundaries. Machine output emits 345 target candidates with hash
  `74329c58e5ae31ed3ad7a0dd9a1897b8d6e1438a186a483628a46804a20525b0`;
  gold output emits 288 with hash
  `0502c68ee173c92a21ea05b1447853e2cc30280d1e550b2333440a3d40d24713`;
- repeated audit and snapshot generation is byte-identical, and neither side
  contains an unregistered source or target signal;
- 2,079 non-integration tests pass, Pyright reports zero errors, Ruff passes,
  and aggregate static quality remains byte-identical to the frozen Phase 3
  baseline;
- no product module, prompt, model request, translated text, timing, or repair
  decision changed in this inventory checkpoint.

Chinese migration must preserve first-match precedence. Unlike English scoring,
these rules do not accumulate: moving one broad soft rule ahead of a narrow hard
rule changes both the diagnostic and the downstream reasoning/repair path even
when the set of conditions is unchanged.

**Chinese immutable-feature and visible-pause checkpoint (2026-08-30)**

- immutable `ChineseBoundaryFeatures` now performs the existing text trimming,
  filler normalization, compacting, canonicalization, terminal-punctuation, and
  gap extraction once without changing those operations;
- `_chinese_boundary_signal` reads its initial normalization from that feature
  object, while `_long_gap_chinese_boundary_signal` is a thin compatibility
  adapter over the pure `detect_visible_pause_boundary` detector;
- the first two migrated display rules keep their original first-match order,
  threshold, message, and stable registry identity: visible-pause number/unit
  separation and unfinished predicate/modifier separation;
- the static auditor scans detector modules as well as the legacy adapter and
  distinguishes a semantic inventory hash from the physical layout hash. The
  semantic hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`, the
  post-extraction layout hash is
  `d58eb0d3a11e91af294bc5ca7edbc0fe1b58d2b717cb59c3415311b69f05140c`, and the
  call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`;
- the frozen machine/gold Chinese boundary snapshot remains byte-identical:
  5,275 machine boundaries with 345 target candidates and 5,276 gold boundaries
  with 288 target candidates, with no unregistered source or target signal;
- the aggregate static report for all 13 samples remains byte-identical to the
  Phase 3 baseline;
- 2,085 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued;
- prompts, provider requests, candidate selection, repair windows, reasoning,
  output text, and subtitle timing are unchanged.

The first syntax extraction was limited to one contiguous first-match family.
All later Chinese branches remained in place, and the checkpoint was accepted
only after the semantic inventory hash and both machine/gold snapshots remained
identical.

**Chinese foundation-detector checkpoint (2026-08-30)**

- the first five contiguous branches of `_chinese_boundary_signal` now live in
  the pure `detect_foundation_boundary` detector: standalone connectives,
  demonstrative subjects, sentence adverbs, subject-plus-adverb tails, and
  coordinated subjects stranded from their predicates;
- the compatibility function invokes the detector exactly where those branches
  previously appeared and still stops at the first registered match;
- focused activation, non-activation, and adapter-parity tests cover every moved
  branch without changing later branch order;
- the migration-safe semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the post-extraction physical layout hash is
  `a99fc8712ad8cd7bd709212259e4faccd98af7830a6fac68a69f6edd06e5f7d7`;
- the frozen machine/gold Chinese boundary snapshot remains byte-identical at
  5,275 machine boundaries and 5,276 gold boundaries, with zero unregistered
  source or target signals;
- the 13-sample aggregate static report remains byte-identical to the Phase 3
  baseline;
- 2,096 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

The second syntax extraction was limited to the immediately following
relative-clause/head-noun, nominal-head, and reporting-complement family. It
stopped before `把` and disposal constructions and introduced no behavioral rule
change.

**Chinese nominal-attachment checkpoint (2026-08-30)**

- nine immediately following branches now live in the pure
  `detect_nominal_attachment_boundary` detector: relative/head-noun attachment,
  demonstrative-relative completion, comparative-object attachment, comparison
  frames, vehicle model modifiers, reporting complements, classifier completion,
  demonstrative modifiers, and contextual count classifiers;
- the detector remains after the foundation detector and before every `把` or
  disposal branch, preserving exact first-match precedence;
- the semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the post-extraction physical layout hash is
  `436abe430ddbda377d1ec1024821b06e9d9ee3c794dd5bbaa9773678675f01f7`;
- the frozen machine/gold Chinese boundary snapshot and the aggregate static
  report remain byte-identical, with no unregistered signal or hard failure;
- 2,106 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

The third syntax extraction was limited to the immediately following
`把`/disposal, required-predicate-complement, and locative/temporal
governing-clause family. It stopped before duplicate-boundary and literal-fluency
diagnostics and introduced no behavioral rule change.

**Chinese governing-attachment checkpoint (2026-08-30)**

- five immediately following branches now live in the pure
  `detect_governing_attachment_boundary` detector: `把` attachment, general
  disposal attachment, required predicate complements, locative phrases, and
  standalone temporal phrases;
- predicate-present `把`/disposal cases and complete `由……承受/承担` passive
  structures retain their legacy exceptions and fall-through order;
- the semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the post-extraction physical layout hash is
  `22d910b161709554b52b3a41b3ffa13f22ac0cd8b7b16f4f0e2c515ee0f2af96`;
- the frozen machine/gold Chinese boundary snapshot and aggregate static report
  remain byte-identical, with zero unregistered signal or hard failure;
- 2,114 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

The fourth syntax extraction was limited to the contiguous surface-duplication
and literal-fluency family, from the explicit `所受的` duplicate through repeated
locative meaning. It stopped before material-subject attachment rules and
introduced no behavioral change.

**Chinese surface-fluency checkpoint (2026-08-30)**

- eleven immediately following branches now live in the pure
  `detect_surface_fluency_boundary` detector: explicit duplication, superlative
  attachment, literal Japanese phrasing, duplicated construction nominalization,
  stacked connectives, canonical overlap, bigram/sequence similarity, repeated
  lexical units, repeated predicates, and repeated locatives;
- all overlap lengths, `0.72` similarity thresholds, canonical forms, and
  first-match precedence remain unchanged;
- the semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the post-extraction physical layout hash is
  `c87c6ee41b443253b77e1900763c34a44aae3add830997f25671a2f6fe8aedb0`;
- the frozen machine/gold Chinese boundary snapshot and aggregate static report
  remain byte-identical, with zero unregistered signal or hard failure;
- 2,126 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

The fifth syntax extraction was limited to material-subject ownership and
coordinated-modifier attachment. It stopped before stranded-connective and
copular-bridge rules and introduced no behavioral change.

**Chinese subject-attachment checkpoint (2026-08-30)**

- three immediately following branches now live in the pure
  `detect_subject_attachment_boundary` detector: material subjects, existential
  people subjects, and coordinated modifiers;
- the object-role exclusion for verbs such as `帮助/告诉/采访` and the terminal
  punctuation guard retain their original behavior and precedence;
- the semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the post-extraction physical layout hash is
  `8b5fdcd6b3c62fc53da297d023f5d4c620483d10693da0c35de026258e38d561`;
- the frozen machine/gold Chinese boundary snapshot and aggregate static report
  remain byte-identical, with zero unregistered signal or hard failure;
- 2,132 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

The sixth syntax extraction was limited to stranded-connective and copular-bridge
detection. It stopped before unfinished grammatical and reporting-frame rules
and introduced no behavioral change.

**Chinese discourse-bridge checkpoint (2026-08-30)**

- three immediately following branches now live in the pure
  `detect_discourse_bridge_boundary` detector: a connective stranded at the
  previous subtitle end, an `在于` bridge, and a topic noun separated from its
  following `是` clause;
- exact first-match order, punctuation guards, messages, and compatibility
  adapter behavior remain unchanged; 36 main-path rules and two visible-pause
  rules are now isolated in pure detectors;
- the semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the post-extraction physical layout hash is
  `5545b371f6c9ce926f4d61629e74464d44f99ffc651f95544ecec2f91420847a`;
- the frozen machine/gold Chinese boundary snapshot and aggregate static report
  remain byte-identical, with zero unregistered signal or hard failure;
- 2,137 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

The seventh syntax extraction was limited to unfinished grammatical, reporting,
locative, disposal, and copular-frame detection. It stopped before the reason
construction and domain-specific predicate rules and introduced no behavioral
change.

**Chinese unfinished-frame checkpoint (2026-08-30)**

- five immediately following branches now live in the pure
  `detect_unfinished_frame_boundary` detector: special unfinished tails,
  reporting frames, embedded locative frames, pronoun-plus-`把` frames, and
  copular frames separated from their result;
- exact regular expressions, first-match order, messages, and compatibility
  adapter behavior remain unchanged; 41 main-path rules and two visible-pause
  rules are now isolated in pure detectors;
- the semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the post-extraction physical layout hash is
  `8b63337d27e5e63ecf990c0c1f621a3ebc794619c202e8c611bdcfcb1375c341`;
- the frozen machine/gold Chinese boundary snapshot and aggregate static report
  remain byte-identical, with zero unregistered signal or hard failure;
- 2,143 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

The eighth syntax extraction was limited to the immediately following Chinese
reason-construction rule. It stopped before percentage-specific and other domain
predicate rules and introduced no behavioral change.

**Chinese reason-construction checkpoint (2026-08-30)**

- the `之所以……原因` branch now lives in the pure
  `detect_reason_construction_boundary` detector;
- the earlier copular-bridge result for an `原因 / 是我们……` boundary is locked by
  a dedicated precedence test; exact matching, messages, and fall-through remain
  unchanged;
- 42 main-path rules and two visible-pause rules are now isolated in pure
  detectors;
- the semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the post-extraction physical layout hash is
  `2876b7839958fad8adccf989c8a0db7eeae4f84d3b979fce3e7608833f258eae`;
- the frozen machine/gold Chinese boundary snapshot and aggregate static report
  remain byte-identical, with zero unregistered signal or hard failure;
- 2,148 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

The ninth syntax extraction was limited to percentage-use predicates,
resultative predicates, classifier phrases, and comparison examples. It stopped
before numeric-range rules and introduced no behavioral change.

**Chinese completion-frame checkpoint (2026-08-30)**

- four immediately following branches now live in the pure
  `detect_completion_frame_boundary` detector: percentage-use predicates,
  resultative predicates, classifier phrases, and comparison examples;
- the complete `的车/车辆/东西` resultative noun-phrase exclusion has a dedicated
  negative test; exact matching, messages, first-match order, and fall-through
  remain unchanged;
- 46 main-path rules and two visible-pause rules are now isolated in pure
  detectors;
- the semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the post-extraction physical layout hash is
  `df84678f7bdd891ca5168b828703ca95e7c823d8721eb4e0bfce867a34f32d36`;
- the frozen machine/gold Chinese boundary snapshot and aggregate static report
  remain byte-identical, with zero unregistered signal or hard failure;
- 2,154 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

The tenth syntax extraction was limited to split numeric ranges and stranded
numeric complements. It stopped before consequence-predicate rules and
introduced no behavioral change.

**Chinese numeric-completion checkpoint (2026-08-30)**

- two immediately following branches now live in the pure
  `detect_numeric_completion_boundary` detector: split numeric ranges and
  stranded numeric complements;
- left/right requirements, Arabic and Chinese numeral sets, classifier scope,
  continuation words, messages, and first-match order remain unchanged;
- 48 main-path rules and two visible-pause rules are now isolated in pure
  detectors;
- the semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the post-extraction physical layout hash is
  `c74ce1450c522e6ced99b50122f2af47bb5545ad4623d97662a4a6b1bc29ffb9`;
- the frozen machine/gold Chinese boundary snapshot and aggregate static report
  remain byte-identical, with zero unregistered signal or hard failure;
- 2,160 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

The eleventh syntax extraction was limited to the immediately following missing-
consequence-predicate rule. It stopped before semantic-frame rules and introduced
no behavioral change.

**Chinese consequence-predicate checkpoint (2026-08-30)**

- the missing-consequence branch now lives in the pure
  `detect_consequence_predicate_boundary` detector;
- its trigger and complete-predicate exclusion remain colocated, preserving the
  distinction between bare result nouns and results governed by `颁发/获得/产生/`
  `取得/带来` and the other legacy predicates;
- 49 main-path rules and two visible-pause rules are now isolated in pure
  detectors;
- the semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the post-extraction physical layout hash is
  `c0c5838e0843fb46afae5c7dcd8d616f78e65d41435d95a5b4f0537a8e63e87a`;
- the frozen machine/gold Chinese boundary snapshot and aggregate static report
  remain byte-identical, with zero unregistered signal or hard failure;
- 2,165 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

The twelfth syntax extraction was limited to incomplete semantic frames,
reporting frames, and stranded nominal modifiers. It stopped before generic
unfinished-predicate rules and introduced no behavioral change.

**Chinese semantic-attachment checkpoint (2026-08-30)**

- three immediately following branches now live in the pure
  `detect_semantic_attachment_boundary` detector: incomplete semantic frames,
  reporting frames, and stranded nominal modifiers;
- exact keyword sets, character spans, messages, first-match order, and
  fall-through remain unchanged;
- 52 main-path rules and two visible-pause rules are now isolated in pure
  detectors;
- the semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the post-extraction physical layout hash is
  `facd6630d7a26fa46bc0977693efa32f1eaa8672a32b8cf4458d09a6df166ba4`;
- the frozen machine/gold Chinese boundary snapshot and aggregate static report
  remain byte-identical, with zero unregistered signal or hard failure;
- 2,171 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

The thirteenth syntax extraction was limited to the generic unfinished-predicate
rule. It stopped before aspect-predicate rules and introduced no behavioral
change.

**Chinese unfinished-predicate checkpoint (2026-08-30)**

- the single immediately following branch now lives in the pure
  `detect_unfinished_predicate_boundary` detector;
- the trigger set and nominal-attempt exclusion remain colocated, preserving
  complete noun phrases such as `一项大胆尝试`;
- exact messages, first-match order, and fall-through remain unchanged;
- 53 main-path rules and two visible-pause rules are now isolated in pure
  detectors;
- the semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the post-extraction physical layout hash is
  `2aa15b3fd73b76be114f8d47f186ecdbad98fabf57adc943ae8377bf9f870d88`;
- the frozen machine/gold Chinese boundary snapshot and aggregate static report
  remain byte-identical, with zero unregistered signal or hard failure;
- 2,176 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

The remaining Phase 7 syntax extraction continued through ten independently
verified checkpoints: predicate completion, temporal/locative attachment,
incomplete nominal frames, semantic completion, clause attachment,
subject/nominal completion, structural tails, late structural frames,
adverb/pronoun attachment, and terminal tokens.

**Chinese detector extraction complete (2026-08-30)**

- all 94 main-path literal message sites and both visible-pause sites now live in
  pure detector modules;
- exclusions remain colocated with their triggers, including nominal attempts,
  completed choices, passive use, independently nominalized people, complete
  perspective frames, nominal superlatives, and complete locative subjects;
- a shadowed duplicate-connective rule remains present and tested while the
  compatibility adapter correctly preserves the earlier connective-stranding
  result;
- `ORDERED_CHINESE_BOUNDARY_DETECTORS` fixes the complete 23-detector precedence
  in one immutable registry;
- `_chinese_boundary_signal` now normalizes features once, iterates the registry,
  and returns the first match without a complexity waiver;
- the semantic inventory hash remains
  `ec283cc7a03e0d445e72862531c479479c9928a1bbabc40b6b1a949db9f16870`,
  the call-flow hash remains
  `a28401e0ba345a60e56a67b09cb58cfad833b2e7264be7332ac8fca173477f38`,
  and the final physical layout hash is
  `60ffbb2846333f2b755fdb2999db13d22822c7e4a17b02fd7c3604bb961c1a1e`;
- all intermediate and final machine/gold snapshots and 13-sample aggregate
  reports remain byte-identical to the frozen baselines;
- 2,236 non-integration tests pass, Pyright reports zero errors and zero warnings,
  Ruff passes, and no model request was issued.

Phase 8 isolation and telemetry plumbing are complete. The first bounded
behavioral candidate passed development-only admission but subsequently failed
the frozen Phase 9 blind holdout efficiency gates and was removed. Production
remains legacy. Holdout outputs and per-cue reports must not be inspected or used
for further tuning.

### Phase 8: Enable candidate output for development data

**Objective**

Allow the new pipeline to produce files without becoming the product default.

**Requirements**

- feature flag defaults to off;
- legacy and candidate outputs are written separately;
- cache namespaces cannot collide;
- recovery files identify which pipeline produced them;
- reports contain side-by-side metrics;
- no UI setting is exposed yet;
- no existing output file is overwritten.

**Isolation plumbing completed (2026-08-30)**

- `SUBFORGE_QUALITY_PIPELINE_V2` is default-off and accepts only explicit boolean
  values; enabling it without `SUBFORGE_QUALITY_PIPELINE_REVISION` fails closed;
- the resolved identity is frozen when the background task is scheduled;
- legacy output and recovery names, task payloads, and cache keys remain
  compatible;
- candidate output and recovery names contain the normalized revision and task
  ID, existing candidate files are not overwritten, and task results identify
  the producing pipeline;
- candidate cache namespaces flow through smart splitting, source optimization,
  document context, every LLM request, and translator-result caching;
- `evaluate_translation_quality.py compare` writes side-by-side aggregate and
  per-sample metrics without cue text, fails on mismatched sample identity, and
  supports a regression exit code;
- the comparison identity is derived from sample IDs, splits, source hashes,
  gold hashes, and alignment declarations while allowing a different candidate
  machine-output path;
- the frozen 13-sample report self-comparison passes all static gates with zero
  deltas; aggregate metrics and both English and Chinese boundary snapshots are
  byte-identical to their frozen baselines;
- 2,252 non-integration tests pass after the isolation work; Pyright reports zero
  errors and Ruff passes. No model request, prompt change, repair-decision
  change, or UI setting was introduced.

**Efficiency and shadow-admission telemetry completed (2026-08-30)**

- every explicit task client owns a thread-safe accumulator for wall duration,
  provider attempts, successes and failures, retry type and wait, token families,
  reasoning modes, per-stage models, and API duration;
- retry telemetry records the delay already chosen by the provider policy and
  does not alter attempts, wait duration, or exception behavior;
- candidate success and recovery files receive an isolated, atomic
  `.telemetry.json` sidecar; legacy filenames and task payloads remain unchanged;
- the workload identity is a SHA-256 of the source SRT and the sidecar contains
  no prompt, response, subtitle text, cue key, credential, or endpoint;
- efficiency aggregation rejects missing identity, duplicate tasks, mixed
  pipeline revisions, mixed cache states, and partial repair-shadow evidence;
- comparison gates require identical workloads, cache state, and snapshot count,
  cap calls, attempts, tokens, and wall time at legacy x 1.05, and forbid broader
  reasoning use;
- bounded repair-shadow summaries expose only counts for diagnostics, plans,
  strategies, reasoning modes, session modes, and observed legacy-action routes;
- repair-shadow admission rejects a candidate when no typed repair was exercised,
  plan/action comparison coverage is incomplete, or bounded storage dropped an
  observation; planner mismatches remain evidence to review, not automatic
  permission to change behavior;
- legacy actions are observed at main-batch retry, alignment repair, locked-batch
  recovery, single-item fallback, and Chinese fluency-repair exits without
  executing a planner decision or adding an LLM request;
- 2,273 non-integration tests pass and 35 external integration tests remain
  deselected; Ruff and Pyright pass;
- the frozen 13-sample aggregate report, English boundary snapshot, and Chinese
  machine/gold boundary snapshot are byte-identical; all English and Chinese
  registry/audit hashes remain unchanged. No model request was issued.

This completes the structural prerequisite only. First run the unchanged
behavior with a named candidate shadow revision and retain its outputs and
telemetry as the baseline. If typed diagnostics are absent, expand the
development run rather than weakening the admission gate. If comparison
coverage is incomplete, instrument the missing legacy exit before selecting
behavior. Freeze only one fully observed mismatched route under a new revision,
then run the exact same development workload and pass the frozen shadow report
as `legacy` to the comparison command. Compare quality, calls, tokens, cache
state, reasoning use, and latency. The ordinary legacy path never needs to write
a sidecar and remains byte-compatible for users.

**Frozen shadow baseline and first behavioral admission (2026-08-30)**

- `phase8-shadow-baseline` ran all seven development samples with DeepSeek V4
  Flash, concurrency 20, batch size 20, reflection enabled, and an explicit
  no-disk-cache client state;
- it produced six complete files and one Bentley recovery file in 1,118,935 ms,
  using 2,436 successful requests, 5,453,029 total tokens, 356,804 reasoning
  tokens, and a 0.6344 provider cache-hit rate;
- repair shadow recorded and compared all 137 observations with zero drops; 64
  observations followed the fully visible mismatch
  `local_rewrite/disabled -> retry/disabled`;
- `phase8-local-preservation-repair-v1` changes only that route: each affected
  key gets at most one non-reasoning local rewrite, the complete candidate is
  revalidated, and any failure resumes the unchanged batch retry path;
- the behavioral run again produced six complete files and one Bentley recovery
  file in 1,097,242 ms, using 2,403 requests, 5,268,567 tokens, 304,275 reasoning
  tokens, and a 0.6710 cache-hit rate;
- all 193 repair observations were compared with zero drops; 43 planned local
  rewrites executed the intended local route, and the static/efficiency gates
  accepted the revision with no additional hard failure, placeholder, empty
  target, source copy, untranslated target, duplicate risk, or reasoning leak;
- fresh runs do not guarantee identical sentence segmentation. The evaluator's
  cue-level human-change count is therefore not a valid cross-run quality delta
  here; hard invariant metrics, source/gold workload identity, isolated outputs,
  route evidence, and efficiency remain valid;
- the unchanged recovery count prevents production admission. The Bentley
  failure demonstrates two separate literal-validation conflicts: localized
  numeric magnitudes and context-supported ASR canonical-name corrections. They
  must be tested as independent candidates, not folded into the local-rewrite
  behavior.

**Rejected canonical-mapping assumption (2026-08-30)**

- a follow-up revision required an explicit context ASR mapping plus either
  near-identical spelling or repeated source evidence before owning a canonical
  Latin name;
- all focused positive, negative, legacy-isolation, Ruff, Pyright, and 982
  related regression checks passed;
- the live Bentley run still rejected `37:Naim`, proving that the mapping
  contract assumed by the candidate did not activate in that run;
- the full seven-sample run was stopped after Bentley to avoid spending API
  quota on a candidate that could not exercise its intended route;
- the candidate implementation and its tests were removed. The incomplete run
  remains isolated evidence and is not an admission artifact.

The next checkpoint records only integer counts for context-proposed ASR
mappings and their validator consumption. It must establish where evidence is
lost before any new behavior is introduced. Numeric localization remains a
separate observed retry-efficiency issue, not the cause of the retained Bentley
recovery failure.

**Canonical-evidence observation checkpoint (2026-08-30)**

- candidate telemetry now includes a versioned `canonical_evidence` object with
  counts only; it contains no terminology text, subtitle text, cue identifiers,
  canonical names, prompts, responses, endpoint data, or credentials;
- collection is enabled only by an isolated candidate identity, leaving the
  production legacy path free of the additional scan;
- aggregation validates non-negative integer counts, verifies that every source
  match is classified as supported or rejected, and refuses to mix snapshots
  with and without the evidence;
- a fresh Bentley-only run used the development-admitted behavior at that
  checkpoint, `phase8-local-preservation-repair-v1`, at concurrency 20 and batch size
  20; it completed with 244 requests, 631,932 tokens, and 87,449 reasoning tokens;
- its 48 formatted terminology lines contained five ASR-labelled lines, one
  parseable canonical mapping, one source match, zero supported matches, and one
  ownership-gate rejection;
- the other four ASR-labelled lines cannot yet be called lost canonical names:
  they may be numeric, translated-target, or otherwise intentionally non-Latin
  corrections. This checkpoint therefore does not justify broadening the parser;
- the fresh run's completion is not an admission result because model output and
  segmentation differed from the frozen run and its workload cost increased.
  Only the context-to-validator count evidence is used here.

The next safe behavior candidate must isolate the one observed
`parseable + source-matched + ownership-rejected` route and demonstrate a
general evidence condition stronger than spelling similarity alone. If that
condition cannot be established from development data, retain the current
validator and improve context evidence production instead.

**Rejected structured-ASR ownership series (2026-08-30)**

- four Bentley-only revisions tested an explicit canonical field, narrow
  entity-slot ownership, and a candidate-only identifier-caption exception;
- focused tests, the full translation regression suite, Ruff, and Pyright passed
  before every live rerun, while production remained on the legacy identity;
- the runs produced incompatible activation evidence on the identical workload:
  one supported source match, then zero, then eight, then zero;
- the third run corrected one target name but remained a recovery result; the
  fourth run completed but omitted that name and consumed 262 requests, 676,870
  tokens, and 91,881 reasoning tokens;
- therefore neither completion nor a correct isolated name is admission
  evidence. Route activation is stochastic and efficiency exceeds the intended
  bounded change;
- every structured-ASR behavior flag, prompt schema, parser promotion, ownership
  exception, and target-script exception was removed. The count-only telemetry
  stays because it is privacy-safe observation, and all incomplete/complete run
  artifacts remain isolated by revision.

Do not retry this design by adding more spelling thresholds or product-specific
suffix lists. Revisit canonical ownership only when the evidence source is
deterministic or independently repeated. The next Phase 8 candidate must come
from a stable typed diagnostic and must preserve the same five-percent quality
and efficiency gates.

**Rejected numeric-equivalence follow-up (2026-08-30)**

- `phase8-local-preservation-numeric-equivalence-v2` isolated a real validator
  bug: integral ten-thousands values lost trailing zeroes during formatting, so
  `300000 -> 30万` and `400000 -> 40万` were treated as missing numbers;
- focused tests and an offline replay against all seven frozen v1 outputs showed
  that only those two Bentley false positives were removed; smaller values such
  as `3万` remained invalid for `300000`;
- the Bentley-only live run reduced recovery diagnostics from three to two but
  still produced a recovery artifact;
- the full seven-sample development run completed all samples and improved hard
  static failures from eight to six, while empty and placeholder counts both
  improved from one to zero;
- the frozen efficiency comparison rejected the revision: request count was
  1.0350 times baseline and passed, but tokens were 1.0526, wall duration 1.0720,
  and reasoning-enabled requests 1.0504 times baseline;
- model and segmentation variance account for unrelated diagnostic movement,
  but the established admission contract deliberately treats that end-to-end
  cost as part of candidate risk. The five-percent limits must not be relaxed to
  rescue a narrowly useful rule;
- all numeric-equivalence behavior flags and tests were removed. The complete
  run artifacts and comparison report remain as evidence; production and the
  development-admitted `phase8-local-preservation-repair-v1` revision were
  unchanged at that checkpoint.

Do not rerun this exact candidate without a lower-variance causal evaluation or
a cheaper activation design. Continue with a different stable typed diagnostic
instead of broadening numeric exceptions.

**Rejected missing-key follow-up (2026-08-30)**

- frozen v1 telemetry contained only two `schema.missing_key` observations, one
  in the canal sample and one in the Japan sample;
- the candidate was constrained to one omitted key, no extra key, one
  non-reasoning local translation, complete-batch revalidation, and unchanged
  legacy retry fallback;
- focused positive, mixed-schema, invalid-local-result, isolation, and planner
  tests passed before the live run;
- the smaller canal sample completed with 157 requests, 308,025 tokens, and
  107,582 ms wall duration, but emitted no missing-key diagnostic and therefore
  never exercised the candidate route;
- the run was not expanded to the remaining development corpus. All behavior
  flags and tests were removed, while the isolated artifact remains as explicit
  non-activation evidence;
- provider-format omissions this sparse must be tested with deterministic
  response replay before another behavioral candidate is justified.

The next Phase 8 candidate must have repeatable route activation on development
data. Historical aggregate frequency alone is insufficient when fresh model
runs cannot reproduce the diagnostic.

**Rejected untranslated-output repair follow-up (2026-08-30)**

- `translation.untranslated` was repeatable in Bentley and the multi-speaker
  Topher sample across multiple independent runs, so it passed the initial route
  frequency screen;
- the candidate rewrote at most five affected keys without reasoning, preserved
  all other batch entries, revalidated the complete response, and retained the
  full legacy fallback;
- the Bentley run activated the intended route, but the local rewrite remained
  Latin-only and failed the same target-script check;
- the affected subtitle was a short list of vehicle model names. The human gold
  also preserves those names in Latin script, proving this instance was a
  validator false positive rather than a translation omission;
- the candidate still ended with three failed items and a recovery artifact, so
  no full-corpus run was justified. All untranslated-repair flags and tests were
  removed;
- at that checkpoint, the generic local-response-key helper remained as a
  behavior-neutral extraction of the then-admitted preservation repair and
  passed the full regression suite. The final Phase 9 rollback later removed it
  after the candidate path and its last caller were deleted.

Do not retry local translation for this diagnostic. First design a narrow,
provider-independent identifier-list classifier and prove that it accepts model
captions while still rejecting ordinary English clauses containing a model name.

**Rejected identifier-caption exemption follow-up (2026-08-30)**

- `phase8-identifier-caption-exemption-v1` used a narrow provider-independent
  classifier with ordinary-English negative tests and inherited only the
  local-preservation repair admitted at that checkpoint;
- a Bentley evidence run inspected 422 cues and exercised one exemption, removing
  that file's false untranslated/placeholder failure;
- the complete seven-sample run inspected 3,680 cues and still exercised exactly
  one exemption, proving that the new route had extremely narrow causal reach;
- hard static failures improved from eight to six, empty and placeholder outputs
  improved from one each to zero, but one real adjacent duplicate appeared in the
  mixed Japanese/English sample;
- against the admitted v1 run, requests were 1.0445 times baseline and passed,
  while tokens were 1.0527, wall duration 1.1151, and reasoning-enabled requests
  1.1223 times baseline and failed the frozen limits;
- all candidate-only behavior, activation telemetry, evaluator exceptions, and
  tests were removed. The isolated outputs and reports remain available for
  audit; production remained legacy and the sole development-admitted behavior
  at that checkpoint was `phase8-local-preservation-repair-v1`.

Do not weaken target-script validation or the efficiency gates to recover this
single caption. The next behavior candidate must have materially broader,
repeatable typed-diagnostic activation and an execution path that does not expand
reasoning use.

**Phase 9 blind holdout rejection (2026-08-30)**

- the shadow runner accepts `--split holdout` only with the explicit
  `--blind-holdout` safeguard and rejects every holdout `--sample-id`, preventing
  favorable-subset selection;
- task stdout and stderr are suppressed at the file-descriptor level for blind
  runs, and sample names are redacted from progress output;
- an initial run was invalidated and stopped as soon as unsuppressed validation
  text was observed. It is excluded from every report and comparison;
- the valid baseline and candidate runs each processed all five frozen holdout
  samples with DeepSeek V4 Flash, concurrency 20, batch size 20, reflection
  enabled, and `explicit-client-no-disk-cache` state;
- both aggregate reports had zero empty targets, placeholders, reasoning leaks,
  source copies, untranslated targets, and adjacent duplicate risks. Their five
  hard counts were identical declarations from non-exact sample structure, so
  the candidate produced no hard-quality improvement;
- baseline totals were 1,141 requests, 2,387,732 tokens, 31 reasoning-enabled
  requests, and 693,624 ms wall duration;
- candidate totals were 1,251 requests, 2,578,472 tokens, 38 reasoning-enabled
  requests, and 731,839 ms wall duration;
- request count rose 9.64%, tokens 7.99%, wall duration 5.51%, and reasoning-
  enabled requests 22.58%. Every frozen efficiency gate failed despite complete
  workload identity, cache-state parity, and repair-shadow coverage;
- `phase8-local-preservation-repair-v1` is rejected for rollout. Its revision
  switch, local rewrite path, factory plumbing, and candidate-only tests were
  removed; production behavior never changed.
- final rollback verification passes all 2,289 selected non-integration tests
  with 35 external integration tests deselected. Ruff, Pyright, and `git diff
  --check` pass, and a repository search finds no remaining local-preservation
  revision switch, factory argument, runtime repair function, or candidate-only
  test.

Do not inspect holdout subtitle files or per-cue differences and do not retune
thresholds from this result. Future work returns to development-only observation
until a new, broadly activated candidate independently passes all Phase 8 gates.

**Exit gate**

Development corpus quality is no worse, hard gates pass, and efficiency remains
within budget.

### Phase 9: Blind holdout evaluation

**Objective**

Test generalization without designing against the test material.

**Procedure**

1. freeze candidate revision and configuration;
2. verify holdout hashes;
3. run legacy and candidate under equivalent cache conditions;
4. generate aggregate metrics;
5. perform blinded pairwise human review when required;
6. record preference, severe errors, tokens, calls, and latency;
7. accept or reject the candidate without threshold tuning.

**Failure handling**

- Reject the candidate if any severe regression appears.
- Do not inspect and patch one holdout sentence.
- If detailed analysis is unavoidable, reclassify that sample as development,
  replace the holdout, and repeat the freeze process.

**Exit gate (not met by the rejected candidate)**

Required condition: every acceptance gate in the architecture plan passes. The
evaluated candidate did not satisfy this condition.

### Phase 10: Controlled product rollout

**Full-pipeline state: not entered because the historical Phase 9 exit gate
failed. The 2026-09-05 local app update includes later scoped fixes and does not
constitute full-pipeline acceptance or public release.**

**Objective**

Deploy without losing rollback capability.

**Rollout order**

1. internal opt-in;
2. local developer default with release default still legacy;
3. limited release opt-in;
4. product default with legacy fallback;
5. legacy removal only after several stable releases.

**Monitoring**

- task failure and recovery rates;
- empty translation count;
- repair count by rule ID;
- fallback frequency;
- token and latency distribution;
- model-specific failures;
- user exports after successful completion;
- reports of missing, repeated, or displaced translation.

**Rollback trigger**

Immediately return to legacy default when:

- structural corruption occurs;
- recovery rate rises materially;
- empty or placeholder translation escapes final validation;
- speaker ownership regresses;
- mixed-language text is lost;
- token or latency budget exceeds its approved range;
- a provider-specific branch changes results for unrelated providers.

## 9. Test Strategy

### 9.1 Unit tests

Every new rule or type requires tests for:

- normal positive detection;
- correct non-detection;
- empty and missing fields;
- malformed source data;
- ambiguous language or entity cases;
- single-cue and multi-cue windows;
- speaker changes;
- mixed-language text;
- idempotency;
- deterministic serialization.

### 9.2 Legacy parity tests

Parity tests must compare:

- validation boolean;
- accepted candidate;
- diagnostic category;
- retry decision;
- reasoning decision;
- repair window;
- fallback decision;
- final Chinese text;
- API call count and order;
- cache key and cache hit behavior.

Where exact parity is impossible because of an approved bug fix, add a named
behavior-change fixture and explain why the new decision is safer.

### 9.3 Corpus tests

Corpus tests run outside normal CI because the source data is local. They must:

- validate hashes;
- process every development sample;
- write machine-readable and readable reports;
- compare against the frozen baseline;
- identify new severe diagnostics;
- summarize improvements and regressions separately;
- fail with a nonzero status when a hard gate fails.

### 9.4 Holdout tests

Holdout tests run only at explicit gates. Ordinary development commands must not
include holdout samples by default.

### 9.5 Integration tests

External-provider tests require explicit authorization and must record:

- provider;
- model;
- endpoint family without credentials;
- concurrency and batch size;
- cache state;
- retry policy;
- token and latency telemetry.

Provider instability must be distinguished from translation-quality regression.

### 9.6 Desktop smoke tests

Before a product rollout, verify:

- backend startup in packaged mode;
- model and key settings still load;
- translation starts once and does not open a duplicate app;
- live results update;
- stop and close operations finish cleanly;
- recovery output is preserved on failure;
- final SRT is saved and visible immediately;
- export buttons work;
- macOS and Windows paths do not rely on a development checkout;
- the feature flag can restore the legacy path.

## 10. Required Verification Commands

Use the project's existing toolchain. Commands below define the expected checks;
adjust paths only when the corresponding implementation files exist.

### 10.1 Focused tests during development

```bash
uv run pytest tests/test_translate/test_llm_validator.py -q
uv run pytest tests/test_translate/test_llm_translator.py -q
uv run pytest tests/test_split/test_boundary.py -q
uv run pytest tests/test_split/test_split_core.py -q
uv run pytest tests/test_scripts/test_translation_quality_manifest.py -q
uv run pytest tests/test_scripts/test_translation_quality_metrics.py -q
```

### 10.1.1 Translation-quality evaluator

```bash
export SUBFORGE_TRANSLATION_CORPUS_ROOT=/path/to/local/corpus

uv run python scripts/evaluate_translation_quality.py discover \
  --root "$SUBFORGE_TRANSLATION_CORPUS_ROOT" \
  --output artifacts/translation-quality/corpus.local.json

uv run python scripts/evaluate_translation_quality.py validate \
  --root "$SUBFORGE_TRANSLATION_CORPUS_ROOT" \
  --manifest artifacts/translation-quality/corpus.local.json

uv run python scripts/evaluate_translation_quality.py baseline \
  --root "$SUBFORGE_TRANSLATION_CORPUS_ROOT" \
  --manifest artifacts/translation-quality/corpus.local.json \
  --output-dir artifacts/translation-quality/baselines/<run-id>
```

The local manifest and reports remain under ignored `artifacts/`. Holdout details
are redacted by the report writer and must not be re-enabled during normal rule
development.

### 10.2 Translation and split regression set

```bash
uv run pytest tests/test_translate tests/test_split -m "not integration" -q
```

### 10.3 Full local verification

```bash
uv run ruff check subforge backend launcher.py scripts
uv run pyright subforge backend launcher.py scripts
uv run pytest -m "not integration" -q
```

### 10.4 Frontend verification when settings or product wiring changes

```bash
cd frontend
npm run test
npm run lint
npm run build
```

### 10.5 Package verification before rollout

Use the repository's supported desktop build and smoke-test scripts. Packaging is
not required for purely internal evaluator changes, but it is mandatory before
changing the product default.

An explicit local app update also requires packaging checks. Record the installed
path, version/build identity, source-to-package verification, signing, runtime
checks, and actual rollback method. Keep local delivery and public release
separate from algorithm adoption. Passing tests proves only their covered
behavior; successful startup proves runtime availability, not translation quality.
Documentation-only updates need consistency/link and diff checks, not another
algorithm suite, model run, or app build when the implementation has not changed.

## 11. Review Checklist for Every Change

### 11.1 Scope

- [ ] The change belongs to one migration phase.
- [ ] Structural and behavioral changes are not mixed.
- [ ] Unrelated user modifications remain untouched.
- [ ] No historical full sentence was added as a production replacement.
- [ ] No provider-specific behavior leaked into unrelated providers.

### 11.2 Correctness

- [ ] Subtitle keys and timing remain intact.
- [ ] Source text remains intact.
- [ ] Empty and placeholder outputs cannot be accepted.
- [ ] Numbers, identifiers, entities, negation, and speaker ownership are checked.
- [ ] Repairs are idempotent.
- [ ] A failed repair cannot overwrite a valid prior result.
- [ ] Recovery output remains available after terminal failure.

### 11.3 Architecture

- [ ] New types are immutable where appropriate.
- [ ] Feature extraction is not repeated unnecessarily.
- [ ] Diagnostics have stable IDs.
- [ ] Human-readable messages are not parsed for control flow.
- [ ] Rule modules have focused responsibilities.
- [ ] Orchestration does not contain linguistic case libraries.
- [ ] Dependencies still point inward toward stable domain types.

### 11.4 Efficiency

- [ ] No hidden extra LLM call was added.
- [ ] Reasoning remains limited to confirmed high-risk windows.
- [ ] Cache behavior is unchanged or intentionally versioned.
- [ ] Retry loops have explicit budgets.
- [ ] Repeated feature extraction was not introduced.
- [ ] Baseline and candidate cache states are comparable.

### 11.5 Tests and evidence

- [ ] Focused tests pass.
- [ ] Full non-integration tests pass when blast radius requires them.
- [ ] Ruff and Pyright pass for touched production paths.
- [ ] Corpus report is attached for behavioral changes.
- [ ] Holdout was not used for rule design.
- [ ] Performance telemetry is present for LLM changes.
- [ ] Remaining uncertainty is documented.

## 12. Stop Conditions

Stop the current phase and investigate before further edits when:

- the same input produces nondeterministic static metrics;
- manifest provenance cannot establish which files belong together;
- a candidate changes subtitle keys, timestamps, or source text;
- an adapter changes API call count, prompt text, cache keys, or accepted output;
- a typed diagnostic disagrees with its legacy validator unexpectedly;
- a rule requires an exact historical full-sentence translation to pass;
- holdout content has influenced rule design;
- a provider test would use an unapproved paid API;
- unrelated worktree changes make safe integration impossible;
- token or latency growth exceeds the prospectively approved track-specific budget;
- a severe semantic regression appears in any holdout sample;
- the new path cannot immediately fall back to the legacy path.

Do not work around a stop condition by weakening tests or changing the baseline.

## 13. Rollback Procedure

Every migration phase must identify its rollback before implementation.

Minimum rollback capability:

1. feature flag restores the legacy path;
2. cache namespaces remain distinguishable;
3. candidate output never overwrites the legacy or gold file;
4. schema migrations are additive until rollback is no longer needed;
5. recovery files state which path produced them;
6. removing a new module does not require reconstructing deleted legacy code.

If a production regression occurs:

1. disable `quality_pipeline_v2` by default;
2. preserve failing inputs, diagnostics, and telemetry locally;
3. confirm the legacy path still succeeds;
4. classify the failure by stable rule ID or orchestration stage;
5. add a development regression fixture only after preserving an independent
   holdout;
6. fix the general rule, not the one sentence;
7. rerun baseline, development corpus, and holdout gates before re-enabling.

## 14. Reporting Template

Every completed phase should produce a short engineering report with:

```text
Phase:
Revision:
Working-tree baseline:
Component revisions / combination / final retained revision and code hashes:
Evidence-to-version bindings / applicable and invalidated evidence after removal:
Corpus manifest hash:
Development samples:
Holdout samples:
Code paths changed:
Behavior intended to change:
Behavior required to remain identical:
Tests run:
Static baseline comparison:
Hard invariant failures:
Severe semantic regressions:
Improvements:
Regressions:
LLM calls:
Prompt tokens:
Completion/reasoning tokens:
Cache state:
Activation / confirmed repairs / cache hits / demonstrably avoided requests:
Wall time:
Rollback mechanism:
Decision: accept | reject | continue in shadow mode
Per-component retention / proven outcomes / unverified claims:
Local app identity and delivery checks / public release status:
Remaining risks:
```

Reports must describe actual evidence. “Looks better” is not an acceptance
criterion.

## 15. First Implementation Session

The first work session must remain behavior-preserving.

### Step 1: Preserve the workspace baseline

- inspect current revision and worktree state;
- record modified and untracked paths;
- identify files unrelated to translation-quality refactoring;
- do not stage or revert them.

### Step 2: Inventory the 13 three-file groups

- compute hashes and cue counts;
- classify each source, machine, and gold file;
- detect URL-encoded filenames without renaming source data;
- identify exact structural pairs;
- flag advertisement or timing-edited variants;
- record missing model and algorithm provenance as unknown.

### Step 3: Freeze corpus splits

- assign development, validation, and holdout sets from metadata;
- verify category coverage;
- record any missing two-speaker, multi-speaker, mixed-language, or long-video
  category before claiming the corpus complete.

### Step 4: Implement manifest validation

- define the schema;
- load paths relative to the corpus root;
- verify hashes and required fields;
- validate split isolation;
- add malformed-manifest tests.

### Step 5: Implement read-only SRT comparison

- parse source, machine, and gold files;
- compare count, keys, timing, source text, and translation presence;
- support exact pairs and explicitly aligned pairs;
- never modify the source files.

### Step 6: Generate the first baseline report

- run static metrics over the development set;
- run aggregate-only metrics over the frozen holdout;
- store the report under ignored artifacts;
- confirm repeatability with a second run;
- update the architecture plan inventory and status only after evidence matches.

### Step 7: Stop before production refactoring

Review the corpus and evaluator results. Phase 2 begins only after Phase 1 exit
gates pass. The first session must not edit prompts, boundary weights, repair
rules, provider strategies, or final translation output.

## 16. Definition of Done

The complete refactor is done only when:

- the corpus manifest is reproducible and holdout contamination is controlled;
- baseline and candidate reports are deterministic;
- production uses immutable task context and precomputed cue features;
- validators return typed diagnostics with stable rule IDs;
- message text no longer controls repair behavior;
- hard invariants live in focused, tested modules;
- mode policies are explicit and independently testable;
- the repair planner is budgeted, idempotent, and observable;
- English and Chinese boundary rules are declarative and explainable;
- no production rule contains a fixed full translation for a historical passage;
- keys and timing integrity remain 100%;
- empty translations and placeholders remain zero;
- holdout severe semantic regressions remain zero;
- speaker and mixed-language quality do not regress;
- pairwise human preference is no worse than legacy;
- calls, tokens, and latency do not increase by more than the approved budget;
- the product has completed a controlled rollout with a tested fallback;
- legacy code is removed only after several stable releases.

Until all conditions hold, the work remains a migration with a legacy fallback,
not a completed replacement.
