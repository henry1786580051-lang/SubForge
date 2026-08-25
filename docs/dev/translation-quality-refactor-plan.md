# Translation Quality Refactor Plan

Status: Deferred

Last reviewed: 2026-08-23

This document archives the proposed refactor of SubForge's subtitle splitting,
translation validation, and repair pipeline. It is intentionally not an active
implementation specification. Structural migration should begin only after the
evaluation corpus described below is available.

## Why This Work Is Deferred

The current pipeline is functionally mature and covered by a large regression
suite, but its central translation and boundary modules have exceeded a reliably
maintainable size. Replacing them without an independent quality baseline would
risk trading known edge cases for less visible regressions.

The immediate decision is therefore:

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
- `assess_english_boundary` in `subforge/core/split/boundary.py` spans roughly
  1,208 lines and is shared by splitting, translation audit, and recovery paths.
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

The local machine currently contains one verified full gold group:

- `The Sphere Failed. Why Are They Building More`
  - word-level source: 2,637 cues, two speakers, 16:31;
  - machine result: 262 cues;
  - ChatGPT-edited result: 262 cues;
  - 255 of 262 Chinese cues were edited while English and timing stayed fixed;
  - a separate 229-cue version contains manual timing changes and advertisement
    removal, so it requires alignment before timing comparison.

Useful but incomplete groups include:

- Audi Q3: single speaker, word-level source and machine result, no verified
  human gold result;
- five-speaker interview: word-level source and machine result, no verified
  human gold result;
- nuclear-plant video: word-level source and machine result, no verified human
  gold result;
- `Why Japan Builds Like Nowhere Else`: approximately 1:29:40 of mixed English
  and Japanese word-level output, without matching machine and human results;
- GR Corolla and reading/interview refinement variants under `AppData`: useful
  regression candidates, but their provenance is not recorded well enough to
  treat them as independent gold data;
- historical Lexus and Russian source/result pairs in Git history, without
  verified human refinements.

This inventory is enough to build an evaluation harness and verify a
behavior-preserving refactor. It is not enough to authorize a general algorithm
replacement.

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
- LLM calls, token use, and latency do not increase by more than 5%;
- reasoning is restricted to demonstrably high-risk windows;
- no production rule contains a complete fixed translation for one historical
  source passage;
- orchestration modules remain small enough to review, with ordinary rule
  functions targeting cyclomatic complexity no greater than 15.

## First Safe Future Step

When video production has produced enough material, begin only with the corpus
manifest and behavior-preserving session/diagnostic types. Do not simultaneously
change prompts, rule semantics, and orchestration. This keeps every quality
change measurable and reversible.
