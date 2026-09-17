# Bridge / Seltos development audit — 2026-09-11

Two newly supplied triples were compared without modifying their six files. The
bridge pair has 263 cues and the vehicle pair 743. Neither reference changes cue
timestamps. References change 205 and 188 translations respectively, but also
change 16 and 12 source-English cues. These are development samples, not holdout
truth: stylistic edits, reconstructed ASR, and changes of cue ownership require
separate review.

## Retained fix

The percentage guidance selector incorrectly required a word boundary after `%`.
A percentage followed by a space, punctuation, or the end of the cue consequently
failed to activate the existing guidance. It now recognizes bounded numeric
percentages, including decimals, without matching identifier fragments or the
suffix of an oversized numeric token. The hint text and validation policy are
unchanged. Eight positive/negative/target-language checks cover this defect.
This restores an intended route; it is not evidence of improved overall prose.

## Default-off candidate

`domain-precision` is available only through the evaluation harness. It replaces
one generic technical hint with source-selected distinctions for consumption vs
fuel economy, turning measurements, interior components, and bridge/tendering
roles. It does not alter source text, timestamps, numeric validators, cue ownership,
provider configuration, or the application default. Normal exit and exception
rollback are tested. Production must not import the experiment module.

Six source-only development windows were paired using the saved
`zhipu/glm-5.3-flash` profile. No edited reference was sent to the model. This was
an isolated hint probe, not the complete translator/repair pipeline. Both variants
used temperature 0, four workers, one repetition, a 4,096-token output cap and the
application adapter's low-effort reasoning mode. The initial direct transport
preflight was rejected because GLM does not accept disabled thinking; all 12
rejections remain recorded separately. No usage was returned for those failures.

| Successful probe | Baseline | Candidate |
| --- | ---: | ---: |
| Requests | 6 | 6 |
| Prompt tokens | 5,575 | 5,551 |
| Completion tokens | 509 | 528 |
| Total tokens | 6,084 | 6,079 |
| Exact key sets / nonempty values | Pass | Pass |

Semantic review did **not** support promotion. Turning measurement and console
sense remained wrong; a bridge output added unsupported emphasis and a company
rendering was unreliable. Both variants also mishandled a spoken price fragment.
The candidate remains default-off for reproducibility. Near-equal token counts
are not a quality acceptance result, and this one probe establishes neither
full-video cost nor independent/holdout quality.

## Validation and delivery

- Baseline: 2,768 non-integration tests passed.
- Final: 2,778 passed, 35 integration tests deselected; one existing warning.
- Changed-file Ruff and targeted Pyright passed; diff whitespace check passed.
- All six supplied SRT hashes remained unchanged.
- Existing UI changes were preserved. No app build, installation, GitHub push,
  or release was performed in this algorithm audit.

Local evidence, source manifest/hashes, frozen candidate hashes, exact probe
outputs, failed transport records and decisions are stored under ignored
`artifacts/translation-quality/runs/20260911-bridge-seltos/`. Subtitle text is not
included in tracked fixtures. Further semantic work should address context and
ownership using a new frozen candidate, rather than adding more unconditional
instructions or a phrase-replacement dictionary for these two videos.


### Local delivery — 2026-09-11

After the audit, the user requested an app update. The retained percentage-selector fix was installed in place at `~/Applications/SubForge.app`. Installed Python bytecode behavior, signature and native startup passed. Only a ZIP of the replaced module was retained for rollback. The default-off domain experiment was not installed. No second app bundle or public release was created. Evidence: `app-update.json` in the run directory.
