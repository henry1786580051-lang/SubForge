# Changelog

## v1.1.17 - 2026-08-25

### Improved

- Audits word-level MLX coverage against high-confidence speech activity and restores short omissions only when two differently sized context windows decode the same new words.
- Carries exact-content timing metadata from transcription into subtitle processing, allowing final sentence cues to receive bounded acoustic tail extension without reanalyzing unchanged media.
- Expands spoken English amounts, currency ranges, percentages, years, models, and units before forced alignment, then restores display tokens with character-level mapping that tolerates contractions, sentence splits, and isolated unaligned words.
- Extends cross-cue boundary review for subjects, predicates, complements, comparison phrases, sentence adverbials, names, numeric heads, and dialogue continuations while retaining selective reasoning for confirmed semantic risks.
- Isolates third-party model downloads in cancellable worker processes and closes decoded audio resources deterministically.
- Stores configured API credentials in the operating-system credential store when available and resolves only the secret needed by the active operation.

### Fixed

- Fixed short spoken phrases disappearing inside otherwise covered MLX transcription ranges.
- Fixed compact numeric tokens receiving only a fraction of their spoken duration when one unrelated alignment word was missing or tokenized differently.
- Fixed duplicate word cleanup shortening the surviving subtitle instead of retaining the later acoustic end time.
- Fixed sentence cues ending before sustained speech, including cases where VAD continuation crosses one short following cue; extension remains bounded and never creates timeline overlap.
- Fixed isolated word-level hallucinations over music surviving without corroborating speech evidence.
- Fixed translation progress overcounting repeated preview updates and final subtitle processing losing access to the source media timing context.
- Fixed repeated macOS Keychain prompts caused by resolving every saved provider credential during application startup.
- Fixed DMG staging replacing an explicitly configured stable macOS signing identity with an ad-hoc signature.
- Fixed long-running model downloads retaining uncancellable workers after the user cancels a task.

### Validation

- Python: `1786 passed, 11 skipped`.
- Ruff and whitespace validation passed.
- Regression coverage includes MLX omission recovery, mixed-language alignment, spoken-number restoration, conservative sentence-tail extension, cross-cue boundary ownership, credential persistence, and cancellable model downloads.
- The corrected Heathrow sample exports 303 cues with zero overlaps and zero empty translations; reported amount, date, comparative, political-subject, and sentence-tail cases were checked against the source audio.

### Packaging

- macOS Apple Silicon: `SubForge-1.1.17-macos-arm64.dmg`.
- Windows x64: `SubForge-1.1.17-windows-x64-setup.exe`, built and smoke-tested by GitHub Actions.
- Whisper, forced-alignment, Community-1, ECAPA, and DeepFilterNet models remain separate on-demand downloads.

## v1.1.16 - 2026-08-22

### Improved

- Protects open complements, fixed expressions, and cross-cue dependency pairs during sentence boundary selection so semantic units are not split merely to satisfy a preferred subtitle length.
- Uses document-level evidence before correcting recurring ASR homophones, keeping genuine acoustic terms intact when the surrounding transcript does not support a correction.
- Detects manufacturer-introduced product identifiers and carries their canonical form into later translation and validation windows.
- Distinguishes audio demonstrations from visual demonstrations and expands automotive guidance for vents, one-touch windows, equipment parity, and figurative reviewer language.
- Keeps native reasoning selective: ordinary translation and format validation remain non-reasoning operations, while confirmed ownership, word-order, and proper-name risks receive focused review.

### Fixed

- Fixed phrase fragments such as `write home about`, `sound system`, and open `put` complements being separated across subtitle boundaries.
- Fixed recurring `bass/base` ASR ambiguity being corrected without sufficient document evidence.
- Fixed official seat, package, and system names drifting between adjacent translation batches.
- Fixed quiet components being translated as human behavior, audio playback being described as something shown on screen, and automatic window controls losing their established meaning.
- Fixed Traditional Chinese characters and Latin proper names not owned by the source or confirmed context leaking into Simplified Chinese output.

### Validation

- Python: `1682 passed, 11 skipped`.
- Ruff and Pyright passed with `0 errors`; whitespace validation passed.
- Full DeepSeek V4 Flash regression: 618 bilingual cues with zero empty translations, placeholders, untranslated cues, Traditional-script leakage, or timeline overlaps.
- The full regression completed 351 successful API requests with a 66.6% prompt-cache hit ratio; native reasoning remained confined to confirmed high-risk review windows.

### Packaging

- macOS Apple Silicon: `SubForge-1.1.16-macos-arm64.dmg`.
- Windows x64: `SubForge-1.1.16-windows-x64-setup.exe`, built and smoke-tested by GitHub Actions.
- Whisper, forced-alignment, Community-1, ECAPA, and DeepFilterNet models remain separate on-demand downloads.

## v1.1.15 - 2026-08-19

### Improved

- Adds an opt-in hybrid-language mode for fixed-language WhisperX tasks. The selected language remains primary, while sustained foreign-language speech can be locally re-decoded and aligned with its own model.
- Persists per-segment language metadata across SRT task boundaries so mixed English, Japanese, Korean, and other source material reaches alignment and translation validation intact.
- Refines long-video processing into 30-minute overlapping chunks and retries only coverage-failed sections at progressively smaller sizes instead of discarding an otherwise valid transcription.
- Adds document-level alias and lexical evidence for recurring names and technical terms, while requiring repeated support before correcting phonetic ASR variants.
- Preserves imagery, contrast, irony, and rhetorical force with concise idiomatic Chinese; native reasoning remains limited to confirmed semantic ownership, difficult word order, and proper-name repairs.

### Fixed

- Fixed hour-long WhisperX jobs failing at a chunk boundary or appearing to finish the remaining video after an early single-worker failure.
- Fixed short Japanese speech being mislabeled as Korean from isolated weak probes; language confirmation now uses sustained voiced evidence, script evidence, and local continuity.
- Fixed forced-alignment echoes such as repeated suffix words and numeric `two/too` fragments surviving into final word-level subtitles.
- Fixed forced alignment erasing otherwise valid native MLX word timestamps when only part of a segment lacked alignment coverage.
- Fixed complete provisional translations being reported as failed even after document-level finalization could repair and validate every subtitle key.
- Fixed Japanese source text being accepted as a Chinese translation merely because both scripts contain CJK characters.
- Fixed recurring cross-cue errors involving modifiers, comparative complements, proper names, acronyms, localized units, open clauses, and adjacent semantic duplication.

### Validation

- Python: `1573 passed, 11 skipped`.
- Frontend: `3 passed`; ESLint and Next.js 16.3 production build passed.
- Ruff, Pyright (`0 errors`), whitespace validation, and VitePress production build passed.
- Full DeepSeek V4 Flash regression: 247 bilingual cues, zero empty translations, placeholders, untranslated cues, timeline overlaps, or unstable English boundaries.
- The full-sample source stream retained 99.87% normalized token similarity; differences were verified as duplicate ASR echoes or explicit filler removal.
- The mounted DMG passed packaged FastAPI, FFmpeg, MLX Metal, PyTorch MPS, WhisperX, pyannote, and speaker-assignment smoke checks.

### Packaging

- macOS Apple Silicon: `SubForge-1.1.15-macos-arm64.dmg`.
- Windows x64: `SubForge-1.1.15-windows-x64-setup.exe`, built and smoke-tested by GitHub Actions.
- Whisper, forced-alignment, Community-1, ECAPA, and DeepFilterNet models remain separate on-demand downloads.

## v1.1.14 - 2026-08-15

### Improved

- Added conservative speech-gap recovery around MLX transcription and forced alignment without globally rewriting correct timestamps.
- Kept single-speaker and dialogue translation strategies separate while expanding general subject, modifier, complement, connector, and adjacent-duplication audits.
- Improved complete live previews, periodic recovery snapshots, task cancellation, and automatic recovery-file loading after a failed translation.
- Added a redesigned DMG drag-install window, duplicate build-output cleanup, and packaged MLX Metal, MPS, WhisperX, FFmpeg, and backend smoke checks.
- Added the ChatGPT SRT final-review protocol for preserving cue keys, timelines, source text, and semantic ownership during manual review.

### Validation

- Python: `1405 passed`; 35 integration tests excluded by release configuration.
- Frontend tests, Ruff, Pyright, Next.js, VitePress, and packaged macOS smoke checks passed.

### Packaging

- macOS Apple Silicon: `SubForge-1.1.14-macos-arm64.dmg`.
- Windows x64: `SubForge-1.1.14-windows-x64-setup.exe`.

## v1.1.13 - 2026-08-12

### Improved

- Separates single-speaker and multi-speaker subtitle strategies: monologues retain the proven local repair path, while dialogue receives anonymous speaker context, turn-aware boundary auditing, and narrowly scoped multi-turn repairs.
- Uses DeepSeek V4 Flash native reasoning only for confirmed semantic ownership or difficult Chinese word-order repairs. Routine translation, formatting checks, candidate audits, retries, and deterministic dialogue cleanup remain non-reasoning operations.
- Adds conservative dialogue boundary protection for numeric ranges, proper names, comparison complements, discourse frames, trailing fillers, modifiers, coordinated subjects, and tightly edited speaker handoffs.
- Uses a second speaker-embedding pass to confirm proposed short-interjection and label corrections before changing production speaker assignments.
- Treats the two-speaker preset as two primary speakers with room for short advertisement or inserted voices; fixed-count mode remains strict.
- Extends global terminology context to use spoken letter-by-letter spellings as strong evidence for one canonical person or product name.

### Fixed

- Fixed multi-speaker translations shifting questions, subjects, comparisons, temporal frames, or conclusions into adjacent subtitle keys.
- Fixed repeated Chinese conclusions, isolated conjunction subtitles, stranded numeric ranges, and placeholders such as “merged into previous” surviving final validation.
- Fixed native-reasoning requests consuming the output budget without returning final JSON by using low reasoning effort, a larger final-answer budget, and bounded non-reasoning fallback.
- Fixed LLM provider changes reusing a stale global client or cache entry from a different Base URL or API key.
- Fixed an explicitly cleared custom prompt falling back to an older persisted prompt.
- Fixed the subtitle workspace allowing the active model to diverge from Settings; it now displays the selected provider and model as read-only task configuration.
- Fixed standalone web addresses being reported as untranslated content while retaining translation requirements for URL calls to action.
- Fixed exported dialogue subtitles exposing internal speaker labels.

### Validation

- Python: `1201 passed, 11 skipped`.
- Multi-speaker translation regression suite: `22 passed` after final dialogue-sequence additions.
- Ruff, whitespace validation, and Next.js production build passed.
- Full 54-minute two-speaker DeepSeek V4 Flash test: 826 bilingual cues, zero real empty translations, zero placeholders, zero suspicious duplicates, and zero hard Chinese boundary failures.
- Compared with the initial dialogue run, reasoning requests fell from 68 to 32 and reasoning tokens from 260,392 to 79,528 while preserving the final quality audit.

### Packaging

- macOS Apple Silicon: `SubForge-1.1.13-macos-arm64.dmg`.
- Windows x64: `SubForge-1.1.13-windows-x64-setup.exe`, built and smoke-tested by GitHub Actions.
- Whisper, forced-alignment, Community-1, ECAPA, and DeepFilterNet models remain separate on-demand downloads.

## v1.1.12 - 2026-08-11

### Improved

- Uses DeepSeek V4 Flash native reasoning selectively for difficult semantic ownership, terminology, and Chinese word-order repairs instead of spending the reasoning budget on routine translation and JSON-format checks.
- Records reasoning requests, final-answer availability, accepted and rejected repairs, and fallback requests so reasoning efficiency can be diagnosed without exposing chain-of-thought content.
- Adds bounded whole-document evidence for vehicle names, trims, brands, and technical terms that may be absent from sampled context windows.
- Promotes explicit ASR corrections from global terminology while requiring repeated document support before applying a phonetic model-name correction.
- Strengthens Chinese subtitle fidelity for negated comparisons, irony, direct answers, colloquial acronyms, elliptical units, numeric self-corrections, automotive controls, suspension descriptions, and trim names.
- Expands conservative English boundary protection for `rev matching`, ordinal gear names, separated `take ... away` constructions, revised-component phrases, and negated comparisons.

### Fixed

- Fixed native-reasoning calls occasionally returning reasoning without a usable final JSON answer; the task now retries through a bounded non-reasoning fallback instead of losing the repair.
- Fixed globally sampled context missing a recurring model or phonetic ASR variant that appeared outside the beginning, middle, and ending transcript windows.
- Fixed correct source meaning being weakened or changed by literal translations of domain phrases such as racing line, fresh slate, vehicle trim names, and ride-quality terms.
- Fixed abandoned spoken numbers being translated twice and elliptical automotive values such as `20 softer` or `1 to 2,000 RPM` losing their intended scale.
- Fixed added editorial labels such as “讽刺地” entering the final subtitle when the label was not spoken in the source.
- Removed stale Finder-style ` 2` source and test copies that could be collected by pytest and fail CI against obsolete interfaces.

### Validation

- Python: `1111 passed, 11 skipped`.
- Translation and boundary regression suite: `337 passed`.
- Ruff, whitespace validation, and Next.js production build passed.
- Verified that repair validation preserves subtitle keys, source ownership, timeline data, and already-correct neighboring translations.

### Packaging

- macOS Apple Silicon: `SubForge-1.1.12-macos-arm64.dmg`.
- Windows x64: `SubForge-1.1.12-windows-x64-setup.exe`, built and smoke-tested by GitHub Actions.
- Whisper, forced-alignment, Community-1, ECAPA, and DeepFilterNet models remain separate on-demand downloads.

## v1.1.11 - 2026-08-09

### Improved

- Publishes the final subtitle snapshot with the task-completion event so the editor can update immediately without waiting for a later poll.
- Writes SRT files through one atomic UTF-8 BOM and CRLF path for reliable Microsoft Word, WPS, macOS, and Windows interoperability.
- Extends conservative English boundary normalization for hyphenated attributive phrases, `what`-clause subjects, `one of these/those` noun phrases, and vehicle-type compounds.
- Preserves a natural boundary after complete phrases such as `ease of use` when the following prepositional phrase belongs to the next readable subtitle.

### Fixed

- Fixed completed translation results occasionally remaining stale in the preview because the completion event raced the final preview update.
- Fixed exported SRT files opening as mojibake in Microsoft Word despite appearing correct in WPS or UTF-8-aware editors.
- Fixed sentence ownership errors around `day-to-day / life`, `Toyota electric / vehicles`, and `what I really wanted to show / was`, which could move Chinese meaning into the wrong cue.
- Fixed SRT reads preserving a BOM as subtitle content by decoding with `utf-8-sig`.

### Validation

- Python: `1020 passed, 11 skipped`.
- Ruff, whitespace validation, and Next.js production build passed.
- Added focused regression coverage for terminal task snapshots, SRT byte encoding, import/export round trips, and translation boundary validation.

### Packaging

- macOS Apple Silicon: `SubForge-1.1.11-macos-arm64.dmg`.
- Windows x64: `SubForge-1.1.11-windows-x64-setup.exe`, built and smoke-tested by GitHub Actions.
- Whisper, forced-alignment, Community-1, ECAPA, and DeepFilterNet models remain separate on-demand downloads.

## v1.1.10 - 2026-08-08

### Improved

- Reads the active LLM provider, Base URL, API key, and model as one immutable task snapshot instead of combining independently cached fields.
- Sends the frontend's expected provider and model with each subtitle task and rejects stale page state before any model request is issued.
- Makes the full-pipeline test runner provider-neutral, with explicit service and model overrides plus preflight validation.
- Extends English boundary normalization for vehicle model years, mixed-number measurements, opinion markers, fixed phrases, coordinated predicates, `what ... use ... for`, and instrument-name boundaries.
- Adds focused Chinese style validation for resultative degree statements, vehicle use-case subjects, discourse markers, numeric shorthand, and exact ten-thousand number equivalents.
- Runs boundary normalization again after short-cue merging so a merge cannot expose a new unchecked dependency.

### Fixed

- Fixed the video pipeline test silently preferring legacy `MIMO_*` variables over an explicitly requested DeepSeek-compatible configuration.
- Fixed task requests silently ignoring a user-visible model change and calling whichever provider happened to remain active in backend settings.
- Fixed model discovery and connection tests reading stale flat LLM fields instead of the active provider profile.
- Fixed legacy mixed-provider profiles retaining another provider's endpoint or model after switching services; credentials remain isolated and preserved.
- Fixed invalid combinations such as a MiMo endpoint with a DeepSeek model reaching the network.
- Fixed DMG assembly adding Finder/provenance metadata after signing; the exact app inside the writable image is now cleaned and strictly verified before final compression.
- Fixed subtitle boundaries splitting `2026 / Ford F-150`, `five / and a half`, `don't get me / wrong`, `little puny / RPM gauge`, and related dependent structures.
- Fixed literal or incomplete Chinese results such as “这就是安静模式下它有多安静”, vehicle-use percentages with the wrong subject, and cue-final attributive `的`.

### Validation

- Python: `1013 passed, 11 skipped`.
- LLM routing regression suite: `63 passed`.
- Ruff and Next.js production build passed.
- Verified that MiMo endpoint plus DeepSeek model is rejected before network access and that legacy MiMo variables no longer override a DeepSeek test run.
- Verified the mounted DMG checksum, `/Applications` link, app signature, and `1.1.10` bundle version.

### Packaging

- macOS Apple Silicon: `SubForge-1.1.10-macos-arm64.dmg`.
- Windows x64: `SubForge-1.1.10-windows-x64-setup.exe`, built and smoke-tested by GitHub Actions.
- Whisper, forced-alignment, Community-1, ECAPA, and DeepFilterNet models remain separate on-demand downloads.

## v1.1.9 - 2026-08-08

### Improved

- Finalizes Chinese subtitle punctuation with one final save, one preview serialization, and one editor update instead of repeating temporary SRT writes and full snapshot publication.
- Sends field-level preview patches for large same-shape subtitle updates and removes duplicate completed-task subtitle payloads from WebSocket events while retaining REST and reconnect compatibility.
- Virtualizes the subtitle editor so long files render only visible rows while preserving dynamic row heights, empty-translation navigation, selection, keyboard focus, and inline editing.
- Prefers semantically safer English display boundaries around degree complements such as `resonate so deeply` and `became so widespread`.
- Expands the conservative Chinese boundary audit to cover incomplete reason constructions and stranded `变得` predicates without changing timestamps or source text.

### Fixed

- Fixed DeepFilterNet3 first-run downloads hanging indefinitely at `Enhancing audio with DeepFilterNet3...` after a partial model archive was left in the cache.
- Replaced DeepFilterNet's implicit unbounded model download with a pinned, size- and SHA-256-verified download using bounded connection/read timeouts and atomic extraction.
- Added recovery for incomplete DeepFilterNet3 caches and fallback to the original audio when enhancement is unavailable, so ASR can continue instead of losing the transcription task.
- Isolated DeepFilterNet3 in packaged desktop builds and added download, model-load, and audio-chunk progress reporting plus stalled-worker termination.
- Fixed `in America / is because` being misclassified as a proper-name subject split, which could force a worse `resonate / so deeply` boundary.
- Fixed fragmented English cues causing invented Chinese subjects and unnatural translations such as “我们之所以如此扎根美国”.
- Fixed “之所以……部分原因” and “变得 / 如此” pairs bypassing the mandatory Chinese fluency-repair path.
- Fixed Chinese words beginning with `了`, such as `了解`, being mistaken for a standalone aspect particle and triggering unnecessary repair attempts.
- Fixed the final punctuation stage appearing slow because saving, serialization, WebSocket transfer, and full-table rendering were grouped under the same progress message.

### Validation

- Python: `967 passed, 11 skipped`.
- DeepFilterNet3 regression suite: `41 passed`; verified real inference in both the development runtime and the packaged macOS app.
- Split and translation regression suite: `237 passed`.
- Ruff, ESLint, TypeScript, and Next.js production build passed.
- Verified the reported 153-cue Blue Zone subtitle: affected boundaries now enter repair while the `了解` false positive no longer does.

### Packaging

- macOS Apple Silicon: `SubForge-1.1.9-macos-arm64.dmg`.
- Windows x64: `SubForge-1.1.9-windows-x64-setup.exe`, built and smoke-tested by GitHub Actions.
- Whisper, forced-alignment, Community-1, ECAPA, and DeepFilterNet models remain separate on-demand downloads.

## v1.1.8 - 2026-08-07

### Added

- Added direct navigation from the subtitle-quality empty-translation indicator to the affected subtitle row.
- Repeated clicks cycle through every empty translation while keeping the current selection unchanged and briefly highlighting the destination row.

### Improved

- Treats reliable Chinese magnitude conversions such as `20 thousand` to `两万` as preserved numeric meaning.
- Recognizes domain-introduction idioms such as `health promotion 101` translated as `健康促进基础常识`, without relaxing route numbers, room numbers, years, prices, model names, or technical specifications.
- Uses trimmed translation text consistently in quality totals, completion status, table placeholders, and translated-item counts.

### Fixed

- Fixed valid natural Chinese translations being rejected repeatedly by the numeric-fidelity validator, leaving empty cues in the recovery subtitle and failing the complete translation task.
- Fixed whitespace-only translations appearing complete in parts of the interface while the quality report correctly counted them as empty.

### Packaging

- macOS Apple Silicon: `SubForge-1.1.8-macos-arm64.dmg`.
- Windows x64: `SubForge-1.1.8-windows-x64-setup.exe`, built and smoke-tested by GitHub Actions.
- Whisper, forced-alignment, Community-1, ECAPA, and DeepFilterNet models remain separate on-demand downloads.

## v1.1.7 - 2026-08-05

### Added

- Added an optional dual-model speaker-verification gate: Community-1 proposes uncertain boundary corrections and WeSpeaker ECAPA-TDNN512-LM independently confirms them without storing enrolled voiceprints.
- Added reproducible AMI and VoxConverse diarization benchmarks covering strict DER, automatic speaker count, boundary quality, word-level speaker ownership, overlap protection, stability, and runtime.
- Added bounded automatic speaker detection for 2-10 participants and conservative overlap-aware assignment diagnostics.

### Improved

- Reduced non-overlap AMI word-speaker error from 4.70% to 4.61% with the dual-model verifier while preserving ASR text and timestamps; all sampled conditions remained non-regressive.
- Upgraded the bundled media runtime to pinned, checksum-verified FFmpeg/FFprobe 8.1.2 builds on macOS and Windows.
- Simplified the supported CLI and documentation around import, transcription, subtitle restructuring, translation, review, and export.
- Reduced desktop resources and dependency surface by retaining only modules used by the current Next.js, FastAPI, and pywebview application.

### Fixed

- Fixed semantic speaker-boundary heuristics overriding acoustically correct labels; they now act only as proposals behind conservative acoustic validation.
- Fixed packaged multi-speaker startup after dependency cleanup by retaining Pillow for the pyannote, torchmetrics, and torchvision import chain.
- Fixed stale media routes, configuration sections, and command help exposing features that the current desktop workflow no longer provides.
- Fixed FFmpeg package drift by downloading platform-specific archives with pinned versions, SHA-256 verification, safe extraction, and runtime codec checks.
- Fixed CI explicitly referencing retired test directories instead of discovering the maintained test suite.

### Removed

- Removed the retired PyQt desktop interface and its background-thread implementation.
- Removed unused subtitle burning, ASS styling, video synthesis, TTS, dubbing, bundled fonts, translation resources, legacy assets, commands, and tests.
- Removed PyQt, Fluent Widgets, Edge TTS, fontTools, CairoSVG, and related dependencies from the supported runtime.

### Packaging

- macOS Apple Silicon: `SubForge-1.1.7-macos-arm64.dmg`.
- Windows x64: `SubForge-1.1.7-windows-x64-setup.exe`, built and smoke-tested by GitHub Actions.
- Whisper, forced-alignment, Community-1, ECAPA, and DeepFilterNet models remain separate on-demand downloads.

## v1.1.6 - 2026-08-04

### Improved

- Made semantic splitting aware of the downstream target language so English place names, relative clauses, temporal modifiers, subjects, and predicates are less likely to be separated into awkward Chinese cues.
- Added a document-level Chinese boundary-fluency shortlist and conservative rewrite path for subject, adverbial, modifier, and neighboring-clause ownership problems while preserving subtitle keys, source text, and timestamps.
- Reallocated DeepSeek V4 reasoning to semantic translation audits and first-pass boundary rewrites; constrained copy-only splitting now uses non-thinking output to avoid exhausting the response budget before emitting a final answer.
- Kept content-safe LLM split results when only semantic-boundary or length feedback remains after retries, then applies deterministic normalization instead of replacing the entire batch with rule-based splitting.
- Allowed punctuation-only changes after the Latin source word sequence has been locked exactly, reducing unnecessary retries without permitting dropped, inserted, or reordered words.

### Fixed

- Fixed DeepSeek V4 Flash returning reasoning-only split responses after consuming the complete output budget.
- Fixed transient HTTP 429, 5xx, timeout, and connection failures immediately degrading subtitle batches instead of retrying with bounded exponential backoff.
- Fixed Chinese translations placing locations, subjects, time adverbials, or short semantic completions on the wrong side of a subtitle boundary.
- Fixed headless macOS test runs aborting near completion because subtitle tests created a windowed Qt application for thread-only behavior.

### Packaging

- Re-signs the exact staged app copy before DMG creation so Finder metadata cannot invalidate the distributed bundle.
- Publishes a locally built macOS Apple Silicon DMG and a standard Windows x64 installer built by GitHub Actions. Models remain separate on-demand downloads.

## v1.1.5 - 2026-08-02

### Added

- Added live MLX Whisper subtitle previews by streaming completed decode segments through the existing task WebSocket without changing the final ASR or forced-alignment result.
- Added provider-aware DeepSeek reasoning controls so focused semantic audits can use thinking while structured splitting, context extraction, translation, and constrained repair keep bounded output budgets.

### Improved

- Redesigned the transcription workspace around a full-height result list and compact timeline-quality summary after removing the unused video player.
- The transcription result list now retains every subtitle, reports the total count, follows new results while live, and stops auto-scrolling when the user reviews earlier entries.
- Parallelized independent Chinese boundary-fluency repair windows for long videos while preserving deterministic document order and validation.
- Reduced oversized long-video context and terminology payloads and added no-thinking fallback when a focused DeepSeek audit exhausts its structured response budget.
- Automatic speaker mode now degrades cleanly to normal transcription if diarization is unavailable, while explicit speaker modes still report configuration failures.

### Fixed

- Fixed the transcription workspace permanently showing only the last 6–8 subtitles.
- Fixed Apple Silicon MLX transcription publishing no subtitle preview until the entire ASR pass completed.
- Fixed macOS cloud-placeholder alignment and Community-1 model files being mistaken for usable offline models.
- Fixed desktop shutdown blocking the native window thread and appearing frozen while background model workers were still exiting.
- Fixed LLM duration logs stopping when response headers arrived instead of after the generated body had been consumed.
- Fixed malformed or empty structured LLM output aborting constrained optimization and targeted recovery loops before validation feedback could run.

### Packaging

- Publishes a locally built macOS Apple Silicon DMG and a standard Windows x64 installer built by GitHub Actions. Models remain separate on-demand downloads.

## v1.1.4 - 2026-07-30

### Added

- Added NVIDIA NIM as an isolated LLM provider with company-grouped model browsing, search, independent credentials, and persistent rate-limit recovery.
- Added native desktop file selection for media and subtitle imports so multi-gigabyte local files no longer need to be copied through the browser upload layer.
- Added a shared subtitle-length policy that exposes an English soft target and a semantic hard limit consistently across prompts, validation, boundary repair, desktop settings, and CLI processing.

### Improved

- Packaged Apple Silicon transcription now runs MLX Whisper in a monitored child process, keeping the desktop backend responsive and making cancellation deterministic.
- Strengthened sentence-boundary repair for dangling subjects, conjunctions, relative clauses, time adverbials, modifiers, and short speaker-assignment flips while preserving every source word and timestamp order.
- Strengthened translation ownership and document-level audits for shifted meanings, adjacent duplicates, untranslated residue, reasoning leakage, unsupported corrections, quantities, units, and model identifiers.
- The subtitle settings page now offers compact, balanced, relaxed, and custom English length modes and displays both the effective target and hard maximum.
- Video decoding pauses while transcription is active to reduce resource contention with ASR.

### Fixed

- Fixed browser and packaged desktop subtitle jobs ignoring the configured CJK and English length values.
- Fixed the LLM split validator and final boundary normalizer calculating different English hard limits.
- Fixed large desktop imports being duplicated into temporary upload storage and potentially freezing the interface.
- Fixed packaged MLX inference competing with the web backend in the same process and making long transcription jobs appear permanently frozen.
- Fixed macOS ad-hoc signing occasionally racing Finder metadata updates while assembling the app under Desktop.

### Packaging

- Publishes a locally built macOS Apple Silicon DMG and a standard Windows x64 installer built by GitHub Actions. No CUDA-specific installer is produced.

## v1.1.3 - 2026-07-28

### Added

- Added conservative mixed-language detection for WhisperX automatic source-language mode on Apple Silicon, including local re-transcription of high-confidence language switches and language-specific forced alignment.
- Added a recoverable missing-alignment-model prompt that shows the detected language, affected ranges, model details, and download progress without discarding the completed ASR pass.
- Added explicit choices to download and resume word alignment, continue with native sentence timing, or ignore confirmed foreign-language ranges.

### Improved

- Automatic source-language transcription now retains the original audio instead of applying DeepFilterNet, protecting short foreign-language passages from over-suppression.
- Mixed-language translation prompts now require each subtitle key to be translated from its actual spoken language rather than normalized to the surrounding primary language.
- Windows WhisperX automatic-language jobs now use the same missing-alignment-model decision flow for the detected primary language.

### Fixed

- Fixed automatic source-language jobs being constrained by a manually configured single-language alignment model.
- Fixed skipped alignment languages disappearing from word-timestamp output; they now retain the native sentence-level timestamps produced by ASR.
- Fixed the global LLM client bypassing the provider-aware client factory, which could prevent the MiniMax Anthropic-compatible endpoint from being selected correctly.
- Fixed the desktop fallback version label remaining on an older release number.

### Packaging

- This release publishes the macOS Apple Silicon DMG only. No Windows or CUDA installer is produced.

## v1.1.2 - 2026-07-27

### Improved

- Expanded long-video translation context from head/tail excerpts to representative windows across the full transcript, improving terminology and subject continuity in middle batches.
- Added conservative MiniMax M3 semantic checks for unambiguous ASR formatting and homophone errors, including spoken units, abbreviated model years, automotive controls, and model-facelift wording.
- Added whole-document translation finalization so recovery batches and normal batches receive the same semantic and cross-batch repetition checks.
- Strengthened numeric validation for `grand`/`K` magnitudes, grouped ASR numbers, model identifiers, and equivalent Chinese quantity notation.

### Fixed

- Fixed correct `mpg`/`mph` repairs being rejected when MiniMax returned equivalent Chinese unit wording.
- Fixed isolated fallback translations bypassing final semantic alignment checks and retaining a small number of wrong or neighboring meanings.
- Fixed repeated translations that crossed a batch boundary and survived otherwise valid per-batch responses.
- Fixed Chinese commas and periods only being removed at line endings instead of being replaced throughout translated subtitle lines while preserving decimals, identifiers, and English source text.

### Packaging

- Windows releases continue to provide the standard x64 installer only. No CUDA installer is produced for this release.

## v1.1.1 - 2026-07-26

### Improved

- Strengthened reflective MiniMax M3 translation with conservative, source-grounded alignment audits that repair only repeatedly confirmed subtitle mismatches.
- Improved dialogue translation fidelity for pronouns, titles, names, quantities, units, and probable ASR corrections while preserving natural Chinese phrasing across clause boundaries.
- Added focused revalidation after local repairs so corrected subtitles cannot silently retain neighboring content or unsupported details.
- Improved translation failure messages to distinguish provider and network failures from local subtitle quality-gate rejection.

### Fixed

- Fixed isolated untranslated English words being accepted as valid Chinese translations merely because they used title case.
- Fixed occasional translation expansion, compression, sentence ownership drift, and invented unit conversions in reflective batches.
- Fixed alignment repair accepting its own first-pass judgment without independent confirmation.

### Packaging

- Windows releases now provide the standard x64 installer only. The CUDA installer is not produced for this release.

## v1.1.0 - 2026-07-25

### Added

- Added dialogue-aware translation that uses anonymous speaker turns to improve replies, pronouns, ellipsis, intent, and tone without exposing speaker labels in final subtitles.
- Added a Windows CUDA 12.8 build for supported NVIDIA GPUs while retaining the standard Windows installer for broader compatibility.

### Improved

- Improved multi-speaker assignment and sentence grouping around hesitation, stutter, short interjections, and genuine speaker changes.
- Improved translation recovery for isolated mismatches and repeated phrases without rewriting valid neighboring subtitles.
- Improved Windows WhisperX device selection, model reuse, cancellation, and worker-process recovery.

### Fixed

- Fixed speaker labels, duplicated phrases, and occasional dialogue meaning drift in translated subtitles.
- Fixed MiniMax M3 partial-correction failures and false rejection of legitimate uppercase words and abbreviations.
- Fixed missing Silero VAD assets, transient FFmpeg download failures, and optional-dependency conflicts in Windows packaging and CI.
- Fixed oversized CUDA release packaging by publishing one installer EXE with three required BIN volumes.

### Installation note

- The Windows CUDA edition requires its EXE and all three matching BIN files in the same folder. See `.github/release-notes/v1.1.0.md` for the complete download instructions.

## v1.0.10 - 2026-07-24

### Added

- Added a compact forced-alignment model manager covering 41 languages, with search, download status, automatic source-language matching, and manual override support.

### Changed

- Multi-speaker transcription now always retains the original audio and skips DeepFilterNet candidate calibration, candidate ASR passes, and full-length denoising. Single-speaker audio enhancement remains available.
- WhisperX alignment settings now use automatic source-language matching by default. The transcription workspace shows only the model required by the current language.
- Legacy custom alignment-model settings are preserved as manual overrides instead of being replaced during configuration migration.
- DeepFilterNet uses bounded CPU threading on Apple Silicon to reduce avoidable contention during single-speaker enhancement.

### Removed

- Removed the obsolete adaptive multi-speaker enhancement pipeline and its redundant candidate transcription passes.

## v1.0.9 - 2026-07-22

### Added

- Added incremental subtitle preview events so transcription, splitting, optimization, and translation results appear while a task is still running without repeatedly transferring the complete timeline.
- Added task-scoped resource locks and terminal-state guards to prevent duplicate jobs, cancellation races, and late worker results from overwriting completed or cancelled tasks.
- Added shared audio-analysis contexts that reuse decoded waveforms, energy windows, and TEN-VAD inference across timing refinement and hallucination checks.
- Added thread-safe in-process caches for WhisperX forced-alignment models and Community-1 speaker-diarization pipelines.
- Added configurable LLM log detail levels. The recommended summary mode records task, latency, token, cache, and status metrics without persisting full prompts or responses.
- Added atomic file writes for settings, subtitle exports, native save-dialog output, and log clearing.

### Fixed

- Fixed live subtitle previews being delayed until the complete transcription or translation task finished.
- Fixed large preview snapshots and frequent partial-SRT writes causing avoidable serialization, WebSocket, and disk-I/O overhead.
- Fixed concurrent LLM requests being paired with the wrong response or leaving superseded retry entries in memory.
- Fixed API keys and Hugging Face tokens being returned to the frontend after they had been saved; settings now expose only configured-state metadata.
- Fixed switching LLM providers with a blank password field erasing the provider's previously stored API key.
- Fixed overlapping jobs operating on the same media or subtitle file and producing ambiguous completion state.
- Fixed packaged desktop builds silently reusing stale frontend output when a fresh frontend build failed.

### Changed

- LLM logs are grouped by task and load batch details only when selected, substantially reducing the default log view and response payload.
- Multi-speaker denoise calibration now keeps reusable ASR/alignment resources warm while comparing candidates and retains the original audio when enhancement risks weaker-speaker coverage.
- Removed unused legacy LLM diarization and duplicate ASR content-integrity modules from the runtime and desktop bundle.
- Expanded CI type checking and regression coverage across backend, ASR, translation, optimization, subtitle, thread, TTS, launcher, and packaging code.

## v1.0.8 - 2026-07-22

### Added

- Added adaptive multi-speaker audio enhancement that compares conservative DeepFilterNet settings and retains the candidate that improves recognition without suppressing quieter speakers.
- Added MiniMax M3 support through its native Anthropic-compatible endpoint, including provider-aware retry behavior and prompt-cache usage reporting.
- Added stricter translation ownership checks for neighboring subtitle keys, preserved model/spec tokens, duplicated target phrases, untranslated Latin residue, and incomplete English split boundaries.
- Added Korean/Chinese bilingual SRT parsing coverage, including decomposed Hangul filenames and either source/translation display order.
- Added packaged HTTP runtime smoke testing so desktop CI launches the real FastAPI server and checks its health endpoint before publishing installers.

### Fixed

- Fixed Windows desktop startup failing with `Unable to configure formatter 'default'` when Uvicorn initialized console logging inside a windowed PyInstaller executable.
- Fixed frozen desktop builds exposing missing standard streams to libraries and debug output, which could also break native file export calls.
- Fixed Windows upload, cache, and fallback log paths using Unix-only `/tmp` or `/private/tmp` locations.
- Fixed ffmpeg, DeepFilterNet, and Whisper.cpp subprocesses opening unwanted console windows on Windows.
- Fixed Windows long UNC media paths receiving an invalid extended-path prefix.
- Fixed English-only forced-alignment models being applied to Korean and other non-English transcription jobs.
- Fixed translated subtitles accepting merge notes, neighboring-key content, repeated translations, untranslated fragments, or misplaced model/spec values.
- Fixed subtitle optimization discarding the original batch when all LLM correction attempts failed validation.

### Changed

- Improved MiniMax M3 translation reliability with native Anthropic requests, stable system prompts, rate-limit waiting, and clearer token/cache metrics.
- Improved sentence splitting so English captions cannot end on obvious conjunctions, determiners, prepositions, or incomplete phrases.
- Improved desktop startup diagnostics by writing full backend startup traces to the platform temporary directory and allowing more time for antivirus-scanned Windows bundles to initialize.
- Updated Windows packaging to collect the required Colorama runtime explicitly and strengthened installed-app smoke tests.

## v1.0.7 - 2026-07-12

### Added

- Added provider-specific LLM profiles so each service keeps its own Base URL, API key, and model when users switch providers.
- Added recoverable subtitle output for failed translation jobs, preserving completed translations and applying Chinese punctuation cleanup before saving the recovery file.
- Added stricter final translation validation for empty, untranslated, and merge-placeholder responses, with focused retries for incomplete batches.
- Added a unified ASR scheme summary and self-test that clearly identifies the active engine, transcription model, forced-alignment model, hardware, and acceleration path.

### Fixed

- Fixed translation jobs failing near completion without saving the already completed bilingual subtitles.
- Fixed LLM responses such as `（合并至上一条）`, `内容同上`, and untranslated English lines being accepted as completed Chinese translations.
- Fixed switching LLM providers reusing the previous provider's API key.
- Fixed the desktop application hanging during shutdown while background work was still completing.
- Fixed clipped video controls in the transcription workspace and aligned the two lower quality panels with the video panel.
- Fixed inconsistent ASR engine card heights, a missing FasterWhisper icon, irregular model controls, and cramped reflection-mode spacing.

### Changed

- Redesigned Settings into dedicated LLM, speech recognition, subtitle processing, and file storage workspaces while retaining the existing backend configuration contract.
- Simplified speech-recognition settings into current scheme, engine, transcription model, word-level alignment, and advanced configuration layers; removed the duplicate default-model selector.
- Improved transcription and subtitle workspaces with stable responsive dimensions, clearer model states, and more consistent card alignment.
- Pinned GitHub Actions to `uv 0.10.11` so CI and desktop builds no longer depend on fetching the mutable `latest` version manifest.

## v1.0.6 - 2026-07-11

### Added

- Added final bilingual subtitle integrity validation. Every translated cue must retain its source text, timing, speaker, and a non-empty, non-placeholder translation before export.
- Added task-scoped WebSocket subscriptions and richer live subtitle previews for transcription and subtitle-processing jobs.
- Added cancellation callbacks for blocking desktop tasks so closing or cancelling the application can terminate active subprocesses cleanly.
- Added regression coverage for bilingual clause alignment, translation completeness, concurrent task updates, upload isolation, configuration validation, and packaged model handling.

### Fixed

- Fixed translated subtitles being split again by unrelated source/target character positions, which could create blank Chinese lines or attach a translation to the wrong English clause.
- Fixed failed translation chunks being silently replaced with source text and then exported as apparently completed subtitles.
- Fixed merge-placeholder responses such as `（合并至上一条）` surviving translation validation.
- Fixed desktop tasks failing to display intermediate transcription and translation results consistently.
- Fixed task progress moving backwards and late worker completion overwriting a cancelled task.
- Fixed application shutdown waiting indefinitely for blocking transcription or translation work.
- Fixed concurrent uploads with the same filename overwriting one another and added cleanup for abandoned session files.
- Fixed corrupted or invalid persisted settings silently falling back to unintended models or processing parameters.
- Fixed concurrent model downloads racing to replace the same destination.
- Fixed packaged WhisperX/MLX runtime initialization and model-path handling across source and desktop builds.

### Changed

- Bilingual subtitle cues are now structurally locked after source-language sentence splitting; long lines remain in the same timed cue and are left to the subtitle renderer to wrap naturally.
- Expanded CI to lint backend, launcher, and build scripts, run the broader offline Python test suite, and build/lint the Next.js frontend.
- Updated the architecture documentation to reflect the current Next.js, FastAPI, pywebview, MLX Whisper, WhisperX, TEN-VAD, and DeepFilterNet3 workflow.

## v1.0.5 - 2026-07-11

### Added

- Added Chinese subtitle punctuation beautification. After Chinese translation and bilingual resegmentation, Chinese commas and periods in translated lines are replaced with spaces for a cleaner subtitle appearance.
- Added a subtitle-processing setting to enable or disable Chinese punctuation beautification. It is enabled by default and runs locally without using an LLM or consuming tokens.
- Added a Windows x64 EXE installer built and smoke-tested by GitHub Actions.

### Fixed

- Fixed Windows desktop builds defaulting to the Apple Silicon-only MLX/WhisperX path. Windows now defaults to bundled Whisper.cpp and prevents selecting the unsupported WhisperX engine.
- Bundled the official whisper.cpp Windows CLI and required DLLs, and fixed Windows CLI invocation to use the current official command-line interface.

### Changed

- Release automation now publishes only the Windows EXE installer from GitHub Actions; ZIP bundles remain available only as short-lived CI artifacts.

## v1.0.3 - 2026-06-14

### Added

- Added an ASR model status panel that clearly shows the active engine, selected model, local model path, runtime readiness, and forced-alignment model state.
- Added a model self-test action so users can verify the current transcription configuration before starting a full job.
- Added backend model discovery and health-check APIs, including automatic detection of local MLX Whisper models.

### Fixed

- Fixed packaged macOS builds failing when MLX modules initialized Metal during PyInstaller analysis.
- Fixed MLX Whisper runtime resources not being available inside the installed macOS application.
- Fixed ambiguous WhisperX model selection and misleading manual-configuration labels for locally installed MLX models.
- Fixed WhisperX model resolution so the selected MLX model is consistently used by transcription jobs and model tests.

### Changed

- Improved WhisperX settings with explicit selected, locally ready, and on-demand download states.
- Simplified the ASR implementation by removing unused Bcut and Jianying transcription backends and their obsolete tests and documentation.
- Removed unused repository skills, starter assets, launcher scripts, and inactive Claude workflows.
- Reduced desktop packaging overhead by injecting the MLX runtime after PyInstaller analysis and verifying all required MLX resources in the final bundle.

## v1.0.2 - 2026-06-02

### Fixed

- Fixed Whisper.cpp word-level transcription dropping or misplacing quiet intro speech.
- Fixed smart split output covering silent/no-speech regions by running a second source-audio timing refinement after sentence reconstruction.
- Fixed subtitle timing refinement so VAD can trim both leading silence and trailing silence instead of only subtitle tails.
- Fixed RMS pause restoration cutting through Silero VAD-confirmed continuous speech in noisy driving footage.
- Fixed repeated Whisper.cpp text fragments around chunk/VAD boundaries.
- Fixed English smart-split spacing around punctuation, including cases like `everyone, welcome` and `Torrance, California`.
- Fixed dangling English fragments such as sentence parts split after `to`, `this`, or similar connector words.
- Fixed packaged macOS builds so DeepFilterNet3 denoising is available in the app bundle.

### Changed

- Disabled Whisper.cpp internal VAD for full-audio word timestamp runs to avoid missing quiet opening speech.
- Added live transcription and smart-split UI updates while processing.
- Improved macOS packaging for bundled torch, torchaudio, static ffmpeg/ffprobe, and denoise resources.
- Added regression tests for word-level timestamp preservation, repeated ASR text cleanup, VAD timing edge trimming, smart-split punctuation spacing, dangling-tail split avoidance, and post-split timing refinement.

## v1.0.1 - 2026-06-01

### Fixed

- Fixed Whisper.cpp model selection so local Whisper.cpp transcription no longer falls through to the Jianying path.
- Fixed bundled desktop app startup and whisper.cpp binary discovery for packaged macOS builds.
- Fixed ASR task cache reuse so new transcription requests do not instantly return stale subtitle files.
- Fixed subtitle translation/optimization cache reuse so the translation page does not display old processed subtitles.
- Fixed export actions on the transcription and translation pages.
- Fixed overlapping subtitle timestamps by normalizing boundaries before export.
- Restored silence-gap handling for Whisper.cpp output while preventing adjacent subtitles from collapsing into a continuous timeline.
- Improved ASR post-processing for long subtitle spans that cover internal silence, road noise, or music.
- Added conservative sentence-final tail trimming for noisy driving footage where RMS energy stays high after speech ends.
- Fixed bilingual subtitle re-segmentation so optimized and translated subtitles keep aligned source/target lines.

### Changed

- Improved local desktop packaging with bundled ffmpeg/ffprobe resources.
- Added regression tests for ASR cache behavior, Whisper.cpp routing, subtitle timestamp normalization, translation cache invalidation, optimizer cache invalidation, backend task status, and bilingual subtitle re-segmentation.
