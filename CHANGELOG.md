# Changelog

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
