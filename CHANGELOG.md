# Changelog

## v1.1.0 - 2026-07-25

### Added

- Added dialogue-aware subtitle translation. Anonymous speaker metadata now helps the LLM resolve replies, pronouns, ellipsis, intent, tone, and register without leaking speaker labels into final subtitles.
- Added conservative same-speaker boundary validation for duplicated connectors and repeated Chinese conclusions, with isolated recovery that preserves valid neighboring translations.
- Added a dedicated Windows CUDA 12.8 installer using PyTorch and Torchaudio 2.8 CUDA wheels. The standard Windows installer remains available for broader hardware compatibility.

### Changed

- Improved multi-speaker assignment smoothing around incomplete question, subject, and continuation islands while preserving complete short interjections and genuine turn changes.
- Improved dialogue-aware sentence grouping across hesitation pauses and stuttered continuations without merging complete replies across speaker boundaries.
- Translation context and cache keys now include anonymous speaker turns, while single-speaker payloads retain the existing compact format.
- Windows WhisperX now selects CTranslate2 transcription and PyTorch forced-alignment devices independently, reuses managed FasterWhisper models, and isolates packaged CUDA transcription workers for safer cancellation and failure recovery.

### Fixed

- Fixed sparse MiniMax M3 alignment corrections repeatedly failing without a conservative per-item recovery path.
- Fixed speaker identifiers appearing in translated subtitle text and dialogue meaning drifting across speaker boundaries.
- Fixed legitimate uppercase pronouns being mistaken for protected model identifiers and standard translations such as REM being rejected.
- Fixed Windows desktop packages missing FasterWhisper's Silero VAD asset and failing when the FFmpeg download endpoint returned a transient gateway timeout.
- Fixed CUDA installer packaging exceeding the GitHub Actions time limit by using a large-runtime compression profile.
- Fixed base CI importing the optional Hugging Face runtime before alignment-download tests could install their isolated test substitute.

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
