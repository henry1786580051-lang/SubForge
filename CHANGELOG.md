# Changelog

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
