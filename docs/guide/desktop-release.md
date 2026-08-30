# Desktop release build

SubForge distributes an Apple Silicon DMG and a Windows x64 setup executable.
The installers include the application, frontend, Python runtime, and required
media tools. Large ASR, alignment, and speaker-model weights are downloaded
separately; user settings and credentials are never build inputs.

## Versions

Create a `vX.Y.Z` Git tag for each new release. The desktop builder resolves the
version from `SUBFORGE_BUILD_VERSION`, then the latest reachable version tag,
then package metadata. It injects the same version into the frontend and packaged
Python runtime; macOS bundle metadata and artifact names use that value as well.
Keep the Sidebar fallback version and release notes current for source checkouts.

## Local macOS build

Use Python 3.12 and the repository lockfile. On Apple Silicon, install the audio
and WhisperX extras so the packaged app includes MLX, alignment, and diarization:

```bash
uv sync --frozen --extra denoise --extra whisperx
SUBFORGE_BUILD_VERSION=1.2.0 uv run --no-sync python scripts/build_desktop.py --clean --no-archive
uv run --no-sync python scripts/smoke_desktop.py dist/SubForge.app
```

The clean build removes managed build outputs, but retains evaluation evidence
under `artifacts/`. The verified app is `dist/SubForge.app`; a redundant standalone
`dist/SubForge` directory is removed after successful verification. Do not rename
or duplicate the output as `SubForge 2`.

To build an app and the drag-install DMG together:

```bash
SUBFORGE_BUILD_VERSION=1.2.0 uv run --no-sync python build_macos.py
```

That command writes `~/Desktop/SubForge.dmg`. Name the published asset
`SubForge-X.Y.Z-macos-arm64.dmg`. The DMG builder preserves bundle symlinks,
verifies the staged app signature, and verifies the sealed app inside the image.
`SUBFORGE_CODESIGN_IDENTITY` selects an available signing identity; otherwise
local signing is ad hoc, which is not Apple notarization.

The build downloads checksum-pinned FFmpeg 8.1.2 executables into the packaged
runtime. These support media inspection and audio extraction; subtitle burn-in
is not part of the current application or its release checks.

## GitHub Actions

`.github/workflows/build-desktop.yml` runs on pull requests affecting the desktop
app, pushes to `master`/`main`, version tags, and manual dispatch. The matrix is:

| Build | Runner | Output |
| --- | --- | --- |
| Windows x64 | `windows-latest` | ZIP and Inno Setup EXE |
| macOS Intel | `macos-15-intel` | App ZIP |
| macOS Apple Silicon | `macos-15` | App ZIP |

The current matrix does not build a separate CUDA installer. macOS CI ZIPs are
build artifacts; the public Apple Silicon DMG is built and checked locally.
The Intel build does not include the Apple Silicon-only MLX engine.

Every packaged smoke test checks the backend HTTP runtime and bundled FFmpeg /
ffprobe with a generated media clip. Windows additionally checks whisper.cpp,
DeepFilterNet3 inference, FasterWhisper/CTranslate2/PyAV, WhisperX alignment, and
pyannote imports, then repeats the checks after a silent EXE installation.
Apple Silicon verifies MLX Metal and PyTorch MPS operations through a macOS Aqua
session, plus alignment and diarization imports. These are runtime checks, not a
complete transcription or paid-API translation quality benchmark.

CI in `.github/workflows/ci.yml` separately runs Python tests, Ruff, Pyright, a
Python package build, and frontend tests, lint, and a production build.

## Publication Checklist

1. Review the pending diff for secrets, local data, and unvalidated experiments.
2. Update `CHANGELOG.md` and `.github/release-notes/vX.Y.Z.md`, preserving historical
   entries and separating shipped behavior from future work.
3. Run local regression and static checks, commit, and push the branch and tag.
4. The Windows tag job uploads its EXE to the matching release. If no release
   exists, it creates a **draft** using the checked-in notes, not a public empty
   release. macOS jobs do not race to create release entries.
5. Build the local DMG, mount it, verify the stored app and runtime, and upload
   the versioned DMG to the same draft. Confirm versions, filenames, sizes,
   checksums, Actions results, and that the EXE is the standard x64 build.
6. Publish the draft only after both installers and release checks are complete.

Manual dispatch accepts `release_version` for rebuilding an existing version.
It does not create a new tag or automatically replace release assets; explicitly
select the correct source revision and review the replacement files before upload.
