# Desktop release build

SubForge publishes desktop bundles for Windows and macOS from GitHub Actions.
Users can download the zip files from a GitHub Release, extract them, and run the
bundled `SubForge` executable without installing Python or FFmpeg.

## Local build

```bash
uv sync --frozen
uv run --with static-ffmpeg --with pyinstaller python scripts/build_desktop.py --clean
uv run python scripts/smoke_desktop.py dist/SubForge
```

The build script downloads static `ffmpeg` and `ffprobe` for the current platform
and bundles them under `resource/bin` inside the PyInstaller app. Runtime user data
is kept in the system user-data directory, so app upgrades do not overwrite
settings, logs, cache, models, or custom subtitle styles.

## CI and releases

`.github/workflows/build-desktop.yml` builds on:

- `windows-latest`
- `macos-13`

Each job runs a real packaged-app smoke test:

- starts the packaged executable with `--version`
- lists bundled subtitle styles
- runs `doctor --json`
- generates a short video with bundled FFmpeg
- creates both soft-subtitle and hard-subtitle videos
- validates output duration with bundled ffprobe

On `v*` tags, desktop zip files are uploaded to the GitHub Release. The PyPI
workflow still publishes the Python package and uploads the wheel/sdist to the
same release.
