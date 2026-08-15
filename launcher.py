#!/usr/bin/env python3
"""Subtitle Desktop App.

Starts FastAPI backend in background and opens a native window.
"""
import base64
import multiprocessing
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

_NULL_STREAMS = []


def _configure_frozen_standard_streams() -> None:
    """Provide writable streams for libraries inside a windowed executable."""
    if not getattr(sys, "frozen", False):
        return
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is not None:
            continue
        stream = open(os.devnull, "w", encoding="utf-8")
        _NULL_STREAMS.append(stream)
        setattr(sys, name, stream)


def _configure_frozen_runtime_paths() -> None:
    """Expose packages injected after PyInstaller analysis to frozen Python."""
    if not getattr(sys, "frozen", False):
        backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
        if os.path.isdir(backend_dir) and backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        return

    bundle_root = os.fspath(getattr(sys, "_MEIPASS"))
    runtime_paths = [bundle_root, os.path.join(bundle_root, "backend")]
    executable_dir = os.path.dirname(os.path.abspath(sys.executable))
    contents_dir = os.path.dirname(executable_dir)
    if os.path.basename(executable_dir) == "MacOS":
        runtime_paths.extend(
            [
                os.path.join(contents_dir, "Resources"),
                os.path.join(contents_dir, "Frameworks"),
            ]
        )

    for path in reversed(runtime_paths):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def _configure_application_logging() -> None:
    """Persist backend errors before the embedded server starts."""
    try:
        from subforge.core.utils.logger import configure_root_logger

        configure_root_logger()
    except Exception:
        # Startup error reporting below remains available if the log directory
        # itself cannot be initialized.
        pass


_configure_frozen_standard_streams()
_configure_frozen_runtime_paths()

HOST = "127.0.0.1"
PREFERRED_PORT = 8000


def find_available_port(preferred_port: int = PREFERRED_PORT) -> int:
    """Prefer the stable dev port, then fall back to an OS-assigned port."""
    for port in (preferred_port, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((HOST, port))
            except OSError:
                continue
            return int(sock.getsockname()[1])
    raise RuntimeError("No available localhost port found")


def start_server(port: int, server_errors: list[str], server_holder: list | None = None):
    import uvicorn

    try:
        # A frozen GUI executable has no console streams on Windows. Uvicorn's
        # default colour formatter probes those streams while dictConfig is
        # initialized and can abort startup before FastAPI is loaded.
        config = uvicorn.Config(
            "app.main:app",
            host=HOST,
            port=port,
            log_level="warning",
            log_config=None,
            access_log=False,
        )
        server = uvicorn.Server(config)
        if server_holder is not None:
            server_holder.append(server)
        server.run()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        server_errors.append(detail)
        try:
            log_dir = Path(tempfile.gettempdir()) / "SubForge"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "startup-error.log").write_text(
                "".join(traceback.format_exception(exc)),
                encoding="utf-8",
            )
        except OSError:
            pass


def _cleanup_desktop_session(server=None, *, cleanup_uploads: bool = True) -> None:
    """Request backend shutdown and remove files owned by this process."""
    if server is not None:
        server.should_exit = True
    if not cleanup_uploads:
        return
    try:
        from app.api.files import cleanup_session_uploads
        from app.security import clear_granted_paths

        cleanup_session_uploads()
        clear_granted_paths()
    except Exception:
        # Closing the native window must remain reliable even during a partial
        # backend startup or interpreter teardown.
        pass


def _begin_desktop_shutdown(
    server,
    server_thread: threading.Thread,
    exit_started: threading.Event,
    *,
    hard_exit=None,
) -> bool:
    """Start bounded backend shutdown without blocking the native UI thread."""
    if exit_started.is_set():
        return False
    exit_started.set()

    def finish_shutdown():
        _cleanup_desktop_session(server, cleanup_uploads=False)
        server_thread.join(timeout=1.5)
        if not server_thread.is_alive():
            _cleanup_desktop_session()
        (hard_exit or os._exit)(0)

    threading.Thread(
        target=finish_shutdown,
        name="subforge-desktop-shutdown",
        daemon=True,
    ).start()
    return True


def wait_for_server(url: str, server_errors: list[str], timeout_seconds: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    health_url = f"{url}/api/health"
    while time.monotonic() < deadline:
        if server_errors:
            return False
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    return False


def check_backend_runtime(timeout_seconds: float = 30.0) -> None:
    """Start the embedded server and verify the packaged HTTP runtime."""
    port = find_available_port()
    url = f"http://{HOST}:{port}"
    server_errors: list[str] = []
    server_holder: list = []
    server_thread = threading.Thread(
        target=start_server,
        args=(port, server_errors, server_holder),
        daemon=True,
    )
    server_thread.start()
    if not wait_for_server(url, server_errors, timeout_seconds):
        detail = server_errors[-1] if server_errors else f"Timed out waiting for backend at {url}"
        raise RuntimeError(f"SubForge backend failed to start: {detail}")
    _cleanup_desktop_session(server_holder[0] if server_holder else None)
    server_thread.join(timeout=5.0)


class Api:
    """API exposed to the pywebview JavaScript context."""

    def open_file(self, kind: str = "media"):
        """Select a local input without copying multi-gigabyte media through HTTP."""
        import traceback

        import webview

        try:
            if not webview.windows:
                return {"ok": False, "error": "No window available"}
            file_types = (
                ("Subtitle files (*.srt;*.vtt;*.ass)",)
                if kind == "subtitle"
                else (
                    "Media files (*.mp4;*.mov;*.mkv;*.avi;*.webm;*.mp3;*.wav;*.m4a;*.flac)",
                    "All files (*.*)",
                )
            )
            result = webview.windows[0].create_file_dialog(
                webview.OPEN_DIALOG,
                allow_multiple=False,
                file_types=file_types,
            )
            if not result:
                return {"ok": False, "cancelled": True}
            file_path = result if isinstance(result, str) else result[0]
            from app.security import grant_path

            granted_path = grant_path(file_path)
            return {"ok": True, "path": str(granted_path)}
        except Exception as exc:
            traceback.print_exc()
            return {"ok": False, "error": str(exc)}

    def save_file(self, base64_data: str, default_filename: str):
        """Save a base64-encoded file using native save dialog."""
        import traceback

        import webview

        try:
            if not webview.windows:
                return {"ok": False, "error": "No window available"}
            print(f"[Api] save_file called: {default_filename}")
            result = webview.windows[0].create_file_dialog(
                webview.SAVE_DIALOG,
                save_filename=default_filename,
            )
            print(f"[Api] dialog result: {result}")
            if result:
                file_path = result if isinstance(result, str) else result[0]
                data = base64.b64decode(base64_data)
                from subforge.core.utils.atomic_write import atomic_write_bytes

                atomic_write_bytes(file_path, data)
                print(f"[Api] saved to: {file_path}")
                return {"ok": True, "path": file_path}
            print("[Api] dialog cancelled")
            return {"ok": False}
        except Exception as e:
            print(f"[Api] save_file error: {e}")
            traceback.print_exc()
            return {"ok": False, "error": str(e)}

    def open_logs_folder(self):
        """Open the persistent diagnostics directory in the system file manager."""
        try:
            from subforge.config import LOG_PATH

            LOG_PATH.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(str(LOG_PATH))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                import subprocess

                subprocess.Popen(["open", str(LOG_PATH)])
            else:
                import subprocess

                subprocess.Popen(["xdg-open", str(LOG_PATH)])
            return {"ok": True, "path": str(LOG_PATH)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def main():
    # Start FastAPI in background thread
    port = find_available_port()
    url = f"http://{HOST}:{port}"
    server_errors: list[str] = []
    server_holder: list = []
    server_thread = threading.Thread(
        target=start_server,
        args=(port, server_errors, server_holder),
        daemon=True,
    )
    server_thread.start()

    startup_timeout = float(
        os.environ.get(
            "SUBFORGE_STARTUP_TIMEOUT",
            "120" if getattr(sys, "frozen", False) else "30",
        )
    )
    if not wait_for_server(url, server_errors, startup_timeout):
        detail = server_errors[-1] if server_errors else f"Timed out waiting for backend at {url}"
        raise RuntimeError(f"SubForge backend failed to start: {detail}")

    # Open native window with JS API
    import webview
    api = Api()

    def on_loaded():
        """Signal that pywebview API is ready."""
        webview.windows[0].evaluate_js(
            "window.__pywebview_ready = true; "
            "window.dispatchEvent(new Event('pywebviewready'));"
        )

    window = webview.create_window(
        "SubForge",
        url=url,
        width=1400,
        height=900,
        min_size=(900, 600),
        text_select=True,
        js_api=api,
    )

    exit_started = threading.Event()

    def force_exit():
        # PyInstaller desktop builds can keep non-UI worker threads alive after
        # the native window closes. Clean this session first, then retain the
        # hard-exit fallback that prevents shutdown hangs.
        # Never block the native UI event thread on a model worker. Give the
        # backend a short grace period, then terminate the frozen process;
        # stale session files are removed safely on the next startup.
        _begin_desktop_shutdown(
            server_holder[0] if server_holder else None,
            server_thread,
            exit_started,
        )

    try:
        window.events.closed += force_exit
    except Exception:
        pass

    webview.start(debug=False, func=on_loaded)
    force_exit()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _configure_application_logging()

    if os.environ.get("SUBFORGE_DENOISE_WORKER") == "1":
        from subforge.core.asr.audio_enhancer import run_packaged_denoise_worker

        run_packaged_denoise_worker()

    if os.environ.get("SUBFORGE_FASTER_WHISPER_WORKER") == "1":
        from subforge.core.asr.faster_whisper import run_packaged_faster_whisper_worker

        run_packaged_faster_whisper_worker()

    if os.environ.get("SUBFORGE_MLX_WHISPER_WORKER") == "1":
        from subforge.core.asr.whisperx_asr import run_packaged_mlx_whisper_worker

        run_packaged_mlx_whisper_worker()

    if os.environ.get("SUBFORGE_CHECK_DENOISE") == "1":
        import traceback

        from subforge.core.asr.audio_enhancer import enhance_audio, is_available

        available = is_available()
        print(f"DeepFilterNet3 available: {available}")
        if not available:
            try:
                import soundfile  # noqa: F401
                import torch  # noqa: F401
                from df.enhance import enhance as _enhance  # noqa: F401
                from df.enhance import init_df as _init_df  # noqa: F401
            except Exception:
                traceback.print_exc()
        audio_path = os.environ.get("SUBFORGE_DENOISE_AUDIO_PATH", "")
        if available and audio_path:
            enhanced_path = None
            try:
                enhanced_path = enhance_audio(audio_path, atten_lim_db=18.0)
                enhanced_file = Path(enhanced_path)
                if not enhanced_file.is_file() or enhanced_file.stat().st_size == 0:
                    raise RuntimeError("DeepFilterNet3 produced no smoke-test output")
                print(f"DeepFilterNet3 inference: ok ({enhanced_file.stat().st_size} bytes)")
            except BaseException:
                traceback.print_exc()
                raise SystemExit(1)
            finally:
                if enhanced_path:
                    Path(enhanced_path).unlink(missing_ok=True)
        raise SystemExit(0 if available else 1)
    if os.environ.get("SUBFORGE_CHECK_FASTER_WHISPER") == "1":
        import traceback

        try:
            import av  # noqa: F401
            import ctranslate2  # noqa: F401
            import faster_whisper  # noqa: F401
            import numpy as np
            from faster_whisper.vad import get_speech_timestamps

            from subforge.core.asr.faster_whisper import (
                resolve_faster_whisper_runtime,
            )

            device, compute_type = resolve_faster_whisper_runtime("auto", "default")
            vad_segments = get_speech_timestamps(np.zeros(16_000, dtype=np.float32))
            if not isinstance(vad_segments, list):
                raise RuntimeError("Packaged FasterWhisper VAD returned an invalid result")
            print(f"FasterWhisper import: ok ({device}/{compute_type})")
            print("FasterWhisper Silero VAD inference: ok")
            raise SystemExit(0)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
    if os.environ.get("SUBFORGE_CHECK_WHISPERX") == "1":
        import traceback

        try:
            from subforge.core.asr.whisperx_asr import install_whisperx_runtime_stubs

            install_whisperx_runtime_stubs()
            from whisperx.alignment import align as _align  # noqa: F401
            from whisperx.alignment import load_align_model as _load_align_model  # noqa: F401
            from whisperx.asr import load_model as _load_model  # noqa: F401
            from whisperx.audio import load_audio as _load_audio  # noqa: F401

            print("WhisperX ASR import: ok")
            print("WhisperX forced alignment import: ok")
            raise SystemExit(0)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
    if os.environ.get("SUBFORGE_CHECK_ASR") == "1":
        import traceback

        try:
            import mlx.core as mx
            import mlx_whisper  # noqa: F401
            import torch

            mlx_probe = mx.array([1.0], dtype=mx.float32) + 1.0
            mx.eval(mlx_probe)
            if float(mlx_probe.item()) != 2.0:
                raise RuntimeError("MLX Metal probe returned an invalid result")

            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            if not (
                mps_backend
                and mps_backend.is_built()
                and mps_backend.is_available()
            ):
                raise RuntimeError("PyTorch MPS is unavailable in this execution session")
            mps_probe = torch.ones(1, device="mps") + 1
            if float(mps_probe.cpu().item()) != 2.0:
                raise RuntimeError("PyTorch MPS probe returned an invalid result")

            from subforge.core.asr.whisperx_asr import install_whisperx_runtime_stubs

            install_whisperx_runtime_stubs()
            from whisperx.alignment import align as _align  # noqa: F401
            from whisperx.alignment import load_align_model as _load_align_model  # noqa: F401
            from whisperx.audio import load_audio as _load_audio  # noqa: F401

            from subforge.config import MODEL_PATH
            from subforge.core.asr.whisperx_asr import (
                DEFAULT_EN_ALIGN_MODEL,
                default_mlx_model,
            )

            align_file = MODEL_PATH / "wav2vec2_fairseq_large_lv60k_asr_ls960.pth"
            print("MLX Whisper import: ok")
            print("MLX Metal inference: ok")
            print("PyTorch MPS inference: ok")
            print("WhisperX alignment import: ok")
            print(f"Default MLX model: {default_mlx_model()}")
            print(f"Default align model: {DEFAULT_EN_ALIGN_MODEL}")
            print(f"Align model file: {align_file} ({'found' if align_file.exists() else 'missing'})")
            raise SystemExit(0)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
    if os.environ.get("SUBFORGE_CHECK_DIARIZATION") == "1":
        import traceback

        try:
            from pyannote.audio import Pipeline  # noqa: F401

            from subforge.core.asr.asr_data import ASRData, ASRDataSeg
            from subforge.core.asr.speaker_diarization import (
                SpeakerTurn,
                assign_speakers,
                diarize_audio,
            )

            segments = ASRData([ASRDataSeg("Hello", 0, 1000)])
            turns = [SpeakerTurn(start_ms=0, end_ms=1000, speaker_id="Speaker 1")]
            assigned = assign_speakers(segments, turns)
            if assigned.segments[0].speaker_id != "Speaker 1":
                raise RuntimeError("Packaged speaker assignment returned an invalid result")
            print("Pyannote speaker diarization import: ok")
            print("Speaker assignment smoke test: ok")
            audio_path = os.environ.get("SUBFORGE_DIARIZATION_AUDIO_PATH", "")
            model_path = os.environ.get("SUBFORGE_DIARIZATION_MODEL_PATH", "")
            if audio_path and model_path:
                inferred_turns = diarize_audio(
                    audio_path,
                    model=model_path,
                    model_dir=os.path.dirname(model_path),
                )
                print(f"Speaker diarization inference: {len(inferred_turns)} turns")
            raise SystemExit(0)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
    if os.environ.get("SUBFORGE_CHECK_BACKEND") == "1":
        try:
            from app.main import app

            route_paths = {route.path for route in app.routes}
            required_paths = {
                "/api/health",
                "/api/transcribe/start",
                "/api/subtitle/start",
                "/api/subtitles/load",
            }
            missing = required_paths - route_paths
            if missing:
                raise RuntimeError(f"Packaged backend routes are missing: {sorted(missing)}")
            print("Packaged FastAPI routes: ok")
            check_backend_runtime()
            print("Packaged FastAPI HTTP runtime: ok")
            raise SystemExit(0)
        except Exception:
            import traceback

            traceback.print_exc()
            raise SystemExit(1)
    main()
