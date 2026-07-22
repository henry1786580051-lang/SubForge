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


def _cleanup_desktop_session(server=None) -> None:
    """Request backend shutdown and remove files owned by this process."""
    if server is not None:
        server.should_exit = True
    try:
        from app.api.files import cleanup_session_uploads

        cleanup_session_uploads()
    except Exception:
        # Closing the native window must remain reliable even during a partial
        # backend startup or interpreter teardown.
        pass


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
                with open(file_path, "wb") as f:
                    f.write(data)
                print(f"[Api] saved to: {file_path}")
                return {"ok": True, "path": file_path}
            print("[Api] dialog cancelled")
            return {"ok": False}
        except Exception as e:
            print(f"[Api] save_file error: {e}")
            traceback.print_exc()
            return {"ok": False, "error": str(e)}


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

    if not wait_for_server(url, server_errors):
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
        if exit_started.is_set():
            return
        exit_started.set()
        _cleanup_desktop_session(server_holder[0] if server_holder else None)
        os._exit(0)

    try:
        window.events.closed += force_exit
    except Exception:
        pass

    webview.start(debug=False, func=on_loaded)
    force_exit()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    if os.environ.get("SUBFORGE_CHECK_DENOISE") == "1":
        import traceback

        from subforge.core.asr.audio_enhancer import is_available

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
        raise SystemExit(0 if available else 1)
    if os.environ.get("SUBFORGE_CHECK_ASR") == "1":
        import traceback

        try:
            import mlx_whisper  # noqa: F401

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
            print("WhisperX alignment import: ok")
            print(f"Default MLX model: {default_mlx_model()}")
            print(f"Default align model: {DEFAULT_EN_ALIGN_MODEL}")
            print(f"Align model file: {align_file} ({'found' if align_file.exists() else 'missing'})")
            raise SystemExit(0)
        except RuntimeError as exc:
            if "No Metal device available" in str(exc):
                print("MLX Whisper package import reached Metal initialization.")
                print("Metal device is not available in this execution environment.")
                raise SystemExit(0)
            traceback.print_exc()
            raise SystemExit(1)
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
