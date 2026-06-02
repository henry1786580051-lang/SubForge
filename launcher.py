#!/usr/bin/env python3
"""Subtitle Desktop App.

Starts FastAPI backend in background and opens a native window.
"""
import base64
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

# Fix path for PyInstaller frozen mode
if getattr(sys, 'frozen', False):
    sys.path.insert(0, sys._MEIPASS)
    sys.path.insert(0, os.path.join(sys._MEIPASS, 'backend'))

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


def start_server(port: int, server_errors: list[str]):
    import uvicorn

    try:
        uvicorn.run(
            "app.main:app",
            host=HOST,
            port=port,
            log_level="warning",
        )
    except Exception as exc:
        server_errors.append(str(exc))


def wait_for_server(url: str, server_errors: list[str], timeout_seconds: float = 15.0) -> bool:
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
    server_thread = threading.Thread(target=start_server, args=(port, server_errors), daemon=True)
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

    webview.create_window(
        "SubForge",
        url=url,
        width=1400,
        height=900,
        min_size=(900, 600),
        text_select=True,
        js_api=api,
    )
    webview.start(debug=False, func=on_loaded)


if __name__ == "__main__":
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
    main()
