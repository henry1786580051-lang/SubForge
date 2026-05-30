#!/usr/bin/env python3
"""Subtitle Desktop App.

Starts FastAPI backend in background and opens a native window.
"""
import base64
import os
import sys
import threading

# Fix path for PyInstaller frozen mode
if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)
    sys.path.insert(0, sys._MEIPASS)
    sys.path.insert(0, os.path.join(sys._MEIPASS, 'backend'))

PORT = 8000
URL = f"http://127.0.0.1:{PORT}"


def start_server():
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=PORT,
        log_level="warning",
    )


class Api:
    """API exposed to the pywebview JavaScript context."""

    def save_file(self, base64_data: str, default_filename: str):
        """Save a base64-encoded file using native save dialog."""
        import webview
        import traceback

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
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

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
        url=URL,
        width=1400,
        height=900,
        min_size=(900, 600),
        text_select=True,
        js_api=api,
    )
    webview.start(debug=False, func=on_loaded)


if __name__ == "__main__":
    main()
