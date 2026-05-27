#!/usr/bin/env python3
"""Subtitle Desktop App.

Starts FastAPI backend in background and opens a native window.
"""
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


def main():
    # Start FastAPI in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Open native window
    import webview
    webview.create_window(
        "Subtitle",
        url=URL,
        width=1400,
        height=900,
        min_size=(900, 600),
        text_select=True,
    )
    webview.start(debug=False)


if __name__ == "__main__":
    main()
