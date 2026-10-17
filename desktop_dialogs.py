"""Chinese native import dialogs with an explicit, bounded preview lifetime."""

from __future__ import annotations

import sys
import threading
from typing import Any, Callable

MEDIA_EXTENSIONS = ("mp4", "mov", "mkv", "avi", "webm", "mp3", "wav", "m4a", "flac")
SUBTITLE_EXTENSIONS = ("srt", "vtt", "ass")
_dialog_lock = threading.Lock()
_filter_class = None


def configure_native_language() -> None:
    """Use Chinese for this process's AppKit UI without changing OS preferences."""
    if sys.platform != "darwin":
        return
    # PyObjC exposes Foundation symbols lazily through runtime metadata.
    from Foundation import NSArgumentDomain, NSUserDefaults  # pyright: ignore

    defaults = NSUserDefaults.standardUserDefaults()
    arguments = dict(defaults.volatileDomainForName_(NSArgumentDomain) or {})
    arguments["AppleLanguages"] = ["zh-Hans"]
    defaults.setVolatileDomain_forName_(arguments, NSArgumentDomain)


def file_filters(kind: str) -> list[tuple[str, tuple[str, ...]]]:
    if kind == "subtitle":
        return [("字幕文件", SUBTITLE_EXTENSIONS)]
    if kind == "any":
        return [("视频、音频和字幕文件", MEDIA_EXTENSIONS + SUBTITLE_EXTENSIONS)]
    return [("视频或音频文件", MEDIA_EXTENSIONS), ("所有文件", ())]


def _main_thread_call(callback: Callable[[], Any]) -> Any:
    from Foundation import NSThread  # pyright: ignore[reportAttributeAccessIssue]
    from PyObjCTools import AppHelper

    if NSThread.isMainThread():
        return callback()
    finished = threading.Event()
    result: list[Any] = []
    errors: list[BaseException] = []

    def run():
        try:
            result.append(callback())
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    AppHelper.callAfter(run)
    finished.wait()  # The user may browse for any length of time.
    if errors:
        raise errors[0]
    return result[0]


def _make_filter_controller(AppKit: Any):
    global _filter_class
    if _filter_class is None:
        class SubForgeImportTypeFilter(AppKit.NSObject):
            def changeType_(self, sender):
                extensions = self.filters[sender.indexOfSelectedItem()][1]
                self.panel.setAllowedFileTypes_(list(extensions) or None)

        _filter_class = SubForgeImportTypeFilter
    return _filter_class.alloc().init()


def _run_macos_panel(AppKit: Any, kind: str) -> tuple[str, ...] | None:
    # Own a fresh panel instead of leaving pywebview's dismissed modal panel
    # (and its Quick Look/icon player) alive after the bridge returns a path.
    panel = AppKit.NSOpenPanel.alloc().init()
    controller = None
    popup = None
    try:
        panel.setReleasedWhenClosed_(False)
        panel.setTitle_("导入字幕" if kind == "subtitle" else "导入素材")
        panel.setPrompt_("导入")
        panel.setMessage_("请选择要导入的字幕文件" if kind == "subtitle" else "请选择要导入的本地文件")
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(False)
        panel.setAllowsMultipleSelection_(False)
        filters = file_filters(kind)
        panel.setAllowedFileTypes_(list(filters[0][1]))
        if len(filters) > 1:
            controller = _make_filter_controller(AppKit)
            controller.panel = panel
            controller.filters = filters
            popup = AppKit.NSPopUpButton.alloc().initWithFrame_pullsDown_(
                AppKit.NSMakeRect(0, 0, 280, 28), False
            )
            popup.addItemsWithTitles_([title for title, _ in filters])
            popup.setTarget_(controller)
            popup.setAction_("changeType:")
            panel.setAccessoryView_(popup)
            panel.setAccessoryViewDisclosed_(True)
        if panel.runModal() == AppKit.NSModalResponseOK:
            # Copy URLs to Python strings before closing the panel/preview.
            return tuple(str(url.path()) for url in panel.URLs())
        return None
    finally:
        try:
            # runModal dismisses a panel; close explicitly tears down its
            # window and hosted media previews on success, cancel, or failure.
            panel.close()
        finally:
            if popup is not None:
                popup.setTarget_(None)
            if controller is not None:
                controller.panel = None


def choose_input_file(window: Any, kind: str = "media") -> Any:
    """Return only after the picker and its previews have been closed."""
    if not _dialog_lock.acquire(blocking=False):
        raise RuntimeError("文件选择窗口已打开，请先完成选择或取消")
    try:
        if sys.platform == "darwin":
            import AppKit

            return _main_thread_call(lambda: _run_macos_panel(AppKit, kind))
        import webview

        filters = tuple(
            f"{title} ({';'.join('*.' + ext for ext in extensions) if extensions else '*.*'})"
            for title, extensions in file_filters(kind)
        )
        return window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=filters)
    finally:
        _dialog_lock.release()
