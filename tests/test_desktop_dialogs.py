"""Regression coverage for native picker lifetime, language, and bridge safety."""

import threading
import types
from unittest.mock import Mock

import pytest

import desktop_dialogs as dialogs


@pytest.fixture
def cocoa(monkeypatch):
    panels = []

    class NativeObject:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

    class Panel(NativeObject):
        response = 1
        failure = None
        preview_playing = False

        def __init__(self):
            self.settings = {}
            self.closed = 0
            panels.append(self)

        def __getattr__(self, name):
            if name.startswith("set"):
                return lambda value: self.settings.__setitem__(name, value)
            raise AttributeError(name)

        def runModal(self):
            self.preview_playing = True
            if self.failure:
                raise self.failure
            return self.response

        def URLs(self):
            assert not self.closed
            return [types.SimpleNamespace(path=lambda: "/测试目录/视频.mp4")]

        def close(self):
            self.preview_playing = False
            self.closed += 1

    class Popup(NativeObject):
        def initWithFrame_pullsDown_(self, frame, pulls_down):
            self.index = 0
            return self

        def addItemsWithTitles_(self, titles):
            self.titles = titles

        def setTarget_(self, target):
            self.target = target

        def setAction_(self, action):
            self.action = action

        def indexOfSelectedItem(self):
            return self.index

    kit = types.SimpleNamespace(
        NSObject=NativeObject, NSOpenPanel=Panel, NSPopUpButton=Popup,
        NSMakeRect=lambda *args: args, NSModalResponseOK=1,
    )
    monkeypatch.setattr(dialogs, "_filter_class", None)
    return kit, panels


@pytest.mark.parametrize("response", [0, 1])
def test_picker_closes_playing_preview_before_return_on_import_and_cancel(cocoa, response):
    kit, panels = cocoa
    kit.NSOpenPanel.response = response
    result = dialogs._run_macos_panel(kit, "media")
    assert result == (("/测试目录/视频.mp4",) if response else None)
    assert panels[0].closed == 1
    assert not panels[0].preview_playing
    assert panels[0].settings["setAccessoryView_"].target is None


def test_failed_modal_still_closes_preview(cocoa):
    kit, panels = cocoa
    kit.NSOpenPanel.failure = RuntimeError("native modal failed")
    with pytest.raises(RuntimeError, match="native modal failed"):
        dialogs._run_macos_panel(kit, "media")
    assert panels[0].closed == 1
    assert not panels[0].preview_playing


def test_failed_path_snapshot_still_closes_panel(cocoa, monkeypatch):
    kit, panels = cocoa
    monkeypatch.setattr(kit.NSOpenPanel, "URLs", Mock(side_effect=ValueError("bad URL")))
    with pytest.raises(ValueError, match="bad URL"):
        dialogs._run_macos_panel(kit, "media")
    assert panels[0].closed == 1


def test_reopening_uses_fresh_panel_and_preserves_all_files_filter(cocoa):
    kit, panels = cocoa
    for _ in range(2):
        dialogs._run_macos_panel(kit, "media")
    assert len(panels) == 2 and panels[0] is not panels[1]
    assert all(panel.closed == 1 for panel in panels)
    controller = dialogs._make_filter_controller(kit)
    controller.panel = panels[1]
    controller.filters = dialogs.file_filters("media")
    popup = panels[1].settings["setAccessoryView_"]
    assert popup.titles == ["视频或音频文件", "所有文件"]
    popup.index = 1
    controller.changeType_(popup)
    assert panels[1].settings["setAllowedFileTypes_"] is None
    popup.index = 0
    controller.changeType_(popup)
    assert "mp4" in panels[1].settings["setAllowedFileTypes_"]


@pytest.mark.parametrize("kind", ["media", "subtitle", "any"])
def test_native_titles_and_file_filters_are_chinese(cocoa, kind):
    kit, panels = cocoa
    dialogs._run_macos_panel(kit, kind)
    settings = panels[0].settings
    assert settings["setTitle_"] == ("导入字幕" if kind == "subtitle" else "导入素材")
    assert settings["setPrompt_"] == "导入"
    assert settings["setAllowsMultipleSelection_"] is False
    assert settings["setCanChooseDirectories_"] is False
    extensions = settings["setAllowedFileTypes_"]
    assert ("srt" in extensions) == (kind != "media")
    assert ("mp4" in extensions) == (kind != "subtitle")


def test_native_language_is_process_local_and_preserves_other_arguments(monkeypatch):
    original = {"existing": "value"}
    calls = []
    defaults = types.SimpleNamespace(
        volatileDomainForName_=lambda name: original,
        setVolatileDomain_forName_=lambda value, name: calls.append((value, name)),
    )
    foundation = types.SimpleNamespace(
        NSArgumentDomain="arguments",
        NSUserDefaults=types.SimpleNamespace(standardUserDefaults=lambda: defaults),
    )
    monkeypatch.setattr(dialogs.sys, "platform", "darwin")
    monkeypatch.setitem(dialogs.sys.modules, "Foundation", foundation)
    dialogs.configure_native_language()
    assert original == {"existing": "value"}
    assert calls == [({"existing": "value", "AppleLanguages": ["zh-Hans"]}, "arguments")]


@pytest.mark.parametrize("on_main_thread", [True, False])
@pytest.mark.parametrize("failure", [False, True])
def test_main_thread_dispatch_returns_results_and_propagates_errors(
    monkeypatch, on_main_thread, failure
):
    dispatch = Mock(side_effect=lambda callback: callback())
    monkeypatch.setitem(dialogs.sys.modules, "Foundation", types.SimpleNamespace(
        NSThread=types.SimpleNamespace(isMainThread=lambda: on_main_thread)
    ))
    monkeypatch.setitem(dialogs.sys.modules, "PyObjCTools", types.SimpleNamespace(
        AppHelper=types.SimpleNamespace(callAfter=dispatch)
    ))
    callback = Mock(side_effect=RuntimeError("picker failed")) if failure else Mock(return_value="path")
    if failure:
        with pytest.raises(RuntimeError, match="picker failed"):
            dialogs._main_thread_call(callback)
    else:
        assert dialogs._main_thread_call(callback) == "path"
    assert dispatch.call_count == (0 if on_main_thread else 1)
    callback.assert_called_once()


def test_non_macos_keeps_native_picker_with_chinese_file_types(monkeypatch):
    monkeypatch.setattr(dialogs.sys, "platform", "win32")
    monkeypatch.setitem(dialogs.sys.modules, "webview", types.SimpleNamespace(OPEN_DIALOG="open"))
    window = types.SimpleNamespace(create_file_dialog=Mock(return_value=("C:/video.mp4",)))
    assert dialogs.choose_input_file(window, "media") == ("C:/video.mp4",)
    assert window.create_file_dialog.call_args.kwargs["file_types"][0].startswith("视频或音频文件 (")
    assert window.create_file_dialog.call_args.kwargs["file_types"][1] == "所有文件 (*.*)"


def test_simultaneous_dialog_is_rejected_and_cancel_releases_lock(monkeypatch):
    monkeypatch.setattr(dialogs.sys, "platform", "win32")
    monkeypatch.setitem(dialogs.sys.modules, "webview", types.SimpleNamespace(OPEN_DIALOG="open"))
    entered, release = threading.Event(), threading.Event()

    def picker(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return None

    window = types.SimpleNamespace(create_file_dialog=picker)
    worker = threading.Thread(target=lambda: dialogs.choose_input_file(window))
    worker.start()
    try:
        assert entered.wait(2)
        with pytest.raises(RuntimeError, match="已打开"):
            dialogs.choose_input_file(window)
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert dialogs.choose_input_file(window) is None
