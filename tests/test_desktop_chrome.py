import desktop_chrome


def test_bridge_state_is_bounded_and_does_not_trust_truthy_values():
    state = desktop_chrome.normalize_state(
        {
            "title": "x" * 400,
            "status": "y" * 400,
            "appearance": "unknown",
            "can_export": "true",
            "can_inspect": 1,
            "running": True,
            "inspector_open": True,
            "script": "ignored",
        }
    )
    assert len(state["title"]) == 180
    assert len(state["status"]) == 90
    assert state["appearance"] == "system"
    assert state["can_export"] is False
    assert state["can_inspect"] is False
    assert state["running"] is True
    assert "script" not in state


def test_bridge_fallback_needs_no_appkit(monkeypatch):
    monkeypatch.setattr(desktop_chrome, "_controller", None)
    monkeypatch.setattr(desktop_chrome.sys, "platform", "linux")
    assert desktop_chrome.install_native_toolbar(None) is False
    assert desktop_chrome.get_desktop_state() == {"toolbar": False}
    assert desktop_chrome.sync_desktop_state(None) == {"toolbar": False}


def test_bridge_capabilities_are_copied_and_install_is_idempotent(monkeypatch):
    from types import SimpleNamespace

    controller = SimpleNamespace(capabilities={"toolbar": True, "reduce_motion": True})
    monkeypatch.setattr(desktop_chrome, "_controller", controller)
    desktop_chrome.get_desktop_state()["toolbar"] = False
    assert desktop_chrome.install_native_toolbar(None) is True
