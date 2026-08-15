from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import smoke_desktop


def test_macos_aqua_check_launches_app_with_runtime_environment(tmp_path, monkeypatch):
    bundle = tmp_path / "SubForge.app"
    bundle.mkdir()
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        stdout = Path(command[command.index("-o") + 1])
        stderr = Path(command[command.index("--stderr") + 1])
        stdout.write_text(
            "MLX Metal inference: ok\nPyTorch MPS inference: ok\n",
            encoding="utf-8",
        )
        stderr.write_text("", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(smoke_desktop.subprocess, "run", fake_run)

    smoke_desktop._run_macos_aqua_check(
        bundle,
        {"TORCH_USE_RTLD_GLOBAL": "1", "SUBFORGE_CHECK_ASR": "1"},
        ("MLX Metal inference: ok", "PyTorch MPS inference: ok"),
        timeout_seconds=30,
    )

    command = captured["command"]
    assert command[:4] == ["open", "-W", "-n", "-g"]
    assert "SUBFORGE_CHECK_ASR=1" in command
    assert "TORCH_USE_RTLD_GLOBAL=1" in command
    assert command[-1] == str(bundle)
    assert captured["kwargs"] == {"check": False, "timeout": 30}


def test_macos_aqua_check_rejects_missing_acceleration_marker(tmp_path, monkeypatch):
    bundle = tmp_path / "SubForge.app"
    bundle.mkdir()

    def fake_run(command, **_kwargs):
        stdout = Path(command[command.index("-o") + 1])
        stdout.write_text("MLX Whisper import: ok\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(smoke_desktop.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="PyTorch MPS inference: ok"):
        smoke_desktop._run_macos_aqua_check(
            bundle,
            {"SUBFORGE_CHECK_ASR": "1"},
            ("MLX Whisper import: ok", "PyTorch MPS inference: ok"),
        )


def test_macos_aqua_check_requires_app_bundle(tmp_path):
    with pytest.raises(RuntimeError, match=r"\.app bundle"):
        smoke_desktop._run_macos_aqua_check(
            tmp_path / "SubForge",
            {"SUBFORGE_CHECK_ASR": "1"},
            ("MLX Metal inference: ok",),
        )
