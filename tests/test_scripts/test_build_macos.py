import sys
from pathlib import Path

import pytest

import build_macos


@pytest.mark.skipif(sys.platform != "darwin", reason="DMG signing is macOS-only")
def test_create_dmg_resigns_exact_staged_app(tmp_path, monkeypatch):
    source_app = tmp_path / "source" / "SubForge.app"
    executable = source_app / "Contents" / "MacOS" / "SubForge"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"app")

    commands: list[list[str]] = []
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        commands.append(command)

    def fake_build_dmg(**kwargs):
        staged_app = Path(kwargs["settings"]["files"][0])
        captured["staged_app"] = staged_app
        captured["exists_during_build"] = staged_app.exists()
        Path(kwargs["filename"]).write_bytes(b"dmg")

    monkeypatch.setattr(build_macos.subprocess, "run", fake_run)
    monkeypatch.setattr(build_macos, "DMG_OUTPUT", tmp_path / "SubForge.dmg")

    # Import inside the function just as production code does.
    import dmgbuild

    monkeypatch.setattr(dmgbuild, "build_dmg", fake_build_dmg)

    build_macos.create_dmg(source_app)

    staged_app = captured["staged_app"]
    assert captured["exists_during_build"] is True
    assert commands == [
        ["xattr", "-cr", str(staged_app)],
        ["codesign", "--force", "--deep", "--sign", "-", str(staged_app)],
        ["codesign", "--verify", "--deep", "--strict", str(staged_app)],
    ]
