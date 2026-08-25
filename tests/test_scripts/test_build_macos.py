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
    sealed: list[Path] = []

    def fake_run(command, **kwargs):
        commands.append(command)

    def fake_build_dmg(**kwargs):
        staged_app = Path(kwargs["settings"]["files"][0])
        captured["staged_app"] = staged_app
        captured["exists_during_build"] = staged_app.exists()
        captured["settings"] = kwargs["settings"]
        Path(kwargs["filename"]).write_bytes(b"dmg")

    monkeypatch.setattr(build_macos.subprocess, "run", fake_run)
    monkeypatch.setattr(build_macos, "DMG_OUTPUT", tmp_path / "SubForge.dmg")
    monkeypatch.setenv("SUBFORGE_CODESIGN_IDENTITY", "Developer ID Application: SubForge")
    monkeypatch.setattr(
        build_macos,
        "seal_app_inside_dmg",
        lambda path: sealed.append(path),
    )

    # Import inside the function just as production code does.
    import dmgbuild

    monkeypatch.setattr(dmgbuild, "build_dmg", fake_build_dmg)

    build_macos.create_dmg(source_app)

    staged_app = captured["staged_app"]
    settings = captured["settings"]
    assert captured["exists_during_build"] is True
    assert settings["background"] == str(build_macos.DMG_BACKGROUND)
    assert settings["window_rect"] == ((200, 120), (720, 440))
    assert settings["icon_locations"] == {
        "SubForge.app": (190, 250),
        "Applications": (530, 250),
    }
    assert settings["icon_size"] == 104
    assert commands == [
        ["xattr", "-cr", str(staged_app)],
        [
            "codesign",
            "--force",
            "--deep",
            "--sign",
            "Developer ID Application: SubForge",
            str(staged_app),
        ],
        ["codesign", "--verify", "--deep", "--strict", str(staged_app)],
    ]
    assert sealed == [tmp_path / "SubForge.dmg"]


@pytest.mark.skipif(sys.platform != "darwin", reason="DMG signing is macOS-only")
def test_seal_app_inside_dmg_clears_metadata_and_verifies_image_copy(
    tmp_path, monkeypatch
):
    dmg_path = tmp_path / "SubForge.dmg"
    dmg_path.write_bytes(b"initial")
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[:2] == ["hdiutil", "convert"]:
            output = Path(command[command.index("-o") + 1])
            output.write_bytes(b"converted")

    monkeypatch.setattr(build_macos.subprocess, "run", fake_run)

    build_macos.seal_app_inside_dmg(dmg_path)

    image_app = Path(commands[1][-1]) / "SubForge.app"
    assert commands[0][:2] == ["hdiutil", "convert"]
    assert commands[1][:2] == ["hdiutil", "attach"]
    assert commands[2] == ["xattr", "-cr", str(image_app)]
    assert commands[3][:2] == ["codesign", "--verify"]
    assert commands[4][:2] == ["hdiutil", "detach"]
    assert commands[5][:2] == ["hdiutil", "convert"]
    assert dmg_path.read_bytes() == b"converted"
