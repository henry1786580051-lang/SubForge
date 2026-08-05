"""CLI parser coverage for the supported command surface."""

import pytest

from subforge.cli import exit_codes as EXIT
from subforge.cli.main import main


def test_no_args_prints_help(capsys):
    assert main([]) == EXIT.USAGE_ERROR
    assert "transcribe" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "subforge" in capsys.readouterr().out


def test_help_lists_only_supported_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for command in ("transcribe", "subtitle", "process", "download", "config", "doctor"):
        assert command in output
    for removed in ("gui", "dub", "synthesize", "style"):
        assert removed not in output


def test_invalid_subcommand():
    with pytest.raises(SystemExit) as exc:
        main(["nonexistent"])
    assert exc.value.code == 2


def test_transcribe_file_not_found():
    assert main(["transcribe", "/nonexistent/file.mp4"]) == EXIT.FILE_NOT_FOUND


def test_transcribe_rejects_invalid_asr():
    with pytest.raises(SystemExit) as exc:
        main(["transcribe", "test.mp4", "--asr", "invalid"])
    assert exc.value.code == 2


def test_subtitle_file_not_found():
    assert main(["subtitle", "/nonexistent/file.srt"]) == EXIT.FILE_NOT_FOUND


def test_subtitle_rejects_invalid_translator():
    with pytest.raises(SystemExit) as exc:
        main(["subtitle", "test.srt", "--translator", "invalid"])
    assert exc.value.code == 2


def test_process_file_not_found():
    assert main(["process", "/nonexistent/file.mp4"]) == EXIT.FILE_NOT_FOUND


def test_config_init_template_has_no_removed_sections(capsys):
    result = main(["config", "init", "--non-interactive", "--print-template"])
    assert result == EXIT.SUCCESS
    output = capsys.readouterr().out
    assert "[subtitle]" in output
    assert "[dubbing]" not in output
    assert "[synthesize]" not in output


def test_doctor_json(capsys):
    result = main(["doctor", "--json"])
    assert result in {EXIT.SUCCESS, EXIT.DEPENDENCY_MISSING}
    assert '"checks"' in capsys.readouterr().out
