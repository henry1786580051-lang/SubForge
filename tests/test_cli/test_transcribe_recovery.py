from argparse import Namespace
from unittest.mock import Mock

from subforge.cli import exit_codes as EXIT
from subforge.cli.commands import transcribe
from subforge.core import asr
from subforge.core.asr.asr_data import ASRData, ASRDataSeg


def test_transcribe_preserves_partial_result_and_stops_progress(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.touch()
    destination = tmp_path / "source.srt"
    destination.write_text("existing subtitle", encoding="utf-8")
    data = ASRData([ASRDataSeg("Confirmed speech.", 0, 1000)])
    data.coverage_issues = [{"start": 10, "end": 14.2, "reason": "context_disagreement"}]
    monkeypatch.setattr(asr, "transcribe", lambda *args, **kwargs: data)
    progress = Mock()
    progress.start.return_value = progress
    monkeypatch.setattr(transcribe.output, "ProgressLine", lambda message: progress)
    args = Namespace(input=str(source), output=str(destination), quiet=False)
    status = transcribe.run(args, {"transcribe": {"asr": "whisperx"}})
    assert status == EXIT.RUNTIME_ERROR
    assert destination.read_text() == "existing subtitle"
    assert "Confirmed speech." in (tmp_path / "source_recovery.srt").read_text(encoding="utf-8-sig")
    progress.fail.assert_called_once()
    progress.finish.assert_not_called()
