"""Native external-drive input grants must follow saved pipeline outputs only."""

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app import security
from app.api import config, subtitle, subtitles, transcribe
from app.core.task_manager import task_manager

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.translate.factory import TranslatorFactory
from subforge.settings import LlmRuntimeConfig


@pytest.fixture
def external_input(tmp_path, monkeypatch):
    monkeypatch.setattr(security, "_get_allowed_roots", lambda: [])
    monkeypatch.setattr(security, "_granted_paths", set())
    monkeypatch.setattr(config, "get_config_value", lambda key, default=None: default)
    monkeypatch.setattr(
        config,
        "get_llm_runtime_config",
        lambda: LlmRuntimeConfig(
            provider="custom",
            base_url="",
            api_key="",
            model="",
        ),
    )
    source = tmp_path / "external.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nThe car is ready.\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nLet's go.\n",
        encoding="utf-8",
    )
    security.grant_path(source)
    return source


@pytest.mark.parametrize("fail", [False, True])
def test_subtitle_output_and_recovery_remain_accessible(external_input, monkeypatch, fail):
    split_module = importlib.import_module("subforge.core.split.split")

    def forbidden_splitter(**kwargs):
        pytest.fail("Smart splitting is disabled: no LLM splitter may be constructed")

    monkeypatch.setattr(split_module, "SubtitleSplitter", forbidden_splitter)

    class Translator:
        def translate_subtitle(self, data):
            data.segments[0].translated_text = "车已准备好。"
            if fail:
                raise RuntimeError("Test provider unavailable")
            data.segments[1].translated_text = "出发吧。"
            return data

    monkeypatch.setattr(TranslatorFactory, "create_translator", lambda **kwargs: Translator())
    task = task_manager.create_task("subtitle")
    asyncio.run(
        subtitle._run_subtitle(
            task.id,
            subtitle.SubtitleRequest(
                subtitle_file=str(external_input),
                need_optimize=False,
                need_translate=True,
                translator="bing",
            ),
        )
    )
    completed = task_manager.get_task(task.id)
    assert completed.status == ("failed" if fail else "completed")
    output = completed.result["recovery_file" if fail else "subtitle_file"]
    assert security.validate_path(output).is_file()
    loaded = asyncio.run(subtitles.load_subtitle(output))
    assert loaded["count"] == 2
    assert [s["text"] for s in loaded["segments"]] == ["The car is ready.", "Let's go."]
    assert loaded["segments"][0]["translated"] == "车已准备好"
    with pytest.raises(ValueError):
        security.validate_path(str(external_input.with_name("private.txt")))


def test_transcription_output_can_be_loaded_from_external_drive(external_input, monkeypatch):
    video = external_input.with_suffix(".mp4")
    video.touch()
    security.grant_path(video)
    monkeypatch.setattr(
        transcribe,
        "_build_transcribe_config",
        lambda *args: SimpleNamespace(detect_additional_languages=False),
    )
    video_utils = importlib.import_module("subforge.core.utils.video_utils")
    asr_module = importlib.import_module("subforge.core.asr.transcribe")
    monkeypatch.setattr(video_utils, "video2audio", lambda *args: True)
    monkeypatch.setattr(
        asr_module, "transcribe", lambda *args: ASRData([ASRDataSeg("New transcription.", 0, 1000)])
    )
    # The input SRT must not already be granted: the transcriber grants its own output.
    security.clear_granted_paths()
    security.grant_path(video)
    task = task_manager.create_task("transcribe")
    asyncio.run(
        transcribe._run_transcription(
            task.id, transcribe.TranscribeRequest(file_path=str(video), model="whisper_cpp")
        )
    )
    completed = task_manager.get_task(task.id)
    assert completed.status == "completed", completed.error
    output = completed.result["subtitle_file"]
    assert security.validate_path(output).is_file()
    assert (
        asyncio.run(subtitles.load_subtitle(output))["segments"][0]["text"] == "New transcription."
    )
    with pytest.raises(ValueError):
        security.validate_path(str(video.with_name("private.txt")))
