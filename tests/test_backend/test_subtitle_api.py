import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import app.api.config as config_module
from app.api.subtitle import SubtitleRequest, _run_subtitle
from app.api.subtitles import parse_srt
from app.core.task_manager import task_manager

import subforge.core.llm as llm_module
import subforge.core.split.split as split_module


def test_subtitle_request_does_not_default_to_stale_llm_model():
    req = SubtitleRequest(subtitle_file="/tmp/example.srt")

    assert req.llm_model == ""


def test_backend_parse_srt_preserves_bilingual_fields():
    srt = """1
00:00:00,000 --> 00:00:02,000
你肯定还认得出这辆1986年的梅赛德斯-奔驰420 SEL
you should also recognize this 1986 Mercedes-Benz 420 SEL
"""

    segments = parse_srt(srt)

    assert segments == [
        {
            "id": 1,
            "start": "00:00:00.000",
            "end": "00:00:02.000",
            "text": "you should also recognize this 1986 Mercedes-Benz 420 SEL",
            "translated": "你肯定还认得出这辆1986年的梅赛德斯-奔驰420 SEL",
        }
    ]


def test_subtitle_pipeline_uses_explicit_llm_client_without_env_mutation(
    tmp_path,
    monkeypatch,
):
    import asyncio
    import os

    subtitle_path = tmp_path / "input.srt"
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello world\n",
        encoding="utf-8",
    )

    settings = {
        "custom_prompt": "",
        "llm_model": "mimo-v2.5",
        "llm_api_key": "task-key",
        "llm_base_url": "https://example.test/v1",
        "thread_num": 1,
        "batch_size": 1,
    }

    monkeypatch.setattr(config_module, "get_config_value", lambda key, default=None: settings.get(key, default))

    created = {}
    client = object()

    def fake_create_client(base_url: str, api_key: str):
        created["base_url"] = base_url
        created["api_key"] = api_key
        return client

    monkeypatch.setattr(llm_module, "create_client", fake_create_client)

    class FakeSplitter:
        def __init__(self, thread_num, model, llm_client=None, **_kwargs):
            created["splitter_client"] = llm_client

        def split_subtitle(self, asr_data):
            return asr_data

    monkeypatch.setattr(split_module, "SubtitleSplitter", FakeSplitter)
    monkeypatch.setenv("OPENAI_API_KEY", "original-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://original.test/v1")

    task = task_manager.create_task("subtitle")
    req = SubtitleRequest(
        subtitle_file=str(subtitle_path),
        need_optimize=False,
        need_translate=False,
    )

    asyncio.run(_run_subtitle(task.id, req))

    assert created == {
        "base_url": "https://example.test/v1",
        "api_key": "task-key",
        "splitter_client": client,
    }
    assert os.environ["OPENAI_API_KEY"] == "original-key"
    assert os.environ["OPENAI_BASE_URL"] == "https://original.test/v1"
