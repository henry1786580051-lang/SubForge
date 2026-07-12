import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import app.api.config as config_module
from app.api.subtitle import SubtitleRequest, _run_subtitle
from app.api.subtitles import parse_srt
from app.core.task_manager import task_manager

import subforge.core.llm as llm_module
import subforge.core.split.split as split_module
from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.factory import TranslatorFactory


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


def test_subtitle_pipeline_cleans_chinese_translation_punctuation(tmp_path, monkeypatch):
    import asyncio

    subtitle_path = tmp_path / "input.srt"
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello, world.\n",
        encoding="utf-8",
    )
    settings = {
        "thread_num": 1,
        "batch_size": 1,
        "replace_chinese_punctuation": True,
    }
    monkeypatch.setattr(
        config_module,
        "get_config_value",
        lambda key, default=None: settings.get(key, default),
    )

    class FakeSplitter:
        def __init__(self, **_kwargs):
            pass

        def split_subtitle(self, asr_data):
            return asr_data

    class FakeTranslator:
        def translate_subtitle(self, asr_data):
            asr_data.segments[0].translated_text = "你好，世界。"
            return asr_data

    monkeypatch.setattr(split_module, "SubtitleSplitter", FakeSplitter)
    monkeypatch.setattr(
        TranslatorFactory,
        "create_translator",
        staticmethod(lambda **_kwargs: FakeTranslator()),
    )

    task = task_manager.create_task("subtitle")
    asyncio.run(
        _run_subtitle(
            task.id,
            SubtitleRequest(
                subtitle_file=str(subtitle_path),
                target_language="chinese",
                translator="bing",
                need_optimize=False,
                need_translate=True,
            ),
        )
    )

    output = subtitle_path.with_stem("input_processed").read_text(encoding="utf-8")
    assert "你好 世界" in output
    assert "Hello, world." in output


def test_subtitle_pipeline_does_not_split_bilingual_cues_after_translation(
    tmp_path,
    monkeypatch,
):
    import asyncio

    subtitle_path = tmp_path / "input.srt"
    source = (
        "I don't really think there's any point in going into sport, "
        "especially since we're trying to be efficient."
    )
    subtitle_path.write_text(
        f"1\n00:00:00,000 --> 00:00:04,000\n{source}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        config_module,
        "get_config_value",
        lambda key, default=None: {
            "thread_num": 1,
            "batch_size": 1,
            "replace_chinese_punctuation": True,
        }.get(key, default),
    )

    class FakeSplitter:
        def __init__(self, **_kwargs):
            pass

        def split_subtitle(self, asr_data):
            return asr_data

    class FakeTranslator:
        def translate_subtitle(self, asr_data):
            asr_data.segments[0].translated_text = (
                "我觉得没必要切到运动模式 毕竟咱们现在是奔着省油去的"
            )
            return asr_data

    monkeypatch.setattr(split_module, "SubtitleSplitter", FakeSplitter)
    monkeypatch.setattr(
        TranslatorFactory,
        "create_translator",
        staticmethod(lambda **_kwargs: FakeTranslator()),
    )

    task = task_manager.create_task("subtitle")
    asyncio.run(
        _run_subtitle(
            task.id,
            SubtitleRequest(
                subtitle_file=str(subtitle_path),
                target_language="chinese",
                translator="bing",
                need_optimize=False,
                need_translate=True,
            ),
        )
    )

    output_path = subtitle_path.with_stem("input_processed")
    output = output_path.read_text(encoding="utf-8")
    assert output.count(" --> ") == 1
    assert source in output
    assert "我觉得没必要切到运动模式 毕竟咱们现在是奔着省油去的" in output


def test_failed_translation_saves_punctuation_cleaned_recovery_file(
    tmp_path,
    monkeypatch,
):
    import asyncio

    subtitle_path = tmp_path / "input.srt"
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nFirst source.\n\n"
        "2\n00:00:01,100 --> 00:00:02,000\nSecond source.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        config_module,
        "get_config_value",
        lambda key, default=None: {
            "thread_num": 1,
            "batch_size": 2,
            "replace_chinese_punctuation": True,
        }.get(key, default),
    )

    class FakeSplitter:
        def __init__(self, **_kwargs):
            pass

        def split_subtitle(self, asr_data):
            return asr_data

    class FakeTranslator:
        def __init__(self, update_callback):
            self.update_callback = update_callback

        def translate_subtitle(self, _asr_data):
            self.update_callback(
                [
                    SubtitleProcessData(
                        index=1,
                        original_text="First source.",
                        translated_text="第一条，已经完成。",
                    )
                ]
            )
            raise RuntimeError("second item failed")

    monkeypatch.setattr(split_module, "SubtitleSplitter", FakeSplitter)
    monkeypatch.setattr(
        TranslatorFactory,
        "create_translator",
        staticmethod(
            lambda **kwargs: FakeTranslator(kwargs["update_callback"])
        ),
    )

    task = task_manager.create_task("subtitle")
    asyncio.run(
        _run_subtitle(
            task.id,
            SubtitleRequest(
                subtitle_file=str(subtitle_path),
                target_language="chinese",
                translator="bing",
                need_optimize=False,
                need_translate=True,
            ),
        )
    )

    recovery_path = subtitle_path.with_stem("input_recovery")
    recovery = recovery_path.read_text(encoding="utf-8")
    task_result = task_manager.get_task(task.id)

    assert "第一条 已经完成" in recovery
    assert "第二条" not in recovery
    assert task_result.status.value == "failed"
    assert task_result.result == {"recovery_file": str(recovery_path)}
    assert task_result.subtitle_file == str(recovery_path)
