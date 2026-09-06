"""Source-backed equivalence, conservative exclusions, and request reuse."""

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.context import TranslationContext
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.quality.closed_boundary import is_closed_soft_boundary
from subforge.core.translate.quality.preservation import inspect_preserved_tokens
from subforge.core.translate.quality.verdict_cache import PositiveVerdictCache
from subforge.core.translate.types import TargetLanguage


def make_translator():
    return LLMTranslator(
        thread_num=1,
        batch_num=20,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="test",
        custom_prompt="",
        is_reflect=False,
        update_callback=None,
    )


def inspect(source, target, language="简体中文"):
    return inspect_preserved_tokens(
        {"1": target},
        {"1": source},
        str,
        target_language_value=language,
        localized_magnitude_rendered=LLMTranslator._localized_magnitude_rendered,
    )


@pytest.mark.parametrize(
    "target",
    [
        "三十五六万……不对，三万多美元买一台非TCR版",
        "三十五六万……不对，三万五左右买一台非TCR版",
    ],
)
def test_withdrawn_price_expansion_does_not_waive_fabricated_correction(target):
    source = "It is a pretty decent bang for your buck here for in the mid-30s for a non-TCR"
    assert inspect(source, target)


@pytest.mark.parametrize(
    "source,target",
    [
        ("It stays at 3,000 RPM.", "它保持在3000转左右"),
        ("It runs at 2500rpm.", "它在2500转时很顺畅"),
        ("At 6,500 RPM it is loud.", "6500轉時很吵"),
        ("The fan is at 1200 RPM.", "风扇以1200转运行"),
    ],
)
def test_rpm_accepts_attached_chinese_continuation(source, target):
    assert not inspect(source, target)


@pytest.mark.parametrize(
    "target",
    [
        "它保持在13000转左右",
        "它保持在300转左右",
        "它以3000转账",
        "它需要3000转弯",
        "3000元 还有2500转左右",
        "它保持在3000左右",
    ],
)
def test_rpm_requires_exact_bound_quantity_and_real_unit(target):
    assert inspect("It stays at 3000 RPM.", target)


def test_preservation_feedback_is_stable_across_hash_seeds():
    code = """
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.quality.preservation import inspect_preserved_tokens
from subforge.core.translate.quality.invariants import legacy_preserved_token_message
print(legacy_preserved_token_message(inspect_preserved_tokens(
{'7':'全部丢失'}, {'7':'2024 then 10000 and 36000 then 2024.'}, str,
target_language_value='简体中文', localized_magnitude_rendered=LLMTranslator._localized_magnitude_rendered)))
"""
    results = [
        subprocess.check_output(
            [sys.executable, "-c", code], text=True, env={**os.environ, "PYTHONHASHSEED": seed}
        )
        for seed in ["1", "7", "91"]
    ]
    assert len(set(results)) == 1
    assert "['7:2024', '7:10000', '7:36000']" in results[0]


@pytest.mark.parametrize(
    "source,left,right,signal",
    [
        (
            "This is very convenient.",
            "这个用起来很方便的",
            "我们还可以继续",
            "possible function-word split",
        ),
        (
            "That is what I have been doing.",
            "我一直是这么做的",
            "你可以看看",
            "possible function-word split",
        ),
        ("Please let me know.", "有问题请告诉我", "我会回答", "possible pronoun boundary"),
    ],
)
def test_closed_source_and_complete_target_remove_only_weak_signal(source, left, right, signal):
    assert is_closed_soft_boundary(source, "We can continue.", left, right, signal)
    t = make_translator()
    t._all_source_by_index = {1: source, 2: "We can continue."}
    assert t._target_boundary_signal(1, left, right) == ""
    source_items = [
        SubtitleProcessData(index=i, original_text=s) for i, s in t._all_source_by_index.items()
    ]
    translated = {
        1: replace(source_items[0], translated_text=left),
        2: replace(source_items[1], translated_text=right),
    }
    assert t._chinese_fluency_candidates(source_items, translated) == []
    t.stop()


@pytest.mark.parametrize(
    "source,left,right,signal",
    [
        (
            "This is very convenient",
            "这个用起来很方便的",
            "我们还可以继续",
            "possible function-word split",
        ),
        (
            "This is very convenient...",
            "这个用起来很方便的",
            "我们还可以继续",
            "possible function-word split",
        ),
        (
            "This is very convenient.",
            "这个用起来很方便的",
            "东西都在这里",
            "possible function-word split",
        ),
        ("I think he.", "我觉得他", "他是对的", "possible pronoun boundary"),
        ("It is what I want.", "这是我想要的", "颜色和形状", "possible function-word split"),
        (
            "This is very convenient.",
            "这个用起来很方便的",
            "我们还可以继续",
            "numeric value separated from its unit",
        ),
    ],
)
def test_open_ambiguous_and_strong_boundaries_remain_candidates(source, left, right, signal):
    assert not is_closed_soft_boundary(source, "We can continue.", left, right, signal)


def test_long_pause_remains_reviewable():
    t = make_translator()
    t._all_source_by_index = {1: "Please let me know.", 2: "We can continue."}
    t._gap_after_index = {1: 3000}
    assert t._target_boundary_signal(1, "有问题请告诉我", "我会回答")
    t.stop()


def test_cache_only_remembers_success_and_is_bounded():
    cache = PositiveVerdictCache(capacity=1)
    calls = []

    def passed():
        calls.append("pass")

    def failed():
        calls.append("fail")
        raise ValueError("not valid")

    for _ in range(2):
        with pytest.raises(ValueError):
            cache.validate("bad", failed)
    cache.validate("a", passed)
    cache.validate("a", passed)
    cache.validate("b", passed)
    cache.validate("a", passed)
    assert calls == ["fail", "fail", "pass", "pass", "pass"]
    cache.clear()
    cache.validate("a", passed)
    assert len(calls) == 6


def test_concurrent_identical_checks_are_coalesced():
    cache = PositiveVerdictCache()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def check():
        calls.append(1)
        entered.set()
        assert release.wait(3)

    with ThreadPoolExecutor(2) as executor:
        a = executor.submit(cache.validate, "same", check)
        assert entered.wait(3)
        b = executor.submit(cache.validate, "same", check)
        release.set()
        a.result()
        b.result()
    assert calls == [1]


def test_cache_clear_does_not_reuse_old_inflight_result():
    cache = PositiveVerdictCache()
    entered, release = threading.Event(), threading.Event()

    def check():
        entered.set()
        assert release.wait(3)

    with ThreadPoolExecutor(1) as executor:
        old = executor.submit(cache.validate, "same", check)
        assert entered.wait(3)
        cache.clear()
        release.set()
        old.result()
    calls = []
    cache.validate("same", lambda: calls.append(1))
    assert calls == [1]


@pytest.mark.parametrize(
    "change",
    [
        "target",
        "source",
        "neighbor",
        "speaker",
        "gap",
        "language",
        "context",
        "model",
        "custom_prompt",
        "client",
    ],
)
def test_real_fidelity_memo_invalidates_changed_inputs(monkeypatch, change):
    t = make_translator()
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps({"valid": True, "issues": []}))
                )
            ]
        )

    monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake)
    source = [SubtitleProcessData(index=1, original_text="The door is open.")]
    repaired = [replace(source[0], translated_text="门开着")]
    t._all_source_by_index = {1: source[0].original_text, 2: "We can enter."}
    t._validate_chinese_window_fidelity(source, repaired)
    t._validate_chinese_window_fidelity(source, repaired)
    assert len(calls) == 1
    if change == "target":
        repaired[0] = replace(repaired[0], translated_text="门是开着的")
    elif change == "source":
        source[0] = replace(source[0], original_text="The main door is open.")
    elif change == "neighbor":
        t._all_source_by_index[2] = "We cannot enter."
    elif change == "speaker":
        t._all_speaker_by_index[1] = "B"
    elif change == "gap":
        t._gap_after_index[1] = 800
    elif change == "language":
        t._all_language_by_index[1] = "ja"
    elif change == "context":
        t.translation_context = TranslationContext(terminology="door -> 大门")
    elif change == "model":
        t.model = "other"
    elif change == "custom_prompt":
        t.custom_prompt = "Different style"
    elif change == "client":
        t.llm_client = object()
    t._validate_chinese_window_fidelity(source, repaired)
    assert len(calls) == 2
    t.stop()
    with pytest.raises(RuntimeError, match="cancelled"):
        t._validate_chinese_window_fidelity(source, repaired)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "verdict", [{"valid": False, "issues": ["missing fact"]}, {"valid": "true", "issues": []}]
)
def test_fidelity_rejection_and_invalid_response_are_not_memoized(monkeypatch, verdict):
    calls = []

    def fake(**kwargs):
        calls.append(1)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(verdict)))]
        )

    monkeypatch.setattr("subforge.core.translate.llm_translator.call_llm", fake)
    t = make_translator()
    source = [SubtitleProcessData(index=1, original_text="The door is open.")]
    repaired = [replace(source[0], translated_text="门开着")]
    for _ in range(2):
        with pytest.raises(ValueError):
            t._validate_chinese_window_fidelity(source, repaired)
    assert len(calls) == 2
    t.stop()
