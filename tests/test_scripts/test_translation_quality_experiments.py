from copy import deepcopy

import pytest

from scripts.translation_quality.experiments import translation_experiments
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.types import TargetLanguage


@pytest.fixture
def translator():
    value = LLMTranslator(
        thread_num=1, batch_num=20, target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="glm-5.3-flash", custom_prompt="", is_reflect=False,
        update_callback=None, use_cache=False,
    )
    yield value
    value.stop()


@pytest.mark.parametrize("mode", ["single", "multi", "mixed"])
def test_exact_spacing_is_local_to_source_cue_and_does_not_mutate_data(translator, mode):
    sources = {"1": "That redcoat tavern.", "2": "A clear sunny sky."}
    response = {"1": "Red Coat Tavern餐厅", "2": "晴空万里"}
    frozen = deepcopy((sources, response))
    translator._all_source_by_index = {1: sources["1"], 2: sources["2"]}
    translator._all_speaker_by_index = {1: "S1", 2: "S2" if mode == "multi" else "S1"}
    translator._all_language_by_index = {1: "en", 2: "ja" if mode == "mixed" else "en"}
    def check():
        return translator._validate_no_unowned_latin_names(response, sources, str)
    assert not check()[0]
    with translation_experiments(("exact-name-spacing",)):
        assert check()[0]
        assert (sources, response) == frozen
        response["2"] = "Red Coat餐厅"
        assert not check()[0]
        response["2"] = "晴空万里"
        response["1"] = "Red Coat Tavern和Red Boat"
        assert not check()[0]
        response["1"] = frozen[1]["1"]
    assert not check()[0]


def test_experiments_restore_on_error_and_reject_unknown_name(translator):
    original = LLMTranslator._validate_no_unowned_latin_names
    with pytest.raises(RuntimeError), translation_experiments(("exact-name-spacing",)):
        raise RuntimeError("test failure")
    assert LLMTranslator._validate_no_unowned_latin_names is original
    with pytest.raises(ValueError, match="Unknown"):
        with translation_experiments(("unknown",)):
            pass


@pytest.mark.parametrize("target,valid", [
    ("我身高5英尺11英寸", True),
    ("我身高5英尺11英寸（约1米80）", False),
    ("我身高80英寸", False),
])
def test_numeric_feedback_does_not_relax_numeric_rejection(translator, target, valid):
    source = {"1": 'I am 5\'11".', "2": "It has 80 horsepower."}
    response = {"1": target, "2": "它有80马力"}
    before = translator._validate_cross_key_boundaries(response, source, str)
    with translation_experiments(("unowned-fact-feedback",)):
        after = translator._validate_cross_key_boundaries(response, source, str)
    assert before[0] == after[0] == valid
    if not valid:
        assert "unit conversions" in after[1]
        assert "['1:80']" in after[1]


def test_context_scope_changes_only_context_generation(monkeypatch):
    from subforge.core.translate import context

    calls = []
    monkeypatch.setattr(context, "call_llm", lambda **kw: calls.append(kw))
    prompt = [{"role": "system", "content": "You prepare context for professional subtitle translation."}]
    frozen = deepcopy(prompt)
    with translation_experiments(("scoped-terminology",)):
        context.call_llm(messages=prompt, reasoning_mode="disabled")
        context.call_llm(messages=[{"role": "system", "content": "Translate"}])
    assert prompt == frozen
    assert "occurrence-specific" in calls[0]["messages"][0]["content"]
    assert calls[0]["reasoning_mode"] == "disabled"
    assert calls[1]["messages"][0]["content"] == "Translate"

def test_context_role_contract_is_isolated_and_preserves_request_settings(monkeypatch):
    from subforge.core.translate import context

    calls = []
    def original(**kw):
        calls.append(kw)
    monkeypatch.setattr(context, "call_llm", original)
    prompt = [{"role": "system", "content": "You prepare context for professional subtitle translation."}]
    frozen = deepcopy(prompt)
    with translation_experiments(("context-role-contract",)):
        context.call_llm(messages=prompt, model="glm-5.3-flash", reasoning_mode="disabled",
                         max_output_tokens=4096)
        context.call_llm(messages=[{"role": "system", "content": "Translate"}])
    assert context.call_llm is original
    assert prompt == frozen
    assert len(calls) == 2
    assert "SEPARATE row" in calls[0]["messages"][0]["content"]
    assert "occurrence-specific" not in calls[0]["messages"][0]["content"]
    assert calls[0]["reasoning_mode"] == "disabled"
    assert calls[0]["model"] == "glm-5.3-flash"
    assert calls[0]["max_output_tokens"] == 4096
    assert calls[1]["messages"][0]["content"] == "Translate"


def test_domain_precision_is_scoped_and_restored(translator):
    from subforge.core.translate.guidance import target_language_style_rules
    generic = ["A quiet conversation about the weather."]
    vehicle = ["The turning circle is small; compare fuel economy in mpg."]
    baseline = translator._target_language_style_rules(vehicle)
    with translation_experiments(("domain-precision",)):
        assert translator._target_language_style_rules(generic) == target_language_style_rules("简体中文", generic)
        candidate = translator._target_language_style_rules(vehicle)
        assert "diameter from radius" in candidate
        assert "distance per fuel" in candidate
        assert "tendering roles" not in candidate
    assert translator._target_language_style_rules(vehicle) == baseline


def test_domain_precision_restores_after_failure(translator):
    original = translator._target_language_style_rules(["A bridge tower."])
    with pytest.raises(RuntimeError), translation_experiments(("domain-precision",)):
        assert "stay cable" in translator._target_language_style_rules(["A bridge tower."])
        raise RuntimeError("cancelled")
    assert translator._target_language_style_rules(["A bridge tower."]) == original
