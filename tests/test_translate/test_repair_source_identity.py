"""A one-cue repair must keep its document key during ownership validation."""

import ast
import inspect
from types import SimpleNamespace

import pytest

from subforge.core.translate import llm_translator as engine
from subforge.core.translate.context import TranslationContext
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.types import TargetLanguage


@pytest.fixture
def translator():
    instance = LLMTranslator(
        thread_num=1, batch_num=1, target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="test", custom_prompt="", is_reflect=False, update_callback=None,
        translation_context=TranslationContext(
            terminology="- Round Design -> Rayon Design (probable ASR correction)"
        ),
    )
    yield instance
    instance.stop()


@pytest.mark.parametrize("multispeaker", [False, True])
@pytest.mark.parametrize("index", [5, 95, 243])
def test_local_repair_uses_its_own_key_not_first_cue(translator, monkeypatch, multispeaker, index):
    source = "Round Design can vectorise anything you need."
    translator._all_source_by_index = {
        1: "Merdeka stands tall.", index: source, index + 1: "I use Rayon Design."
    }
    translator._all_speaker_by_index = {1: "S1", index: "S2" if multispeaker else "S1"}
    target = "Rayon Design可以把你需要的任何东西矢量化"
    calls = []

    def respond(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=target))])

    monkeypatch.setattr(engine, "call_llm", respond)
    result = translator._translate_alignment_item(source, source_key=str(index), allow_reasoning=False)
    assert result == target
    assert len(calls) == 1
    assert calls[0]["reasoning_mode"] == "disabled"
    # The old synthetic key provably inspected the first cue, not this cue.
    valid, error = translator._validate_no_unowned_latin_names(
        {"1": target}, {"1": "Rayon Design can vectorise anything you need."}, str
    )
    assert not valid
    assert "Rayon" in error


def test_source_key_fix_does_not_license_first_cue_name_in_later_cue(translator, monkeypatch):
    source = "Round Design can vectorise anything you need."
    translator._all_source_by_index = {1: "Merdeka stands tall.", 95: source, 96: "I use Rayon Design."}
    monkeypatch.setattr(engine, "call_llm", lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Rayon Design和Merdeka都能矢量化所需内容"))]
    ))
    with pytest.raises(ValueError, match="Merdeka"):
        translator._translate_alignment_item(source, source_key="95", allow_reasoning=False)


def test_every_document_repair_call_passes_explicit_source_identity():
    tree = ast.parse(inspect.getsource(engine))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "_translate_alignment_item"]
    assert calls
    assert all(any(keyword.arg == "source_key" for keyword in call.keywords) for call in calls)
