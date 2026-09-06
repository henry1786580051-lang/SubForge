import pytest

from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.quality.preservation import inspect_preserved_tokens
from subforge.core.translate.types import TargetLanguage


def _inspect(source, target, language="简体中文"):
    return inspect_preserved_tokens(
        {"21": target},
        {"21": source},
        str,
        target_language_value=language,
        localized_magnitude_rendered=LLMTranslator._localized_magnitude_rendered,
    )


@pytest.mark.parametrize(
    "source,target,language",
    [
        ("It offers a 7-speed wet clutch DCT.", "配备7速湿式双离合变速箱", "简体中文"),
        ("It offers a 7-speed wet clutch DCT.", "配备七速湿式双离合变速箱", "简体中文"),
        ("This DCT car is quiet.", "这辆双离合车型很安静", "简体中文"),
        ("The DCT gearbox is optional.", "可选雙離合變速箱", "繁体中文"),
        ("The dry-clutch DCT is available.", "有雙離合可揀", "粤语"),
    ],
)
def test_local_vehicle_context_accepts_equivalent_transmission_name(source, target, language):
    assert not _inspect(source, target, language)


@pytest.mark.parametrize(
    "source,target",
    [
        ("Compute the DCT coefficients.", "计算双离合系数"),
        ("DCT", "双离合"),
        ("This DCT car is quiet.", "这辆车很安静"),
        ("It offers a 7-speed wet clutch DCT.", "配备8速湿式双离合变速箱"),
        ("It offers a 7-speed wet clutch DCT.", "配备17速湿式双离合变速箱"),
        ("It offers a 7-speed wet clutch DCT.", "配备十七速湿式双离合变速箱"),
        ("It offers a 7-speed wet clutch DCT.", "配备湿式双离合变速箱"),
        ("The ABC gearbox is optional.", "可选双离合变速箱"),
    ],
)
def test_ambiguous_acronyms_missing_facts_and_changed_numbers_stay_rejected(source, target):
    assert _inspect(source, target)


def test_context_must_belong_to_the_current_cue():
    diagnostics = inspect_preserved_tokens(
        {"21": "这辆双离合车很安静", "22": "计算双离合系数"},
        {"21": "This DCT car is quiet.", "22": "Compute the DCT coefficients."},
        str,
        target_language_value="简体中文",
        localized_magnitude_rendered=LLMTranslator._localized_magnitude_rendered,
    )
    assert len(diagnostics) == 1
    assert diagnostics[0].rule_id == "entity.identifier_missing"
    assert diagnostics[0].cue_keys == (22,)


def test_non_chinese_target_does_not_gain_a_chinese_alias():
    assert _inspect("This DCT car is quiet.", "这辆双离合车很安静", "English")


@pytest.mark.parametrize("reflect", [False, True])
def test_standard_and_reflect_response_acceptance_use_same_localization(reflect):
    translator = LLMTranslator(
        model="test",
        is_reflect=reflect,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        thread_num=1,
        batch_num=1,
        custom_prompt="",
        update_callback=None,
    )
    source = {"21": "The DCT gearbox is optional."}
    text = "可选双离合变速箱"
    response = {
        "21": {
            "initial_translation": text,
            "reflection": "",
            "native_translation": text,
        }
        if reflect
        else text
    }
    valid, message = translator._validate_llm_response(response, source)
    assert valid, message
