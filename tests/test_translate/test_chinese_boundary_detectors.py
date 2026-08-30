from dataclasses import FrozenInstanceError

import pytest

from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.quality import (
    ORDERED_CHINESE_BOUNDARY_DETECTORS,
    ChineseBoundaryFeatures,
    detect_adverb_pronoun_attachment_boundary,
    detect_clause_attachment_boundary,
    detect_completion_frame_boundary,
    detect_consequence_predicate_boundary,
    detect_discourse_bridge_boundary,
    detect_foundation_boundary,
    detect_governing_attachment_boundary,
    detect_incomplete_nominal_frame_boundary,
    detect_late_structural_frame_boundary,
    detect_nominal_attachment_boundary,
    detect_numeric_completion_boundary,
    detect_predicate_completion_boundary,
    detect_reason_construction_boundary,
    detect_semantic_attachment_boundary,
    detect_semantic_completion_boundary,
    detect_structural_tail_boundary,
    detect_subject_attachment_boundary,
    detect_subject_nominal_completion_boundary,
    detect_surface_fluency_boundary,
    detect_temporal_locative_attachment_boundary,
    detect_terminal_token_boundary,
    detect_unfinished_frame_boundary,
    detect_unfinished_predicate_boundary,
    detect_visible_pause_boundary,
)


def test_chinese_boundary_features_are_immutable_and_normalize_shared_forms() -> None:
    features = ChineseBoundaryFeatures.from_text(
        "  在隧道内部，你知道  ",
        "你知道吗 会安装支撑结构。",
        gap_ms=320,
    )

    assert features.left == "在隧道内部"
    assert features.right == "会安装支撑结构"
    assert features.compact_left == "在隧道内部"
    assert features.canonical_right == "会安装支撑结构"
    assert features.left_has_terminal_punctuation is False
    assert features.gap_ms == 320
    with pytest.raises(FrozenInstanceError):
        features.gap_ms = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("left", "right", "gap_ms", "expected"),
    [
        ("投资达到12", "亿美元", 300, "number and unit are separated by a visible pause"),
        ("项目规模高达", "五十万平方米", 300, "unfinished predicate or modifier crosses a visible pause"),
        ("投资达到12", "亿美元", 299, None),
        ("项目已经完成", "随后正式开放", 500, None),
    ],
)
def test_visible_pause_detector_preserves_threshold_and_precedence(
    left: str,
    right: str,
    gap_ms: int,
    expected: str | None,
) -> None:
    match = detect_visible_pause_boundary(
        ChineseBoundaryFeatures.from_text(left, right, gap_ms=gap_ms),
        separated_gap_ms=300,
    )

    assert (match.message if match is not None else None) == expected


def test_llm_translator_visible_pause_adapter_preserves_legacy_contract() -> None:
    translator = object.__new__(LLMTranslator)
    translator._gap_after_index = {7: 300}

    assert translator._long_gap_chinese_boundary_signal(7, "投资达到12", "亿美元") == (
        "number and unit are separated by a visible pause"
    )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("所以", "项目继续推进", "standalone connective"),
        ("这项工程", "将于明年动工", "demonstrative subject is stranded"),
        ("这项工作总体上", "已经完成", "sentence adverb is separated from its predicate"),
        (
            "我们目前",
            "仍在评估方案",
            "subject and sentence adverb are separated from their predicate",
        ),
        (
            "我与项目团队",
            "将在明年动工",
            "coordinated subject is separated from its predicate",
        ),
        ("项目已经完成。", "随后正式开放", None),
    ],
)
def test_foundation_detector_preserves_legacy_precedence(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_foundation_boundary(ChineseBoundaryFeatures.from_text(left, right))

    assert (match.message if match is not None else None) == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("所以", "项目继续推进"),
        ("这项工程", "将于明年动工"),
        ("这项工作总体上", "已经完成"),
        ("我们目前", "仍在评估方案"),
        ("我与项目团队", "将在明年动工"),
    ],
)
def test_llm_translator_foundation_adapter_preserves_detector_contract(
    left: str,
    right: str,
) -> None:
    match = detect_foundation_boundary(ChineseBoundaryFeatures.from_text(left, right))

    assert match is not None
    assert LLMTranslator._chinese_boundary_signal(left, right) == match.message


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            "我们看到的",
            "昨天拍摄的照片",
            "relative clause is separated from its head noun",
        ),
        (
            "这就是答案",
            "我昨天提到的那个",
            "demonstrative relative clause lacks its head noun",
        ),
        (
            "这会给乘客更好的",
            "体验",
            "comparative object is omitted after a governing verb",
        ),
        ("整个座舱看起来像", "一间休息室", "comparison frame is separated from its object"),
        (
            "这是一辆非常特别的",
            "Honda Accord",
            "vehicle modifier is separated from its model name",
        ),
        (
            "我一直觉得",
            "这辆车很特别",
            "reporting predicate is separated from its complement",
        ),
        ("它算是一个", "例外", "classifier phrase is stranded"),
        (
            "这里有一个非常特别的",
            "类似旧款的设计",
            "demonstrative modifier is separated from its head noun",
        ),
        (
            "这里已经建成了12座",
            "车站",
            "count classifier lacks its contextual head noun",
        ),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_nominal_attachment_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_nominal_attachment_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            "我们把这座桥",
            "建成新的地标",
            "ba construction is separated from its predicate",
        ),
        (
            "将隧道主体",
            "推进到下一阶段",
            "disposal construction is separated from its predicate",
        ),
        (
            "设备将安装在",
            "隧道内部",
            "predicate is separated from its required complement",
        ),
        (
            "在隧道内部",
            "将安装支撑结构",
            "locative phrase is separated from its predicate",
        ),
        (
            "施工期间必须封路",
            "在大型设备进入现场时",
            "standalone temporal phrase is separated from its governing clause",
        ),
        (
            "把设备安装在",
            "隧道内部",
            "predicate is separated from its required complement",
        ),
        ("将由桥墩来承受", "全部荷载", None),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_governing_attachment_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_governing_attachment_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("桥墩所受的", "荷载所受到的影响", "possible duplicated boundary phrase"),
        (
            "这是全球首次",
            "采用这种施工工艺",
            "superlative modifier is separated from its predicate",
        ),
        (
            "这是非常困难的工程内容",
            "必须谨慎施工",
            "literal Japanese difficulty construction",
        ),
        (
            "这个项目就是如此复杂的工程",
            "团队仍在推进",
            "duplicated construction nominalization",
        ),
        ("所以但是我们仍要继续", "项目不会暂停", "stacked discourse connectives"),
        ("这项工程需要继续推进", "继续推进到下一阶段", "possible duplicated boundary phrase"),
        (
            "这个大型机场项目即将正式完工",
            "该个大型机场项目即将正式完工",
            "possible duplicated boundary meaning",
        ),
        ("我们决定继续", "继续推进项目", "possible duplicated boundary phrase"),
        ("他们热爱阅读", "阅读改变生活", "possible duplicated boundary phrase"),
        ("这个决定让我会", "我会继续", "possible duplicated boundary phrase"),
        (
            "教育方面的问题仍然存在",
            "在教育方面仍需改革",
            "possible duplicated boundary meaning",
        ),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_surface_fluency_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_surface_fluency_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("女性读者", "也开始减少阅读", "material subject may be stranded"),
        ("这里有很多人", "也开始阅读", "material subject may be stranded"),
        ("这是非常重要的", "结构、流程的信息", "coordinated modifier may be stranded"),
        ("这个项目帮助学生", "也开始阅读", None),
        ("这是非常重要的。", "结构、流程的信息", None),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_subject_attachment_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_subject_attachment_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            "这个方案但是",
            "我们仍要继续",
            "connective stranded at previous subtitle end",
        ),
        ("真正的关键在于", "执行方式", "possible copular bridge"),
        ("真正的问题", "是我们缺少时间", "possible copular bridge"),
        ("真正的问题", "已经解决", None),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_discourse_bridge_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_discourse_bridge_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            "我们讨论尤其是对",
            "新项目的影响",
            "unfinished Chinese grammatical structure",
        ),
        ("我们仍觉得", "这个方案可行", "unfinished Chinese grammatical structure"),
        ("我看到在城市之间", "存在明显差异", "unfinished Chinese locative frame"),
        ("我们又把", "设备移到现场", "unfinished Chinese grammatical structure"),
        (
            "这辆车其实就是",
            "一台日常通勤工具",
            "copular frame is separated from its result",
        ),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_unfinished_frame_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_unfinished_frame_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            "也许蓝色地带之所以能引起强烈共鸣的部分原因",
            "来自现代生活与长寿之间的冲突",
            "unfinished Chinese reason construction",
        ),
        (
            "项目之所以延期的主要原因",
            "来自审批进度",
            "unfinished Chinese reason construction",
        ),
        ("这只是部分原因", "项目仍会继续", None),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_reason_construction_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_reason_construction_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


def test_discourse_bridge_still_precedes_reason_construction() -> None:
    left = "项目之所以延期的主要原因"
    right = "是我们缺少审批材料"

    reason_match = detect_reason_construction_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert reason_match is not None
    assert LLMTranslator._chinese_boundary_signal(left, right) == "possible copular bridge"


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            "我觉得你90%的使用场景里 这个转向机",
            "都是好的",
            "percentage use-case predicate is stranded",
        ),
        ("就像会让这车", "比现在更好", "resultative predicate is stranded"),
        ("这会让这辆很好的车", "继续保持优势", None),
        ("我要长途步行 带着一个", "小婴儿在身上", "classifier phrase is stranded"),
        (
            "他们预测会出现像",
            "唐纳德·特朗普这样的人",
            "comparison example is stranded",
        ),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_completion_frame_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_completion_frame_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("数量在零本", "到四本之间", "numeric range is split"),
        ("年龄从18岁", "至25岁", "numeric range is split"),
        ("数量在零本", "大约四本", None),
        (
            "我猜 至少会增长到",
            "我们开始运行时的1.5到2倍",
            "numeric complement is stranded",
        ),
        ("团队规模至少会增长到现在的1.5倍", "之后继续扩大", None),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_numeric_completion_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_numeric_completion_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            "事实上 这项研究非常成功 以至于三项诺贝尔奖",
            "它至今仍在使用",
            "consequence predicate is missing",
        ),
        (
            "项目推进顺利 因此两个成果",
            "团队继续工作",
            "consequence predicate is missing",
        ),
        ("项目推进顺利 因此产生了两个成果", "团队继续工作", None),
        ("因此成果显著", "团队继续工作", None),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_consequence_predicate_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_consequence_predicate_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            "这件事的重点更多是",
            "我们反复做出的选择",
            "semantic frame is incomplete",
        ),
        (
            "我认为我们看到",
            "小学到高中都发生了变化",
            "possible reporting frame",
        ),
        ("这需要一场真正的", "大规模转变", "nominal modifier is stranded"),
        ("问题已经解决", "团队继续工作", None),
        ("我认为这个方案可行", "团队继续工作", None),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_semantic_attachment_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_semantic_attachment_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            "团队希望能够",
            "继续推进项目",
            "unfinished Chinese predicate or governing word",
        ),
        ("项目旨在", "改善交通", "unfinished Chinese predicate or governing word"),
        (
            "团队已经完成尝试",
            "随后总结经验",
            "unfinished Chinese predicate or governing word",
        ),
        ("这是一项大胆尝试", "团队继续评估", None),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_unfinished_predicate_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_unfinished_predicate_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            "他一直开着",
            "一辆老款轿车",
            "aspect predicate is separated from its complement",
        ),
        (
            "机场每年接待",
            "超过五千万名旅客",
            "transitive predicate is split from its quantified object",
        ),
        ("他一直开着", "车辆已经停稳", None),
        ("机场每年接待旅客", "客流继续增长", None),
    ],
)
def test_predicate_completion_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_predicate_completion_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("项目将在明年启动", "随后", "standalone Chinese temporal fragment"),
        (
            "这套设计中",
            "还包含一条备用通道",
            "locative frame is separated from its complement",
        ),
        (
            "在现有基础上",
            "进一步提高效率",
            "locative phrase is separated from its predicate",
        ),
        (
            "距离市中心约3公里以外",
            "新场地已经动工",
            "distance modifier is separated from its noun",
        ),
        ("项目将在明年启动", "随后继续推进", None),
        ("距离市中心约3公里以外", "工程已经动工", None),
    ],
)
def test_temporal_locative_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_temporal_locative_attachment_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("大多数其他", "方案仍在评估", "comparative noun modifier is stranded"),
        ("我们会考虑任何", "可能出现的风险", "unfinished Chinese grammatical structure"),
        ("这并不是", "成本问题", "unfinished Chinese grammatical structure"),
        ("团队希望进一步", "改善现有流程", "unfinished Chinese grammatical structure"),
        ("团队选择某一种", "随后继续测试", None),
        ("这是其他方案", "团队继续测试", None),
    ],
)
def test_incomplete_nominal_frame_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_incomplete_nominal_frame_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("我真正想说的", "是成本并非重点", "semantic frame is incomplete"),
        ("不过我只是觉得吧", "还需要继续讨论", "vague filler-only frame"),
        (
            "我真正想表达的重点并非如此",
            "问题在于执行方式",
            "semantic frame is incomplete",
        ),
        ("它确实很适合", "日常通勤", "adjective complement is missing"),
        ("这篇文章约3万字", "主要讨论城市发展", "classifier phrase is stranded"),
        ("我真正想说的是成本问题", "团队继续讨论", None),
        ("它确实很适合日常通勤", "驾驶体验很好", None),
    ],
)
def test_semantic_completion_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_semantic_completion_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("而在数字时代", "阅读方式已经改变", "unfinished Chinese locative frame"),
        ("我们看到了", "新的变化", "possible reporting frame"),
        ("阅读", "和写作都很重要", "coordinated subject may be stranded"),
        ("这项计划最终", "成为城市地标", "predicate fragment starts at next subtitle"),
        ("我们保存", "这些资料", "transitive predicate is split from its object"),
        ("我们已经做出选择", "这些方案仍需测试", None),
        ("这套设备正在使用", "这些数据会被记录", None),
    ],
)
def test_clause_attachment_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_clause_attachment_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("所以我觉得如此", "阅读正在变化", "unfinished Chinese grammatical structure"),
        ("年轻学生们", "也开始阅读", "material subject may be stranded"),
        ("这成为一个", "重要的转折点", "classifier phrase is stranded"),
        ("年轻学生们正在阅读", "写作也很重要", None),
        ("他是第一个", "随后获得奖励", None),
    ],
)
def test_subject_nominal_completion_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_subject_nominal_completion_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("如今", "如今项目已经完工", "possible duplicated boundary phrase"),
        ("这并不", "像以前一样", "negated comparison is split from its complement"),
        ("我们为了", "提高效率", "possible function-word split"),
        ("团队正在", "推进项目", "unfinished Chinese grammatical structure"),
        ("在这种情况下 当", "项目开始运行", "unfinished Chinese grammatical structure"),
        ("从很多方面来说", "这个方案更合理", None),
        ("这是全球之最", "项目仍在运行", None),
    ],
)
def test_structural_tail_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_structural_tail_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("这个说法并不自然", "对项目来说太根本了", "literal fundamental calque"),
        ("我想买一个", "更适合通勤的车型", "unfinished Chinese grammatical structure"),
        ("年轻人身上", "也出现了这种变化", "unfinished Chinese locative subject"),
        ("问题发生在年轻人身上", "团队开始研究", None),
        ("这是另一个方面", "同样值得关注", None),
    ],
)
def test_late_structural_frame_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_late_structural_frame_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("设备已经在", "施工现场", "unfinished Chinese grammatical structure"),
        ("项目仍然", "需要继续审批", "unfinished Chinese adverbial predicate"),
        ("价格一路", "上涨到新高", "unfinished Chinese degree phrase"),
        ("它不像过去", "那样复杂", "comparison phrase is stranded"),
        ("问题是他们", "尚未提交文件", "possible pronoun boundary"),
        ("我们", "继续推进项目", "standalone subject is separated from its predicate"),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_adverb_pronoun_attachment_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_adverb_pronoun_attachment_boundary(
        ChineseBoundaryFeatures.from_text(left, right)
    )

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("我喜欢这个", "新的设计", "possible demonstrative split"),
        ("项目已经完成", "了", "particle stranded at next subtitle start"),
        ("项目已经完成", "时间一长就会变化", "unfinished Chinese grammatical structure"),
        ("项目已经完成", "随后正式开放", None),
    ],
)
def test_terminal_token_detector_and_adapter_preserve_legacy_contract(
    left: str,
    right: str,
    expected: str | None,
) -> None:
    match = detect_terminal_token_boundary(ChineseBoundaryFeatures.from_text(left, right))

    assert (match.message if match is not None else None) == expected
    assert LLMTranslator._chinese_boundary_signal(left, right) == (expected or "")


def test_terminal_token_detector_preserves_shadowed_duplicate_connector_rule() -> None:
    features = ChineseBoundaryFeatures.from_text("项目延期所以", "所以团队调整计划")

    match = detect_terminal_token_boundary(features)

    assert match is not None
    assert match.message == "duplicated boundary connective"
    assert (
        LLMTranslator._chinese_boundary_signal(features.left, features.right)
        == "connective stranded at previous subtitle end"
    )
def test_ordered_chinese_boundary_detector_registry_preserves_legacy_precedence() -> None:
    assert tuple(detector.__name__ for detector in ORDERED_CHINESE_BOUNDARY_DETECTORS) == (
        "detect_foundation_boundary",
        "detect_nominal_attachment_boundary",
        "detect_governing_attachment_boundary",
        "detect_surface_fluency_boundary",
        "detect_subject_attachment_boundary",
        "detect_discourse_bridge_boundary",
        "detect_unfinished_frame_boundary",
        "detect_reason_construction_boundary",
        "detect_completion_frame_boundary",
        "detect_numeric_completion_boundary",
        "detect_consequence_predicate_boundary",
        "detect_semantic_attachment_boundary",
        "detect_unfinished_predicate_boundary",
        "detect_predicate_completion_boundary",
        "detect_temporal_locative_attachment_boundary",
        "detect_incomplete_nominal_frame_boundary",
        "detect_semantic_completion_boundary",
        "detect_clause_attachment_boundary",
        "detect_subject_nominal_completion_boundary",
        "detect_structural_tail_boundary",
        "detect_late_structural_frame_boundary",
        "detect_adverb_pronoun_attachment_boundary",
        "detect_terminal_token_boundary",
    )

