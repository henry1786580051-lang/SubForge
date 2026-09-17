import pytest

from subforge.core.translate.quality.semantic_actions import detect_semantic_action_mismatch


@pytest.mark.parametrize('source,bad,good,previous,next_source,rule', [
    ('This is the Limited trim.', '这是限量版。', '这是 Limited 配置。', '', '', 'trim_name'),
    ('It is not placeful.', '它不够宽敞。', '它不太好判断车身位置。', '', '', 'vehicle_placement'),
    ('It is get-out-of-the-way-able.', '能让别人让路。', '能迅速避让。', '', '', 'movement_agent'),
    ('It saves you from people doing that.', '救的是那些人。', '能保护你免受那些人的影响。', '', '', 'protection_roles'),
    ('No leakages.', '看看漏不漏水。', '没有漏水。', '', '', 'observed_no_leak'),
    ('We probably had 100 on it still, maybe 80.', '大概还剩油100，也许80。',
     '当时大概还剩100英里续航，也许80英里。', 'We paid for 200 miles of range.', '', 'remaining_range'),
    ('We pulled this off our Level', '我们用三级快充充完。', '我们用的充电桩是', '',
     '2 charger at home.', 'charging_level'),
])
def test_source_anchored_defects_flagged_but_correct_meaning_accepted(source, bad, good, previous, next_source, rule):
    kwargs = {'previous_source': previous, 'next_source': next_source}
    signal = detect_semantic_action_mismatch(source, bad, **kwargs)
    assert signal and signal.rule_id.endswith('.' + rule)
    assert detect_semantic_action_mismatch(source, good, **kwargs) is None


@pytest.mark.parametrize('source,target,previous,next_source', [
    ('This is a limited edition.', '这是限量版。', '', ''),
    ('The cabin is not spacious.', '车内不够宽敞。', '', ''),
    ('Make other people get out of the way.', '让别人让路。', '', ''),
    ('It saves those people.', '救的是那些人。', '', ''),
    ('Check for leakages.', '看看漏不漏水。', '', ''),
    ('We had fuel in the tank still.', '当时还剩油。', '200 miles of range.', ''),
    ('We had 100 on it still.', '当时还剩油。', '', ''),
    ('We use a Level 3 charger.', '使用三级快充。', 'Level 2 charging is slow.', ''),
    ('Our Level 2 charger is slower than Level 3.', '二级充电比三级慢。', '', ''),
])
def test_legitimate_comparisons_and_missing_context_not_flagged(source, target, previous, next_source):
    assert detect_semantic_action_mismatch(source, target, previous_source=previous, next_source=next_source) is None
