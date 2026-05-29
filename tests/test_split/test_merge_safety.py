"""Tests for _merge_segments_based_on_sentences safety guarantees.

Verifies that unmatched, gap, and trailing ASR segments are always
preserved in the result — never silently dropped.
"""

from unittest.mock import MagicMock

from subforge.core.asr.asr_data import ASRDataSeg
from subforge.core.split.split import SubtitleSplitter


def _make_splitter():
    splitter = SubtitleSplitter.__new__(SubtitleSplitter)
    splitter.max_word_count_cjk = 18
    splitter.max_word_count_english = 12
    return splitter


def test_unmatched_sentence_preserves_asr_segment():
    """未匹配的句子对应的ASR片段应以原样保留。"""
    splitter = _make_splitter()

    segments = [
        ASRDataSeg("hello world", 0, 1000),
        ASRDataSeg("foo bar", 1000, 2000),
        ASRDataSeg("baz qux", 2000, 3000),
    ]

    # 只有第一个句子能匹配,第二个完全无法匹配
    sentences = ["hello world", "zzzzzzzzzzzz"]

    result = splitter._merge_segments_based_on_sentences(
        segments, sentences, max_unmatched=10
    )

    result_texts = [seg.text for seg in result]
    # "foo bar" 未匹配但应保留
    assert "foo bar" in result_texts, f"'foo bar' was dropped! Got: {result_texts}"
    # "baz qux" 是尾部片段,也应保留
    assert "baz qux" in result_texts, f"'baz qux' was dropped! Got: {result_texts}"


def test_gap_segments_between_matches_are_preserved():
    """两个匹配之间的间隙ASR片段应以原样保留。"""
    splitter = _make_splitter()

    segments = [
        ASRDataSeg("aaa bbb", 0, 1000),
        ASRDataSeg("ccc", 1000, 1500),
        ASRDataSeg("ddd", 1500, 2000),
        ASRDataSeg("eee fff", 2000, 3000),
    ]

    # 第一个句子匹配 seg0,第二个句子匹配 seg3,seg1-2 是间隙
    sentences = ["aaa bbb", "eee fff"]

    result = splitter._merge_segments_based_on_sentences(
        segments, sentences, max_unmatched=10
    )

    result_texts = [seg.text for seg in result]
    assert "ccc" in result_texts, f"Gap segment 'ccc' was dropped! Got: {result_texts}"
    assert "ddd" in result_texts, f"Gap segment 'ddd' was dropped! Got: {result_texts}"


def test_trailing_segments_after_last_match_are_preserved():
    """最后一个匹配之后的尾部ASR片段应以原样保留。"""
    splitter = _make_splitter()

    segments = [
        ASRDataSeg("hello world", 0, 1000),
        ASRDataSeg("trailing one", 1000, 2000),
        ASRDataSeg("trailing two", 2000, 3000),
    ]

    sentences = ["hello world"]

    result = splitter._merge_segments_based_on_sentences(
        segments, sentences, max_unmatched=10
    )

    result_texts = [seg.text for seg in result]
    assert "trailing one" in result_texts, f"Trailing segment was dropped! Got: {result_texts}"
    assert "trailing two" in result_texts, f"Trailing segment was dropped! Got: {result_texts}"


def test_all_sentences_unmatched_preserves_all_segments():
    """所有句子都未匹配时,所有ASR片段应以原样保留。"""
    splitter = _make_splitter()

    segments = [
        ASRDataSeg("aaa", 0, 1000),
        ASRDataSeg("bbb", 1000, 2000),
        ASRDataSeg("ccc", 2000, 3000),
    ]

    sentences = ["zzzzzzzz", "yyyyyyyy"]

    result = splitter._merge_segments_based_on_sentences(
        segments, sentences, max_unmatched=10
    )

    result_texts = [seg.text for seg in result]
    assert len(result) >= 3, f"Expected >= 3 segments, got {len(result)}: {result_texts}"
    assert "aaa" in result_texts
    assert "bbb" in result_texts
    assert "ccc" in result_texts


def test_empty_sentences_returns_all_segments():
    """LLM返回空句子列表时,所有ASR片段应以原样保留。"""
    splitter = _make_splitter()

    segments = [
        ASRDataSeg("hello", 0, 1000),
        ASRDataSeg("world", 1000, 2000),
    ]

    sentences = []

    result = splitter._merge_segments_based_on_sentences(
        segments, sentences, max_unmatched=10
    )

    result_texts = [seg.text for seg in result]
    assert "hello" in result_texts
    assert "world" in result_texts


def test_timestamps_preserved_in_fallback():
    """fallback片段应保留原始时间戳。"""
    splitter = _make_splitter()

    segments = [
        ASRDataSeg("matched", 0, 1000),
        ASRDataSeg("fallback", 5000, 6000),
    ]

    sentences = ["matched"]

    result = splitter._merge_segments_based_on_sentences(
        segments, sentences, max_unmatched=10
    )

    fallback_seg = next((s for s in result if s.text == "fallback"), None)
    assert fallback_seg is not None, f"'fallback' was dropped! Got: {[s.text for s in result]}"
    assert fallback_seg.start_time == 5000
    assert fallback_seg.end_time == 6000
