"""字幕重断句模块

翻译后的字幕可能包含过长的段落，需要重新断句以符合字幕显示要求。
每条字幕只有一行中文 + 一行英文，如果太长就拆分成多条字幕。
"""

import re
from typing import List

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.utils.logger import setup_logger

logger = setup_logger("subtitle_resegment")

# 默认字符限制
DEFAULT_MAX_CHARS_EN = 50  # 英文每行最大字符数（含空格）
DEFAULT_MAX_CHARS_CJK = 20  # CJK每行最大字符数


def _is_mainly_cjk(text: str) -> bool:
    """判断文本是否主要是CJK字符"""
    if not text:
        return False
    cjk_count = sum(1 for c in text if '一' <= c <= '鿿' or '぀' <= c <= 'ヿ' or '가' <= c <= '힯')
    return cjk_count > len(text) * 0.3


def resegment_subtitles(
    asr_data: ASRData,
    max_chars_en: int = DEFAULT_MAX_CHARS_EN,
    max_chars_cjk: int = DEFAULT_MAX_CHARS_CJK,
) -> ASRData:
    """对翻译后的字幕进行重断句

    每条字幕只有一行中文 + 一行英文。
    如果文本超过字符限制，就拆分成多条字幕。

    Args:
        asr_data: 包含翻译文本的ASR数据
        max_chars_en: 英文每行最大字符数
        max_chars_cjk: CJK每行最大字符数

    Returns:
        重断句后的ASR数据
    """
    new_segments = []

    for seg in asr_data.segments:
        # 处理原文和译文
        text1 = seg.text.strip()
        text2 = seg.translated_text.strip() if seg.translated_text else ""

        if not text1 and not text2:
            continue

        # 检测语言并分配正确的字符限制
        if _is_mainly_cjk(text1):
            # text 是中文，translated_text 是英文
            zh_text, en_text = text1, text2
            zh_field, en_field = 'text', 'translated_text'
        else:
            # text 是英文，translated_text 是中文
            en_text, zh_text = text1, text2
            en_field, zh_field = 'text', 'translated_text'

        # 判断是否需要拆分
        en_needs_split = len(en_text) > max_chars_en if en_text else False
        zh_needs_split = len(zh_text) > max_chars_cjk if zh_text else False

        if not en_needs_split and not zh_needs_split:
            # 不需要拆分，直接保留
            new_segments.append(seg)
        else:
            # 需要拆分
            split_segments = _split_segment(
                seg, en_text, zh_text, max_chars_en, max_chars_cjk,
                en_field, zh_field
            )
            new_segments.extend(split_segments)

    logger.info(f"Resegmented: {len(asr_data.segments)} -> {len(new_segments)} segments")
    return ASRData(new_segments)


def _split_segment(
    seg: ASRDataSeg,
    en_text: str,
    zh_text: str,
    max_chars_en: int,
    max_chars_cjk: int,
    en_field: str = 'text',
    zh_field: str = 'translated_text',
) -> List[ASRDataSeg]:
    """拆分单个字幕段

    策略：
    1. 将中文和英文分别拆分成多行
    2. 配对中文和英文行
    3. 每对成为一个新的字幕段
    4. 均匀分配时间戳
    """
    # 将文本拆分成行（每行不超过字符限制）
    en_lines = _split_text_to_lines(en_text, max_chars_en) if en_text else [""]
    zh_lines = _split_text_to_lines(zh_text, max_chars_cjk) if zh_text else [""]

    # 确定拆分数量（取最大值）
    num_splits = max(len(en_lines), len(zh_lines))
    if num_splits <= 1:
        return [seg]

    # 均匀分配时间戳
    duration = seg.end_time - seg.start_time
    segment_duration = duration / num_splits

    result = []
    for i in range(num_splits):
        start = int(seg.start_time + i * segment_duration)
        end = int(seg.start_time + (i + 1) * segment_duration)

        en_line = en_lines[i] if i < len(en_lines) else ""
        zh_line = zh_lines[i] if i < len(zh_lines) else ""

        # 根据字段名分配文本
        new_seg = ASRDataSeg(
            text=en_line if en_field == 'text' else zh_line,
            start_time=start,
            end_time=end,
            translated_text=zh_line if zh_field == 'translated_text' else en_line,
            speaker_id=seg.speaker_id,
        )
        result.append(new_seg)

    return result


def _split_text_to_lines(text: str, max_chars: int) -> List[str]:
    """将文本拆分成多行，每行不超过max_chars

    优先在句子边界、标点处拆分
    """
    if not text:
        return [""]

    if len(text) <= max_chars:
        return [text]

    # 按句子拆分
    sentences = _split_into_sentences(text)

    lines = []
    current_line = ""

    for sentence in sentences:
        # 如果当前行加上新句子不超过限制
        if current_line and len(current_line) + len(sentence) + 1 <= max_chars:
            current_line += " " + sentence if current_line else sentence
        elif not current_line:
            # 当前行为空，检查句子是否太长
            if len(sentence) <= max_chars:
                current_line = sentence
            else:
                # 句子太长，需要进一步拆分
                sub_lines = _split_long_sentence(sentence, max_chars)
                lines.extend(sub_lines)
                current_line = ""
        else:
            # 当前行已满，开始新行
            lines.append(current_line)
            if len(sentence) <= max_chars:
                current_line = sentence
            else:
                sub_lines = _split_long_sentence(sentence, max_chars)
                lines.extend(sub_lines)
                current_line = ""

    if current_line:
        lines.append(current_line)

    return lines if lines else [""]


def _split_into_sentences(text: str) -> List[str]:
    """按句子边界拆分文本

    注意：不拆分小数点（如2.4）、缩写（如Dr.）等
    """
    # 先保护小数点和常见缩写
    # 将 2.4 替换为 2￭4，Dr. 替换为 Dr￭
    protected = re.sub(r'(\d)\.(\d)', r'\1￭\2', text)
    protected = re.sub(r'(\b[A-Za-z]{1,4})\.', r'\1￭', protected)

    # 英文句子结束符 + 中文句子结束符 + 分号冒号等
    pattern = r'(?<=[.!?。！？；;:\n])\s*'
    sentences = re.split(pattern, protected)

    # 恢复保护的字符
    result = []
    for s in sentences:
        s = s.replace('￭', '.').strip()
        if s:
            result.append(s)

    return result


def _split_long_sentence(sentence: str, max_chars: int) -> List[str]:
    """拆分过长的句子

    优先在标点、空格处拆分，避免在关键参数处断句
    """
    if len(sentence) <= max_chars:
        return [sentence]

    lines = []
    remaining = sentence

    while remaining:
        if len(remaining) <= max_chars:
            lines.append(remaining)
            break

        # 在max_chars范围内寻找最佳断点
        break_pos = _find_best_break_point(remaining, max_chars)

        lines.append(remaining[:break_pos].strip())
        remaining = remaining[break_pos:].strip()

    return lines if lines else [sentence]


def _find_best_break_point(text: str, max_chars: int) -> int:
    """在文本中找到最佳断点

    优先级：
    1. 标点符号后（但不是小数点）
    2. 空格处
    3. CJK/字母边界
    4. 数字/字母边界（但不在数字中间）
    """
    if len(text) <= max_chars:
        return len(text)

    # 优先在标点处断（但排除小数点）
    puncts = ['，', '。', '；', '：', '！', '？', '、']
    for punct in puncts:
        pos = text.rfind(punct, 0, max_chars + 1)
        if pos > max_chars * 0.3:
            return pos + 1

    # 英文标点（需要检查前后字符，排除小数点）
    en_puncts = [',', '.', ';', ':', '!', '?']
    for punct in en_puncts:
        pos = text.rfind(punct, 0, max_chars + 1)
        if pos > max_chars * 0.3:
            # 检查是否是小数点（前后都是数字）
            if punct == '.' and pos > 0 and pos < len(text) - 1:
                if text[pos - 1].isdigit() and text[pos + 1].isdigit():
                    continue  # 跳过小数点
            return pos + 1

    # 在空格处断
    space_pos = text.rfind(' ', 0, max_chars + 1)
    if space_pos > max_chars * 0.3:
        return space_pos

    # 向后搜索空格（允许稍微超过max_chars）
    next_space = text.find(' ', max_chars)
    if next_space != -1 and next_space - max_chars < max_chars * 0.2:
        return next_space

    # 在CJK/字母边界处断
    for i in range(min(max_chars, len(text) - 1), max(0, max_chars - 10), -1):
        if i < len(text) - 1:
            curr = text[i]
            next_char = text[i + 1]

            # CJK和字母之间
            if _is_cjk(curr) and not _is_cjk(next_char):
                return i + 1
            if not _is_cjk(curr) and _is_cjk(next_char):
                return i + 1

    # 在数字和非数字边界处断（但不在数字中间）
    for i in range(min(max_chars, len(text) - 1), max(0, max_chars - 10), -1):
        if i < len(text) - 1:
            curr = text[i]
            next_char = text[i + 1]

            # 数字和CJK文字之间
            if curr.isdigit() and _is_cjk(next_char):
                return i + 1
            if _is_cjk(curr) and next_char.isdigit():
                return i + 1

    # 最后才在max_chars处断
    return max_chars


def _is_cjk(char: str) -> bool:
    """判断字符是否是CJK字符"""
    return '一' <= char <= '鿿' or '぀' <= char <= 'ヿ' or '가' <= char <= '힯'
