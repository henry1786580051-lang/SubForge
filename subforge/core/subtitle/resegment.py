"""字幕重断句模块

翻译后的字幕可能包含过长的段落，需要重新断句以符合字幕显示要求。
每条字幕只有一行中文 + 一行英文，如果太长就拆分成多条字幕。

使用基于位置百分比的对齐方法，确保中英文对应关系正确。
"""

import re
from typing import List, Tuple

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.utils.logger import setup_logger

logger = setup_logger("subtitle_resegment")

# 默认字符限制
DEFAULT_MAX_CHARS_EN = 50  # 英文每行最大字符数（含空格）
DEFAULT_MAX_CHARS_CJK = 18  # CJK每行最大字符数


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
    中英文使用基于位置百分比的对齐方法。

    Args:
        asr_data: 包含翻译文本的ASR数据
        max_chars_en: 英文每行最大字符数
        max_chars_cjk: CJK每行最大字符数

    Returns:
        重断句后的ASR数据
    """
    new_segments = []

    for seg in asr_data.segments:
        text1 = seg.text.strip()
        text2 = seg.translated_text.strip() if seg.translated_text else ""

        if not text1 and not text2:
            continue

        # 检测语言
        if _is_mainly_cjk(text1):
            zh_text, en_text = text1, text2
            zh_field, en_field = 'text', 'translated_text'
        else:
            en_text, zh_text = text1, text2
            en_field, zh_field = 'text', 'translated_text'

        # 判断是否需要拆分
        en_needs_split = len(en_text) > max_chars_en if en_text else False
        zh_needs_split = len(zh_text) > max_chars_cjk if zh_text else False

        if not en_needs_split and not zh_needs_split:
            new_segments.append(seg)
        else:
            split_segments = _split_segment_by_position(
                seg, en_text, zh_text, max_chars_en, max_chars_cjk,
                en_field, zh_field
            )
            new_segments.extend(split_segments)

    logger.info(f"Resegmented: {len(asr_data.segments)} -> {len(new_segments)} segments")
    return ASRData(new_segments)


def _split_segment_by_position(
    seg: ASRDataSeg,
    en_text: str,
    zh_text: str,
    max_chars_en: int,
    max_chars_cjk: int,
    en_field: str = 'text',
    zh_field: str = 'translated_text',
) -> List[ASRDataSeg]:
    """基于位置百分比拆分字幕段

    策略：
    1. 将中文和英文分别拆分成行
    2. 根据每行在原文中的位置百分比来配对
    3. 这样即使句子数量不同，也能正确对齐
    """
    # 将中文和英文分别拆分成行
    zh_lines = _split_text_to_lines(zh_text, max_chars_cjk)
    en_lines = _split_text_to_lines(en_text, max_chars_en)

    if len(zh_lines) <= 1 and len(en_lines) <= 1:
        return [seg]

    # 计算每行在原文中的位置百分比
    zh_positions = _calculate_positions(zh_lines, zh_text)
    en_positions = _calculate_positions(en_lines, en_text)

    # 基于位置百分比配对
    pairs = _align_by_position(zh_lines, en_lines, zh_positions, en_positions)

    if len(pairs) <= 1:
        return [seg]

    # 均匀分配时间戳
    duration = seg.end_time - seg.start_time
    segment_duration = duration / len(pairs)

    result = []
    for i, (en_line, zh_line) in enumerate(pairs):
        start = int(seg.start_time + i * segment_duration)
        end = int(seg.start_time + (i + 1) * segment_duration)

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
        if current_line and len(current_line) + len(sentence) + 1 <= max_chars:
            current_line += " " + sentence
        elif not current_line:
            if len(sentence) <= max_chars:
                current_line = sentence
            else:
                sub_lines = _split_long_text(sentence, max_chars)
                lines.extend(sub_lines)
                current_line = ""
        else:
            lines.append(current_line)
            if len(sentence) <= max_chars:
                current_line = sentence
            else:
                sub_lines = _split_long_text(sentence, max_chars)
                lines.extend(sub_lines)
                current_line = ""

    if current_line:
        lines.append(current_line)

    return lines if lines else [""]


def _is_cjk(char: str) -> bool:
    """判断字符是否是CJK字符"""
    return '一' <= char <= '鿿' or '぀' <= char <= 'ヿ' or '가' <= char <= '힯'


def _calculate_positions(lines: List[str], original_text: str) -> List[float]:
    """计算每行在原文中的位置百分比

    返回值：每行的起始位置在原文中的百分比（0.0 到 1.0）
    """
    if not original_text or not lines:
        return [0.0]

    positions = []
    search_start = 0

    for line in lines:
        # 找到这行在原文中的位置
        pos = original_text.find(line, search_start)
        if pos == -1:
            # 如果找不到，估算位置
            if positions:
                positions.append(positions[-1] + 0.1)
            else:
                positions.append(0.0)
        else:
            positions.append(pos / len(original_text))
            search_start = pos + len(line)

    return positions


def _align_by_position(
    zh_lines: List[str],
    en_lines: List[str],
    zh_positions: List[float],
    en_positions: List[float],
) -> List[Tuple[str, str]]:
    """基于位置百分比配对中英文行

    对于每个中文行，找到位置最接近的英文行
    """
    if not zh_lines:
        zh_lines = [""]
    if not en_lines:
        en_lines = [""]

    pairs = []
    used_en = set()

    for zh_idx, zh_line in enumerate(zh_lines):
        zh_pos = zh_positions[zh_idx] if zh_idx < len(zh_positions) else 0.0

        # 找到位置最接近且未使用的英文行
        best_en_idx = -1
        best_distance = float('inf')

        for j in range(len(en_lines)):
            if j in used_en:
                continue
            en_pos = en_positions[j] if j < len(en_positions) else 0.0
            distance = abs(en_pos - zh_pos)
            if distance < best_distance:
                best_distance = distance
                best_en_idx = j

        if best_en_idx >= 0:
            pairs.append((en_lines[best_en_idx], zh_line))
            used_en.add(best_en_idx)
        else:
            pairs.append((en_lines[-1], zh_line))

    # 处理剩余的英文行
    for j in range(len(en_lines)):
        if j not in used_en:
            pairs.append((en_lines[j], zh_lines[-1] if zh_lines else ""))

    return pairs


def _split_into_sentences(text: str) -> List[str]:
    """按句子边界拆分文本"""
    if not text:
        return [""]

    # 保护小数点和常见缩写
    protected = re.sub(r'(\d)\.(\d)', r'\1￭\2', text)
    protected = re.sub(r'(\b[A-Za-z]{1,4})\.', r'\1￭', protected)

    # 按句子结束符拆分
    pattern = r'(?<=[.!?。！？；;:\n])\s*'
    sentences = re.split(pattern, protected)

    result = []
    for s in sentences:
        s = s.replace('￭', '.').strip()
        if s:
            result.append(s)

    return result if result else [""]


def _split_long_text(text: str, max_chars: int) -> List[str]:
    """拆分过长的文本"""
    if len(text) <= max_chars:
        return [text]

    lines = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            lines.append(remaining)
            break

        break_pos = _find_best_break_point(remaining, max_chars)
        lines.append(remaining[:break_pos].strip())
        remaining = remaining[break_pos:].strip()

    return lines if lines else [text]


def _find_best_break_point(text: str, max_chars: int) -> int:
    """在文本中找到最佳断点"""
    if len(text) <= max_chars:
        return len(text)

    # 优先在标点处断（排除小数点和连字符）
    puncts = ['，', '。', '；', '：', '！', '？', '、']
    for punct in puncts:
        pos = text.rfind(punct, 0, max_chars + 1)
        if pos > max_chars * 0.3:
            return pos + 1

    # 英文标点（排除小数点和连字符）
    en_puncts = [',', '.', ';', ':', '!', '?']
    for punct in en_puncts:
        pos = text.rfind(punct, 0, max_chars + 1)
        if pos > max_chars * 0.3:
            if punct == '.' and pos > 0 and pos < len(text) - 1:
                if text[pos - 1].isdigit() and text[pos + 1].isdigit():
                    continue
            return pos + 1

    # 在空格处断
    space_pos = text.rfind(' ', 0, max_chars + 1)
    if space_pos > max_chars * 0.3:
        return space_pos

    # 向后搜索空格
    next_space = text.find(' ', max_chars)
    if next_space != -1 and next_space - max_chars < max_chars * 0.2:
        return next_space

    # 在CJK和非CJK字符边界处断（不在CJK字符中间）
    # 优先在CJK和空格之间断，其次在CJK和字母之间断
    # 避免在数字和CJK之间断（保持"2026款"、"2009年"完整）
    for i in range(min(max_chars, len(text) - 1), max(0, max_chars - 10), -1):
        if i < len(text) - 1:
            curr = text[i]
            next_char = text[i + 1]
            # CJK和空格之间
            if _is_cjk(curr) and next_char == ' ':
                return i + 1
            if curr == ' ' and _is_cjk(next_char):
                return i + 1

    # 其次在CJK和字母之间断
    for i in range(min(max_chars, len(text) - 1), max(0, max_chars - 10), -1):
        if i < len(text) - 1:
            curr = text[i]
            next_char = text[i + 1]
            # CJK和ASCII字母之间（不是数字）
            if _is_cjk(curr) and next_char.isascii() and next_char.isalpha():
                return i + 1
            if curr.isascii() and curr.isalpha() and _is_cjk(next_char):
                return i + 1

    # 在CJK和数字之间断
    # 策略：CJK后面跟数字时，在CJK后断开（"自"后面跟"2026"）
    # 从max_chars向下搜索，找到最接近的CJK/数字边界
    for i in range(min(max_chars, len(text) - 1), max(0, max_chars - 10), -1):
        if i < len(text) - 1:
            curr = text[i]
            next_char = text[i + 1]
            # CJK后面跟数字 -> 在CJK后断开（保持"2026款"完整）
            if _is_cjk(curr) and next_char.isdigit():
                return i + 1

    return max_chars
