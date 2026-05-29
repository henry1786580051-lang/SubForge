"""字幕重断句模块

翻译后的字幕可能包含过长的段落，需要重新断句以符合字幕显示要求。
每条字幕只有一行中文 + 一行英文，如果太长就拆分成多条字幕。

使用共同拆分段数的顺序对齐方法，避免中英文行数不一致时出现重复或错位。
"""

import math
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
    中英文先计算共同拆分段数，再按顺序对齐。

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
    """按共同段数拆分字幕段

    策略：
    1. 根据中英文长度计算共同段数
    2. 中英文分别拆成相同数量的顺序片段
    3. 按序一一配对，避免短残片重复或跨句错配
    """
    part_count = _determine_part_count(en_text, zh_text, max_chars_en, max_chars_cjk)
    if part_count <= 1:
        return [seg]

    zh_lines = _split_text_to_exact_parts(zh_text, part_count, max_chars_cjk)
    en_lines = _split_text_to_exact_parts(en_text, part_count, max_chars_en)

    if len(zh_lines) <= 1 and len(en_lines) <= 1:
        return [seg]

    # 均匀分配时间戳
    duration = seg.end_time - seg.start_time
    segment_duration = duration / part_count

    result = []
    for i in range(part_count):
        en_line = en_lines[i] if i < len(en_lines) else ""
        zh_line = zh_lines[i] if i < len(zh_lines) else ""
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


def _determine_part_count(
    en_text: str,
    zh_text: str,
    max_chars_en: int,
    max_chars_cjk: int,
) -> int:
    """计算中英文共同拆分段数。

    旧实现会把中英文分别拆分后再贪心配对，当中文比英文多一行时会重复英文，
    或把短残片和下一句错配。这里先取双方所需段数的最大值，再让双方都按同一
    段数顺序拆分，保证每条双语字幕仍然一一对应。
    """
    en_parts = math.ceil(len(en_text) / max_chars_en) if en_text else 1
    zh_parts = math.ceil(len(zh_text) / max_chars_cjk) if zh_text else 1
    return max(1, en_parts, zh_parts)


def _split_text_to_exact_parts(text: str, part_count: int, max_chars: int) -> List[str]:
    """将文本拆成固定数量的顺序片段，尽量靠近自然断点。"""
    if part_count <= 1:
        return [text.strip()] if text else [""]

    remaining = text.strip()
    if not remaining:
        return [""] * part_count

    parts: List[str] = []
    for parts_left in range(part_count, 0, -1):
        if not remaining:
            parts.append("")
            continue

        if parts_left == 1:
            parts.append(remaining)
            break

        target = math.ceil(len(remaining) / parts_left)
        break_pos = _find_balanced_break_point(remaining, target, max_chars)
        break_pos = _adjust_break_for_remaining_parts(
            remaining, break_pos, parts_left - 1, max_chars
        )

        current = remaining[:break_pos].strip()
        if not current:
            break_pos = min(max(1, target), len(remaining))
            current = remaining[:break_pos].strip()

        parts.append(current)
        remaining = remaining[break_pos:].strip()

    if len(parts) < part_count:
        parts.extend([""] * (part_count - len(parts)))
    return parts[:part_count]


def _find_balanced_break_point(text: str, target: int, max_chars: int) -> int:
    """在目标长度附近寻找自然断点。"""
    if len(text) <= target:
        return len(text)

    upper = min(len(text) - 1, max(target + 8, int(target * 1.35)), max_chars)
    lower = max(1, min(int(target * 0.55), upper - 1))

    candidate_groups = [
        set("。！？!?；;"),
        set("，,、：:"),
        {" "},
    ]

    for chars in candidate_groups:
        pos = _find_break_char_skip_digit_cjk(text, chars, lower, upper, target)
        if pos is not None:
            return pos + (0 if text[pos] == " " else 1)

    # 语言边界：先在标准范围内搜索
    boundary = _closest_language_boundary(text, lower, upper, target)
    if boundary is not None:
        return boundary

    # 如果标准范围内找不到好的断点，扩展搜索到 max_chars*1.2
    # 用于保护"数字+量词"等技术参数组合（如"260马力"）
    extended_upper = min(len(text) - 1, int(max_chars * 1.2))
    if extended_upper > upper:
        boundary = _closest_language_boundary(text, lower, extended_upper, target)
        if boundary is not None:
            return boundary

    return min(target, len(text))


def _find_break_char_skip_digit_cjk(
    text: str,
    chars: set[str],
    lower: int,
    upper: int,
    target: int,
) -> int | None:
    """找最近的断点字符，优先选择不会拆散"数字+CJK"组合的位置。

    策略：
    1. 先找所有不会拆散 digit+CJK 的候选，按距离排序
    2. 如果没有安全候选，回退到会拆散 digit+CJK 的候选（选择距离最近的）

    检测两种 digit+CJK 拆分模式：
    - 前向：断点后紧跟数字，数字后跟CJK（如 "，260马力" 在逗号后断）
    - 后向：断点前是数字，断点后是数字，数字后跟CJK（如 "发动机，260" 在逗号后断，
      但"260"实际从逗号前的文本延续——这种情况下标点前后数字属于同一个数）
    """
    safe_candidates = []  # 不会拆散 digit+CJK
    unsafe_candidates = []  # 会拆散 digit+CJK

    for pos in range(lower, upper + 1):
        if text[pos] not in chars:
            continue
        if text[pos] == "." and pos > 0 and pos < len(text) - 1:
            if text[pos - 1].isdigit() and text[pos + 1].isdigit():
                continue

        splits_digit_cjk = _would_split_digit_cjk(text, pos)

        distance = abs(pos - target)
        if splits_digit_cjk:
            unsafe_candidates.append((distance, pos))
        else:
            safe_candidates.append((distance, pos))

    # 优先选择安全候选
    if safe_candidates:
        safe_candidates.sort()
        return safe_candidates[0][1]

    # 回退到不安全候选（当没有安全选项时）
    if unsafe_candidates:
        unsafe_candidates.sort()
        return unsafe_candidates[0][1]

    return None


def _would_split_digit_cjk(text: str, punct_pos: int) -> bool:
    """检查在标点位置断开是否会拆散 digit+CJK 组合。

    检测两种模式：
    1. 前向：标点后紧跟数字，数字后跟CJK → "，260马力" 在逗号后断会拆散
    2. 后向：标点前是数字，标点后也是数字，数字后跟CJK → "发动机，260马力"
       断在逗号后会把"260"的首位从前面的数字序列中拆走
    """
    after = punct_pos + 1

    # 前向检查：标点后紧跟数字
    if after < len(text) and text[after].isdigit():
        digit_end = after
        while digit_end < len(text) and text[digit_end].isdigit():
            digit_end += 1
        if digit_end < len(text) and _is_cjk(text[digit_end]):
            return True

    # 后向检查：标点前是数字，标点后也是数字（同一个数被标点分隔）
    if punct_pos > 0 and text[punct_pos - 1].isdigit() and after < len(text) and text[after].isdigit():
        # 标点前后都是数字 → 属于同一个数（如 "发动机，260" 中的 "260"）
        # 检查数字序列后面是否跟CJK
        digit_end = after
        while digit_end < len(text) and text[digit_end].isdigit():
            digit_end += 1
        if digit_end < len(text) and _is_cjk(text[digit_end]):
            return True

    return False


def _closest_break_char(
    text: str,
    chars: set[str],
    lower: int,
    upper: int,
    target: int,
) -> int | None:
    best_pos = None
    best_distance = float("inf")
    for pos in range(lower, upper + 1):
        if text[pos] not in chars:
            continue
        if text[pos] == "." and pos > 0 and pos < len(text) - 1:
            if text[pos - 1].isdigit() and text[pos + 1].isdigit():
                continue
        # 跳过会拆散"数字+CJK"组合的断点
        # 例："发动机，260马力" — 断在逗号后会把"2"带到上一段
        if pos + 1 < len(text) and text[pos + 1].isdigit():
            digit_end = pos + 1
            while digit_end < len(text) and text[digit_end].isdigit():
                digit_end += 1
            if digit_end < len(text) and _is_cjk(text[digit_end]):
                continue
        distance = abs(pos - target)
        if distance < best_distance:
            best_pos = pos
            best_distance = distance
    return best_pos


def _closest_language_boundary(
    text: str,
    lower: int,
    upper: int,
    target: int,
) -> int | None:
    """在范围内找到离 target 最近的语言/单词边界。

    优先级（同时按距离排序）：
    1. CJK <-> ASCII字母边界
    2. 数字 -> CJK 边界（保持"260马力"完整）
    3. ASCII字母 <-> 空格边界（英文单词边界）
    """
    best_pos = None
    best_distance = float("inf")
    for pos in range(lower, upper + 1):
        prev_char = text[pos - 1] if pos > 0 else ""
        curr_char = text[pos]
        if not prev_char or not curr_char:
            continue

        is_boundary = False

        # CJK <-> ASCII字母
        if _is_cjk(prev_char) and curr_char.isascii() and curr_char.isalpha():
            is_boundary = True
        elif prev_char.isascii() and prev_char.isalpha() and _is_cjk(curr_char):
            is_boundary = True
        # CJK -> 数字：需要检查数字序列后面是否跟着CJK
        # "升24升" → 断（数字后是CJK，数字是独立的）
        # "升24FA" → 断（数字后是ASCII字母，数字是独立的）
        # 不处理：数字→CJK（"260马力"不应被拆散）
        elif _is_cjk(prev_char) and curr_char.isdigit():
            # 检查数字序列后面的字符
            digit_end = pos
            while digit_end < len(text) and text[digit_end].isdigit():
                digit_end += 1
            next_after_digits = text[digit_end] if digit_end < len(text) else ""
            if next_after_digits and _is_cjk(next_after_digits):
                pass  # 数字后跟CJK → 数字是"CJK数字CJK"组合的一部分，不作为边界
            else:
                is_boundary = True
        # ASCII字母 <-> 空格（英文单词边界）
        elif prev_char == ' ' and curr_char.isascii() and curr_char.isalpha():
            is_boundary = True
        elif prev_char.isascii() and prev_char.isalpha() and curr_char == ' ':
            is_boundary = True

        if is_boundary:
            distance = abs(pos - target)
            if distance < best_distance:
                best_pos = pos
                best_distance = distance
    return best_pos


def _adjust_break_for_remaining_parts(
    text: str,
    break_pos: int,
    remaining_parts: int,
    max_chars: int,
) -> int:
    """避免拆出单字残片，或让后续片段明显超长。"""
    remaining_len = len(text[break_pos:].strip())
    if remaining_len == 1 and break_pos > 1:
        return break_pos - 1

    if remaining_parts > 0 and remaining_len > remaining_parts * max_chars:
        return max(1, len(text) - remaining_parts * max_chars)

    return break_pos


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
