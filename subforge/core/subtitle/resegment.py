"""字幕重断句模块

翻译后的字幕可能包含过长的段落，需要重新断句以符合字幕显示要求。
"""

import re
from typing import List

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.utils.logger import setup_logger

logger = setup_logger("subtitle_resegment")

# 默认字符限制
DEFAULT_MAX_CHARS_EN = 42  # 英文每行最大字符数（含空格）
DEFAULT_MAX_CHARS_CJK = 16  # CJK每行最大字符数
DEFAULT_MAX_LINES = 2  # 每条字幕最大行数


def resegment_subtitles(
    asr_data: ASRData,
    max_chars_en: int = DEFAULT_MAX_CHARS_EN,
    max_chars_cjk: int = DEFAULT_MAX_CHARS_CJK,
    max_lines: int = DEFAULT_MAX_LINES,
) -> ASRData:
    """对翻译后的字幕进行重断句

    Args:
        asr_data: 包含翻译文本的ASR数据
        max_chars_en: 英文每行最大字符数
        max_chars_cjk: CJK每行最大字符数
        max_lines: 每条字幕最大行数

    Returns:
        重断句后的ASR数据
    """
    new_segments = []

    for seg in asr_data.segments:
        # 处理原文和译文
        en_text = seg.text.strip()
        zh_text = seg.translated_text.strip() if seg.translated_text else ""

        if not en_text and not zh_text:
            continue

        # 判断是否需要拆分
        en_needs_split = _needs_split(en_text, max_chars_en, max_lines)
        zh_needs_split = _needs_split(zh_text, max_chars_cjk, max_lines)

        if not en_needs_split and not zh_needs_split:
            # 不需要拆分，直接保留
            new_segments.append(seg)
        else:
            # 需要拆分
            split_segments = _split_segment(
                seg, en_text, zh_text, max_chars_en, max_chars_cjk, max_lines
            )
            new_segments.extend(split_segments)

    logger.info(f"Resegmented: {len(asr_data.segments)} -> {len(new_segments)} segments")
    return ASRData(new_segments)


def _needs_split(text: str, max_chars: int, max_lines: int) -> bool:
    """判断文本是否需要拆分"""
    if not text:
        return False

    # 计算文本总字符数是否超过 max_lines * max_chars
    max_total = max_chars * max_lines
    return len(text) > max_total


def _split_segment(
    seg: ASRDataSeg,
    en_text: str,
    zh_text: str,
    max_chars_en: int,
    max_chars_cjk: int,
    max_lines: int,
) -> List[ASRDataSeg]:
    """拆分单个字幕段

    策略：
    1. 将文本拆分成适合 max_lines * max_chars 的块
    2. 每个块成为一个新的字幕段
    3. 均匀分配时间戳
    """
    # 计算每个块的最大字符数
    en_max_total = max_chars_en * max_lines
    zh_max_total = max_chars_cjk * max_lines

    # 将文本拆分成块
    en_chunks = _split_text_to_chunks(en_text, en_max_total, max_chars_en)
    zh_chunks = _split_text_to_chunks(zh_text, zh_max_total, max_chars_cjk)

    # 确定拆分数量（取最大值）
    num_splits = max(len(en_chunks), len(zh_chunks))
    if num_splits <= 1:
        return [seg]

    # 均匀分配时间戳
    duration = seg.end_time - seg.start_time
    segment_duration = duration / num_splits

    result = []
    for i in range(num_splits):
        start = int(seg.start_time + i * segment_duration)
        end = int(seg.start_time + (i + 1) * segment_duration)

        en_chunk = en_chunks[i] if i < len(en_chunks) else ""
        zh_chunk = zh_chunks[i] if i < len(zh_chunks) else ""

        # 格式化块为多行文本
        en_formatted = _format_chunk(en_chunk, max_chars_en)
        zh_formatted = _format_chunk(zh_chunk, max_chars_cjk)

        new_seg = ASRDataSeg(
            text=en_formatted,
            start_time=start,
            end_time=end,
            translated_text=zh_formatted,
            speaker_id=seg.speaker_id,
        )
        result.append(new_seg)

    return result


def _split_text_to_chunks(text: str, max_total_chars: int, max_chars_per_line: int) -> List[str]:
    """将文本拆分成适合显示的块

    每个块的字符数不超过 max_total_chars
    优先在句子边界拆分
    """
    if not text:
        return [""]

    if len(text) <= max_total_chars:
        return [text]

    # 按句子拆分
    sentences = _split_into_sentences(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # 如果当前块加上新句子不超过限制
        if current_chunk and len(current_chunk) + len(sentence) + 1 <= max_total_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        elif not current_chunk:
            # 当前块为空，检查句子是否太长
            if len(sentence) <= max_total_chars:
                current_chunk = sentence
            else:
                # 句子太长，需要进一步拆分
                sub_chunks = _split_long_sentence(sentence, max_total_chars, max_chars_per_line)
                chunks.extend(sub_chunks[:-1])
                current_chunk = sub_chunks[-1] if sub_chunks else ""
        else:
            # 当前块已满，开始新块
            chunks.append(current_chunk)
            if len(sentence) <= max_total_chars:
                current_chunk = sentence
            else:
                sub_chunks = _split_long_sentence(sentence, max_total_chars, max_chars_per_line)
                chunks.extend(sub_chunks[:-1])
                current_chunk = sub_chunks[-1] if sub_chunks else ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [""]


def _split_long_sentence(sentence: str, max_total_chars: int, max_chars_per_line: int) -> List[str]:
    """拆分过长的句子"""
    if len(sentence) <= max_total_chars:
        return [sentence]

    # 按标点或空格拆分
    chunks = []
    remaining = sentence

    while remaining:
        if len(remaining) <= max_total_chars:
            chunks.append(remaining)
            break

        # 在 max_total_chars 范围内寻找最佳断点
        break_pos = max_total_chars

        # 优先在空格处断
        space_pos = remaining.rfind(' ', 0, max_total_chars + 1)
        if space_pos > max_total_chars * 0.5:
            break_pos = space_pos
        else:
            # 在标点处断
            for punct in ['，', ',', '。', '.', '；', ';', '：', ':', '！', '!', '？', '?']:
                pos = remaining.rfind(punct, 0, max_total_chars + 1)
                if pos > max_total_chars * 0.5:
                    break_pos = pos + 1
                    break

        chunks.append(remaining[:break_pos].strip())
        remaining = remaining[break_pos:].strip()

    return chunks if chunks else [sentence]


def _format_chunk(chunk: str, max_chars_per_line: int) -> str:
    """将块格式化为多行文本（最多2行）"""
    if not chunk:
        return ""

    if len(chunk) <= max_chars_per_line:
        return chunk

    # 尝试在中间位置找空格断行
    mid = len(chunk) // 2

    # 优先在空格处断
    space_pos = chunk.rfind(' ', 0, mid + 10)
    if space_pos > len(chunk) * 0.3:
        line1 = chunk[:space_pos].strip()
        line2 = chunk[space_pos + 1:].strip()
    else:
        # 对于CJK文本，在标点处断
        cjk_puncts = ['，', '。', '！', '？', '；', '：', '、']
        best_pos = -1
        for punct in cjk_puncts:
            pos = chunk.rfind(punct, 0, mid + 5)
            if pos > len(chunk) * 0.3:
                best_pos = pos + 1
                break

        if best_pos > 0:
            line1 = chunk[:best_pos].strip()
            line2 = chunk[best_pos:].strip()
        else:
            # 如果找不到好的断点，直接在中间断
            line1 = chunk[:mid].strip()
            line2 = chunk[mid:].strip()

    return f"{line1}\n{line2}"


def _split_into_sentences(text: str) -> List[str]:
    """按句子边界拆分文本"""
    # 英文句子结束符
    # 中文句子结束符
    # 也按分号、冒号等拆分
    pattern = r'(?<=[.!?。！？；;:\n])\s*'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def _wrap_text(text: str, max_chars: int) -> List[str]:
    """将长文本按字符限制换行

    优先在空格、标点处断行
    """
    if len(text) <= max_chars:
        return [text]

    lines = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            lines.append(remaining)
            break

        # 在max_chars范围内寻找最佳断点
        break_pos = max_chars

        # 优先在空格处断（向前搜索）
        space_pos = remaining.rfind(' ', 0, max_chars + 1)
        if space_pos > max_chars * 0.3:  # 至少在30%位置之后
            break_pos = space_pos
        else:
            # 向后搜索空格（允许稍微超过max_chars）
            next_space = remaining.find(' ', max_chars)
            if next_space != -1 and next_space - max_chars < max_chars * 0.2:
                break_pos = next_space
            else:
                # 在标点处断
                for punct in ['，', ',', '。', '.', '；', ';', '：', ':', '！', '!', '？', '?']:
                    pos = remaining.rfind(punct, 0, max_chars + 1)
                    if pos > max_chars * 0.3:
                        break_pos = pos + 1
                        break

        lines.append(remaining[:break_pos].strip())
        remaining = remaining[break_pos:].strip()

    return lines if lines else [text]
