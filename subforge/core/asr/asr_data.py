import json
import math
import os
import platform
import re
from pathlib import Path
from typing import List, Optional, Tuple

from langdetect import LangDetectException, detect

from ..entities import SubtitleLayoutEnum
from ..utils.text_utils import is_mainly_cjk

# 多语言分词模式(支持词级和字符级语言)
_WORD_SPLIT_PATTERN = (
    r"[a-zA-Z\u00c0-\u00ff\u0100-\u017f']+"  # 拉丁字符(含扩展)
    r"|[\u0400-\u04ff]+"  # 西里尔字母(俄文)
    r"|[\u0370-\u03ff]+"  # 希腊字母
    r"|[\u0600-\u06ff]+"  # 阿拉伯文
    r"|[\u0590-\u05ff]+"  # 希伯来文
    r"|\d+"  # 数字
    r"|[\u4e00-\u9fff]"  # 中文
    r"|[\u3040-\u309f]"  # 日文平假名
    r"|[\u30a0-\u30ff]"  # 日文片假名
    r"|[\uac00-\ud7af]"  # 韩文
    r"|[\u0e00-\u0e7f][\u0e30-\u0e3a\u0e47-\u0e4e]*"  # 泰文
    r"|[\u0900-\u097f]"  # 天城文(印地语)
    r"|[\u0980-\u09ff]"  # 孟加拉文
    r"|[\u0e80-\u0eff]"  # 老挝文
    r"|[\u1000-\u109f]"  # 缅甸文
)


def reasonable_word_duration_ms(text: str) -> int:
    letters = len(re.sub(r"[^A-Za-z0-9\u00c0-\u017f]", "", text))
    if letters <= 2:
        return 650
    if letters <= 4:
        return 900
    return max(900, min(1800, letters * 180 + 500))


def handle_long_path(path: str) -> str:
    r"""Handle Windows long path limitation by adding \\?\ prefix.

    Args:
        path: Original file path

    Returns:
        Path with \\?\ prefix if needed (Windows only)
    """
    if (
        platform.system() == "Windows"
        and len(path) > 260
        and not path.startswith("\\\\?\\")
    ):
        return rf"\\?\{os.path.abspath(path)}"
    return path


class ASRDataSeg:
    def __init__(
        self, text: str, start_time: int, end_time: int, translated_text: str = "", speaker_id: str = ""
    ):
        self.text = text
        self.translated_text = translated_text
        self.start_time = start_time
        self.end_time = end_time
        self.speaker_id = speaker_id

    def to_srt_ts(self) -> str:
        """Convert to SRT timestamp format"""
        return f"{self._ms_to_srt_time(self.start_time)} --> {self._ms_to_srt_time(self.end_time)}"

    def to_lrc_ts(self) -> str:
        """Convert to LRC timestamp format"""
        return f"[{self._ms_to_lrc_time(self.start_time)}]"

    def to_ass_ts(self) -> Tuple[str, str]:
        """Convert to ASS timestamp format"""
        return self._ms_to_ass_ts(self.start_time), self._ms_to_ass_ts(self.end_time)

    @staticmethod
    def _ms_to_lrc_time(ms: int) -> str:
        """Convert milliseconds to LRC time format (MM:SS.cc)"""
        seconds = ms / 1000
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes):02}:{seconds:.2f}"

    @staticmethod
    def _ms_to_srt_time(ms: int) -> str:
        """Convert milliseconds to SRT time format (HH:MM:SS,mmm)"""
        total_seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"

    @staticmethod
    def _ms_to_ass_ts(ms: int) -> str:
        """Convert milliseconds to ASS timestamp format (H:MM:SS.cc)"""
        total_seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        centiseconds = int(milliseconds / 10)
        return f"{int(hours):01}:{int(minutes):02}:{int(seconds):02}.{centiseconds:02}"

    @property
    def transcript(self) -> str:
        """Return segment text"""
        return self.text

    def __str__(self) -> str:
        return f"ASRDataSeg({self.text}, {self.start_time}, {self.end_time})"


class ASRData:
    def __init__(self, segments: List[ASRDataSeg]):
        filtered_segments = [seg for seg in segments if seg.text and seg.text.strip()]
        filtered_segments.sort(key=lambda x: x.start_time)
        self.segments = filtered_segments

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def has_data(self) -> bool:
        """Check if there are any utterances"""
        return len(self.segments) > 0

    def _is_word_level_segment(self, segment: ASRDataSeg) -> bool:
        """判断单 segments是否为词级

        Args:
            segment: 待判断的字幕片段

        Returns:
            True 如果片段符合词级模式
        """
        text = segment.text.strip()

        # CJK语言: 1-2个字符
        if is_mainly_cjk(text):
            return len(text) <= 2

        # 非CJK语言（如英文）: 单个单词
        words = text.split()
        return len(words) == 1

    def is_word_timestamp(self) -> bool:
        """检查时间戳是否为词级(非句子级)

        词级判定标准:
        - 英文: 单个单词
        - CJK/亚洲语言: 1-2个字符
        - 允许20%误差容忍

        Returns:
            True 如果80%+的片段符合词级模式
        """
        if not self.segments:
            return False

        # 统计符合词级模式的片段数量
        word_level_count = sum(
            1 for seg in self.segments if self._is_word_level_segment(seg)
        )

        WORD_LEVEL_THRESHOLD = 0.8
        word_level_ratio = word_level_count / len(self.segments)

        return word_level_ratio >= WORD_LEVEL_THRESHOLD

    def split_to_word_segments(self) -> "ASRData":
        """将句子级字幕分割为词级字幕,并按音素估算分配时间戳

        时间戳分配基于音素估算(每4个字符约1个音素)

        Returns:
            修改后的ASRData实例
        """
        CHARS_PER_PHONEME = 4
        new_segments = []

        for seg in self.segments:
            text = seg.text
            duration = seg.end_time - seg.start_time

            # 使用统一的多语言分词模式
            words_list = list(re.finditer(_WORD_SPLIT_PATTERN, text))

            if not words_list:
                continue

            # 计算总音素数
            total_phonemes = sum(
                math.ceil(len(w.group()) / CHARS_PER_PHONEME) for w in words_list
            )
            time_per_phoneme = duration / max(total_phonemes, 1)

            # 为每个词分配时间戳
            current_time = seg.start_time
            for word_match in words_list:
                word = word_match.group()
                word_phonemes = math.ceil(len(word) / CHARS_PER_PHONEME)
                word_duration = int(time_per_phoneme * word_phonemes)

                word_end_time = min(current_time + word_duration, seg.end_time)
                new_segments.append(
                    ASRDataSeg(
                        text=word, start_time=current_time, end_time=word_end_time
                    )
                )
                current_time = word_end_time

        self.segments = new_segments
        return self

    def remove_punctuation(self) -> "ASRData":
        """Remove trailing Chinese punctuation (comma, period) from segments."""
        punctuation = r"[，。]"
        for seg in self.segments:
            seg.text = re.sub(f"{punctuation}+$", "", seg.text.strip())
            seg.translated_text = re.sub(
                f"{punctuation}+$", "", seg.translated_text.strip()
            )
        return self

    def save(
        self,
        save_path: str,
        ass_style: Optional[str] = None,
        layout: SubtitleLayoutEnum = SubtitleLayoutEnum.ORIGINAL_ON_TOP,
    ) -> None:
        """Save ASRData to file in specified format.

        Args:
            save_path: Output file path
            ass_style: ASS style string (optional, uses default if None)
            layout: Subtitle layout mode
        """
        self.fix_boundary_overlaps()
        save_path = handle_long_path(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        if save_path.endswith(".srt"):
            self.to_srt(save_path=save_path, layout=layout)
        elif save_path.endswith(".txt"):
            self.to_txt(save_path=save_path, layout=layout)
        elif save_path.endswith(".json"):
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.to_json(), f, ensure_ascii=False, indent=2)
        elif save_path.endswith(".ass"):
            self.to_ass(save_path=save_path, style_str=ass_style, layout=layout)
        else:
            raise ValueError(f"Unsupported file extension: {save_path}")

    def to_txt(
        self,
        save_path=None,
        layout: SubtitleLayoutEnum = SubtitleLayoutEnum.ORIGINAL_ON_TOP,
    ) -> str:
        """Convert to plain text subtitle format (without timestamps)"""
        result = []
        for seg in self.segments:
            original = seg.text
            translated = seg.translated_text

            if layout == SubtitleLayoutEnum.ORIGINAL_ON_TOP:
                text = f"{original}\n{translated}" if translated else original
            elif layout == SubtitleLayoutEnum.TRANSLATE_ON_TOP:
                text = f"{translated}\n{original}" if translated else original
            elif layout == SubtitleLayoutEnum.ONLY_ORIGINAL:
                text = original
            else:  # ONLY_TRANSLATE
                text = translated if translated else original
            result.append(text)
        text = "\n".join(result)
        if save_path:
            save_path = handle_long_path(save_path)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(result))
        return text

    def to_srt(
        self,
        layout: SubtitleLayoutEnum = SubtitleLayoutEnum.ORIGINAL_ON_TOP,
        save_path=None,
    ) -> str:
        """Convert to SRT subtitle format"""
        self.fix_boundary_overlaps()
        srt_lines = []
        for n, seg in enumerate(self.segments, 1):
            original = seg.text
            translated = seg.translated_text

            if layout == SubtitleLayoutEnum.ORIGINAL_ON_TOP:
                text = f"{original}\n{translated}" if translated else original
            elif layout == SubtitleLayoutEnum.TRANSLATE_ON_TOP:
                text = f"{translated}\n{original}" if translated else original
            elif layout == SubtitleLayoutEnum.ONLY_ORIGINAL:
                text = original
            else:  # ONLY_TRANSLATE
                text = translated if translated else original

            # Prefix speaker label if speaker_id is set
            if seg.speaker_id:
                speaker_prefix = f"[{seg.speaker_id}] "
                text = "\n".join(
                    speaker_prefix + line if line.strip() else line
                    for line in text.split("\n")
                )

            srt_lines.append(f"{n}\n{seg.to_srt_ts()}\n{text}\n")

        srt_text = "\n".join(srt_lines)
        if save_path:
            save_path = handle_long_path(save_path)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(srt_text)
        return srt_text

    def to_lrc(self, save_path=None) -> str:
        """Convert to LRC subtitle format"""
        raise NotImplementedError("LRC format is not supported")

    def to_json(self) -> dict:
        """Convert to JSON format"""
        self.fix_boundary_overlaps()
        result_json = {}
        for i, segment in enumerate(self.segments, 1):
            result_json[str(i)] = {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "original_subtitle": segment.text,
                "translated_subtitle": segment.translated_text,
            }
        return result_json

    def to_ass(
        self,
        style_str: Optional[str] = None,
        layout: SubtitleLayoutEnum = SubtitleLayoutEnum.ORIGINAL_ON_TOP,
        save_path: Optional[str] = None,
        video_width: int = 1280,
        video_height: int = 720,
    ) -> str:
        """Convert to ASS subtitle format

        Args:
            style_str: ASS style string (optional, uses default if None)
            layout: Subtitle layout mode
            save_path: Save path for ASS file (optional)
            video_width: Video width (default 1280)
            video_height: Video height (default 720)

        Returns:
            ASS format subtitle content
        """
        self.fix_boundary_overlaps()
        if not style_str:
            style_str = (
                "[V4+ Styles]\n"
                "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
                "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
                "Alignment,MarginL,MarginR,MarginV,Encoding\n"
                "Style: Default,MicrosoftYaHei-Bold,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,"
                "0,0,1,2,0,2,10,10,15,1\n"
                "Style: Secondary,MicrosoftYaHei-Bold,30,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,"
                "0,0,1,2,0,2,10,10,15,1"
            )

        ass_content = (
            "[Script Info]\n"
            "; Script generated by SubForge\n"
            "; https://github.com/weifeng2333\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {video_width}\n"
            f"PlayResY: {video_height}\n\n"
            f"{style_str}\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        dialogue_template = "Dialogue: 0,{},{},{},,0,0,0,,{}\n"
        for seg in self.segments:
            start_time, end_time = seg.to_ass_ts()
            # ASS uses \N for line breaks within dialogue
            original = seg.text.replace("\n", "\\N") if seg.text else ""
            translated = seg.translated_text.replace("\n", "\\N") if seg.translated_text else ""
            has_translation = bool(translated and translated.strip())

            if layout == SubtitleLayoutEnum.TRANSLATE_ON_TOP:
                if has_translation:
                    # Secondary(原文)先写(渲染在下)，Default(译文)后写(渲染在上)
                    ass_content += dialogue_template.format(
                        start_time, end_time, "Secondary", original
                    )
                    ass_content += dialogue_template.format(
                        start_time, end_time, "Default", translated
                    )
                else:
                    ass_content += dialogue_template.format(
                        start_time, end_time, "Default", original
                    )
            elif layout == SubtitleLayoutEnum.ORIGINAL_ON_TOP:
                if has_translation:
                    # Secondary(译文)先写(渲染在下)，Default(原文)后写(渲染在上)
                    ass_content += dialogue_template.format(
                        start_time, end_time, "Secondary", translated
                    )
                    ass_content += dialogue_template.format(
                        start_time, end_time, "Default", original
                    )
                else:
                    ass_content += dialogue_template.format(
                        start_time, end_time, "Default", original
                    )
            elif layout == SubtitleLayoutEnum.ONLY_ORIGINAL:
                ass_content += dialogue_template.format(
                    start_time, end_time, "Default", original
                )
            else:  # ONLY_TRANSLATE
                text = translated if has_translation else original
                ass_content += dialogue_template.format(
                    start_time, end_time, "Default", text
                )

        if save_path:
            save_path = handle_long_path(save_path)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(ass_content)
        return ass_content

    def to_vtt(self, save_path=None) -> str:
        """Convert to WebVTT subtitle format

        Args:
            save_path: Optional save path

        Returns:
            WebVTT format subtitle content
        """
        raise NotImplementedError("WebVTT format is not supported")
        # # WebVTT头部
        # vtt_lines = ["WEBVTT\n"]

        # for n, seg in enumerate(self.segments, 1):
        #     # 转换时间戳格式从毫秒到 HH:MM:SS.mmm
        #     start_time = seg._ms_to_srt_time(seg.start_time).replace(",", ".")
        #     end_time = seg._ms_to_srt_time(seg.end_time).replace(",", ".")

        #     # 添加序号（可选）和时间戳
        #     vtt_lines.append(f"{n}\n{start_time} --> {end_time}\n{seg.transcript}\n")

        # vtt_text = "\n".join(vtt_lines)

        # if save_path:
        #     with open(save_path, "w", encoding="utf-8") as f:
        #         f.write(vtt_text)

        # return vtt_text

    def merge_segments(
        self, start_index: int, end_index: int, merged_text: Optional[str] = None
    ):
        """Merge segments from start_index to end_index (inclusive)."""
        if (
            start_index < 0
            or end_index >= len(self.segments)
            or start_index > end_index
        ):
            raise IndexError("Invalid segment index")
        merged_start_time = self.segments[start_index].start_time
        merged_end_time = self.segments[end_index].end_time
        if merged_text is None:
            merged_text = "".join(
                seg.text for seg in self.segments[start_index : end_index + 1]
            )
        merged_translated = " ".join(
            seg.translated_text for seg in self.segments[start_index : end_index + 1]
            if seg.translated_text
        )
        merged_seg = ASRDataSeg(merged_text, merged_start_time, merged_end_time,
                                translated_text=merged_translated,
                                speaker_id=self.segments[start_index].speaker_id)
        self.segments[start_index : end_index + 1] = [merged_seg]

    def merge_with_next_segment(self, index: int) -> None:
        """Merge segment at index with next segment."""
        if index < 0 or index >= len(self.segments) - 1:
            raise IndexError("Index out of range or no next segment to merge")
        current_seg = self.segments[index]
        next_seg = self.segments[index + 1]
        merged_text = f"{current_seg.text} {next_seg.text}"
        merged_translated = ""
        if current_seg.translated_text or next_seg.translated_text:
            merged_translated = f"{current_seg.translated_text} {next_seg.translated_text}".strip()
        merged_seg = ASRDataSeg(merged_text, current_seg.start_time, next_seg.end_time,
                                translated_text=merged_translated,
                                speaker_id=current_seg.speaker_id)
        self.segments[index] = merged_seg
        del self.segments[index + 1]

    def filter_hallucinations(self, audio_path: Optional[str] = None) -> "ASRData":
        """Remove segments likely caused by ASR hallucination.

        If audio_path is provided, uses audio energy analysis to detect speech
        regions and removes segments in non-speech areas. Otherwise falls back
        to text-based heuristics.

        Args:
            audio_path: Path to audio file for energy-based speech detection

        Returns:
            Self for method chaining
        """
        import logging
        logger = logging.getLogger(__name__)

        if not self.segments:
            return self

        logger.info(f"filter_hallucinations called with audio_path={audio_path}, segments={len(self.segments)}")

        if self.is_word_timestamp():
            logger.info("Skipping hallucination energy filter for word-level timestamps")
            return self

        if audio_path:
            self._filter_by_audio_energy(audio_path)
        else:
            logger.info("No audio_path provided, using text heuristics")
            self._filter_by_text_heuristics()

        return self

    def deduplicate_adjacent_text(self, max_gap_ms: int = 1500) -> "ASRData":
        """Remove duplicate text emitted around adjacent ASR/VAD boundaries.

        VAD-segmented transcription gives Whisper overlap context around each
        speech segment. Some engines still timestamp that context inside the
        clipped VAD window, producing adjacent duplicate fragments such as a
        short prefix followed by the complete sentence. This pass only compares
        close neighboring subtitles and leaves in-segment repetitions intact.
        """
        import logging
        logger = logging.getLogger(__name__)

        def _token_spans(text: str) -> list[tuple[str, int, int]]:
            return [
                (m.group().lower(), m.start(), m.end())
                for m in re.finditer(_WORD_SPLIT_PATTERN, text)
            ]

        def _meaningful(tokens: list[str]) -> bool:
            return len(tokens) >= 3 or (len(tokens) >= 2 and sum(len(t) for t in tokens) >= 6)

        def _find_subsequence(haystack: list[str], needle: list[str]) -> int:
            if not needle or len(needle) > len(haystack):
                return -1
            last_start = len(haystack) - len(needle)
            for start in range(last_start + 1):
                if haystack[start:start + len(needle)] == needle:
                    return start
            return -1

        def _compact(tokens: list[str]) -> str:
            return "".join(tokens)

        def _longest_suffix_prefix(left: list[str], right: list[str]) -> int:
            max_len = min(len(left), len(right))
            for length in range(max_len, 0, -1):
                overlap = left[-length:]
                if overlap == right[:length] and _meaningful(overlap):
                    return length
            return 0

        def _trim_leading_tokens(seg: ASRDataSeg, count: int) -> bool:
            spans = _token_spans(seg.text)
            if count <= 0:
                return True
            if count >= len(spans):
                return False

            old_start = seg.start_time
            old_duration = max(0, seg.end_time - seg.start_time)
            ratio = count / len(spans)
            shift_ms = min(int(old_duration * ratio), max(0, old_duration - 200))
            cut_at = spans[count][1]

            seg.text = seg.text[cut_at:].lstrip(" \t\r\n,.;:!?，。！？；：-–—")
            if seg.translated_text:
                seg.translated_text = ""
            seg.start_time = min(seg.end_time, seg.start_time + shift_ms)

            logger.debug(
                "Trimmed duplicate ASR boundary prefix: %.2fs -> %.2fs, %s tokens",
                old_start / 1000,
                seg.start_time / 1000,
                count,
            )
            return bool(seg.text.strip())

        self.segments.sort(key=lambda s: (s.start_time, s.end_time))

        i = 0
        removed = 0
        trimmed = 0
        while i < len(self.segments) - 1:
            current = self.segments[i]
            next_seg = self.segments[i + 1]
            gap_ms = next_seg.start_time - current.end_time
            if gap_ms < 0 or gap_ms > max_gap_ms:
                i += 1
                continue

            current_tokens = [t for t, _, _ in _token_spans(current.text)]
            next_tokens = [t for t, _, _ in _token_spans(next_seg.text)]
            if not current_tokens or not next_tokens:
                i += 1
                continue

            # Same subtitle emitted twice at a chunk boundary.
            if current_tokens == next_tokens and _meaningful(current_tokens):
                logger.debug("Removed duplicate adjacent ASR segment: %r", next_seg.text)
                del self.segments[i + 1]
                removed += 1
                continue

            current_in_next = _find_subsequence(next_tokens, current_tokens)
            if current_in_next >= 0 and _meaningful(current_tokens):
                logger.debug(
                    "Removed contained ASR boundary fragment: %r inside %r",
                    current.text,
                    next_seg.text,
                )
                del self.segments[i]
                removed += 1
                i = max(0, i - 1)
                continue

            next_in_current = _find_subsequence(current_tokens, next_tokens)
            if next_in_current >= 0 and _meaningful(next_tokens):
                logger.debug(
                    "Removed repeated ASR boundary fragment: %r already in %r",
                    next_seg.text,
                    current.text,
                )
                del self.segments[i + 1]
                removed += 1
                continue

            current_compact = _compact(current_tokens)
            next_compact = _compact(next_tokens)
            if (
                len(current_compact) >= 10
                and current_compact in next_compact
                and _meaningful(current_tokens)
            ):
                logger.debug(
                    "Removed compact-contained ASR boundary fragment: %r inside %r",
                    current.text,
                    next_seg.text,
                )
                del self.segments[i]
                removed += 1
                i = max(0, i - 1)
                continue

            if (
                len(next_compact) >= 10
                and next_compact in current_compact
                and _meaningful(next_tokens)
            ):
                logger.debug(
                    "Removed compact-repeated ASR boundary fragment: %r already in %r",
                    next_seg.text,
                    current.text,
                )
                del self.segments[i + 1]
                removed += 1
                continue

            overlap_len = _longest_suffix_prefix(current_tokens, next_tokens)
            if overlap_len:
                if overlap_len >= len(next_tokens):
                    logger.debug("Removed duplicate ASR boundary suffix: %r", next_seg.text)
                    del self.segments[i + 1]
                    removed += 1
                    continue

                if _trim_leading_tokens(next_seg, overlap_len):
                    trimmed += 1
                    i += 1
                else:
                    del self.segments[i + 1]
                    removed += 1
                continue

            i += 1

        if removed or trimmed:
            logger.info(
                "Deduplicated adjacent ASR text: removed=%s, trimmed=%s",
                removed,
                trimmed,
            )

        return self

    def merge_sentence_fragments(
        self,
        max_gap_ms: int = 500,
        max_merged_duration_ms: int = 8000,
        max_merged_words: int = 22,
    ) -> "ASRData":
        """Merge short neighboring subtitles that split a sentence mid-phrase.

        whisper.cpp can emit subtitle boundaries based on decoder chunks rather
        than sentence boundaries, producing lines such as "I've always wanted"
        followed immediately by "to drive." This pass is deliberately
        conservative: it only merges close neighbors when the first line has no
        terminal punctuation and the combined subtitle remains readable.
        """
        import logging
        logger = logging.getLogger(__name__)

        if len(self.segments) < 2:
            return self

        continuation_starts = {
            "a", "an", "and", "as", "at", "because", "but", "by", "for",
            "from", "have", "in", "into", "is", "it", "its", "of", "on",
            "or", "seems", "than", "that", "the", "this", "to", "was",
            "were", "which", "with",
        }
        dangling_ends = {
            "a", "an", "and", "as", "at", "because", "but", "by", "for",
            "from", "i", "if", "in", "into", "is", "it", "of", "on", "or",
            "that", "the", "this", "to", "was", "were", "which", "with",
        }

        def _tokens(text: str) -> list[str]:
            return [m.group().lower() for m in re.finditer(_WORD_SPLIT_PATTERN, text)]

        def _ends_sentence(text: str) -> bool:
            return bool(re.search(r"[.!?。！？]\s*$", text.strip()))

        def _starts_lowercase(text: str) -> bool:
            stripped = text.lstrip()
            return bool(stripped) and stripped[0].islower()

        def _join_text(left: str, right: str) -> str:
            left = left.rstrip()
            right = right.lstrip()
            if not left:
                return right
            if not right:
                return left
            return f"{left} {right}"

        def _translated_text(left: ASRDataSeg, right: ASRDataSeg) -> str:
            if left.translated_text and right.translated_text:
                return _join_text(left.translated_text, right.translated_text)
            return ""

        def _should_merge(left: ASRDataSeg, right: ASRDataSeg) -> bool:
            gap_ms = right.start_time - left.end_time
            if gap_ms < 0 or gap_ms > max_gap_ms:
                return False
            if _ends_sentence(left.text):
                return False

            left_tokens = _tokens(left.text)
            right_tokens = _tokens(right.text)
            if not left_tokens or not right_tokens:
                return False

            combined_words = len(left_tokens) + len(right_tokens)
            combined_duration = right.end_time - left.start_time
            if combined_words > max_merged_words:
                return False
            if combined_duration > max_merged_duration_ms:
                return False

            left_last = left_tokens[-1]
            right_first = right_tokens[0]
            right_short = len(right_tokens) <= 5

            return (
                _starts_lowercase(right.text)
                or right_first in continuation_starts
                or left_last in dangling_ends
                or right_short
            )

        self.segments.sort(key=lambda s: (s.start_time, s.end_time))
        merged: list[ASRDataSeg] = []
        merge_count = 0

        for seg in self.segments:
            if merged and _should_merge(merged[-1], seg):
                prev = merged[-1]
                merged[-1] = ASRDataSeg(
                    _join_text(prev.text, seg.text),
                    prev.start_time,
                    seg.end_time,
                    translated_text=_translated_text(prev, seg),
                    speaker_id=prev.speaker_id,
                )
                merge_count += 1
            else:
                merged.append(seg)

        if merge_count:
            logger.info("Merged sentence-fragment subtitles: %s", merge_count)
            self.segments = merged

        return self

    def refine_timing_with_speech_segments(
        self,
        speech_segments: list[tuple[int, int]],
        min_leading_silence_ms: int = 1200,
        min_tail_silence_ms: int = 1000,
        speech_pad_ms: int = 250,
    ) -> "ASRData":
        """Trim subtitle edges that extend well past detected speech.

        This is intentionally conservative: VAD can miss quiet speech in car
        videos, so it only trims an edge when a detected speech overlap gives a
        strong replacement boundary and the subtitle has a large silent overrun.
        """
        import logging
        logger = logging.getLogger(__name__)

        if not self.segments or not speech_segments:
            return self

        def _word_count(text: str) -> int:
            return len(re.findall(_WORD_SPLIT_PATTERN, text))

        def _merge_speech_segments(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
            merged: list[tuple[int, int]] = []
            for start, end in sorted(segments):
                if end <= start:
                    continue
                if merged and start - merged[-1][1] <= 300:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            return merged

        def _min_readable_duration_ms(words: int) -> int:
            if words <= 0:
                return 0
            if words <= 5:
                return max(900, min(2600, words * 260 + 450))
            return max(2200, min(6000, words * 300 + 600))

        def _max_spoken_duration_ms(words: int) -> int:
            if words <= 0:
                return 0
            if words <= 4:
                return max(1400, words * 420 + 450)
            return max(2600, min(6500, words * 360 + 750))

        def _has_internal_sentence_boundary(text: str) -> bool:
            stripped = text.strip()
            return bool(re.search(r"[.!?。！？]\s+\S", stripped))

        speech = _merge_speech_segments(speech_segments)
        trimmed = 0

        for seg in self.segments:
            duration_ms = seg.end_time - seg.start_time
            words = _word_count(seg.text)
            if duration_ms < 1500 or words == 0:
                continue

            overlaps = []
            for speech_start, speech_end in speech:
                if speech_end < seg.start_time:
                    continue
                if speech_start > seg.end_time:
                    break
                overlap_start = max(seg.start_time, speech_start)
                overlap_end = min(seg.end_time, speech_end)
                if overlap_end > overlap_start:
                    overlaps.append((overlap_start, overlap_end))

            if not overlaps:
                continue

            first_speech_start = min(start for start, _ in overlaps)
            leading_silence_ms = first_speech_start - seg.start_time
            if leading_silence_ms >= min_leading_silence_ms:
                candidate_start = max(seg.start_time, first_speech_start - speech_pad_ms)
                remaining_duration = seg.end_time - candidate_start
                min_spoken_ms = max(500, min(2200, words * 160 + 250))
                if remaining_duration >= min_spoken_ms and candidate_start <= seg.end_time - 500:
                    logger.debug(
                        "Trimmed VAD-confirmed subtitle lead %.2fs-%.2fs to %.2fs-%.2fs: %r",
                        seg.start_time / 1000,
                        seg.end_time / 1000,
                        candidate_start / 1000,
                        seg.end_time / 1000,
                        seg.text,
                    )
                    seg.start_time = candidate_start
                    duration_ms = seg.end_time - seg.start_time
                    trimmed += 1

            last_speech_end = max(end for _, end in overlaps)
            tail_silence_ms = seg.end_time - last_speech_end
            if tail_silence_ms < min_tail_silence_ms:
                continue
            if words <= 6 and _has_internal_sentence_boundary(seg.text):
                continue

            max_spoken_ms = _max_spoken_duration_ms(words)
            short_tail_overrun = words <= 5 and tail_silence_ms >= 1200
            long_tail_overrun = (
                tail_silence_ms >= 1800
                and duration_ms > _min_readable_duration_ms(words) + 800
            )
            if duration_ms <= max_spoken_ms + 400 and not short_tail_overrun and not long_tail_overrun:
                continue

            min_end = seg.start_time + _min_readable_duration_ms(words)
            candidate_end = max(min_end, last_speech_end + speech_pad_ms)
            if candidate_end >= seg.end_time - 250:
                continue

            logger.debug(
                "Trimmed VAD-confirmed subtitle tail %.2fs-%.2fs to %.2fs-%.2fs: %r",
                seg.start_time / 1000,
                seg.end_time / 1000,
                seg.start_time / 1000,
                candidate_end / 1000,
                seg.text,
            )
            seg.end_time = candidate_end
            trimmed += 1

        if trimmed:
            logger.info("Trimmed subtitle edges with speech VAD: %s", trimmed)

        return self

    def _filter_by_audio_energy(self, audio_path: str) -> None:
        """Filter segments using audio energy-based speech detection.

        Two-pass approach:
        1. Remove segments in long silent regions (> 1s)
        2. Insert micro-gaps between segments where audio energy drops significantly
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            from pydub import AudioSegment
        except ImportError:
            logger.warning("pydub not available, falling back to text heuristics")
            self._filter_by_text_heuristics()
            return

        try:
            logger.info(f"Analyzing audio for speech detection: {audio_path}")
            audio = AudioSegment.from_file(audio_path)
            logger.info(f"Audio loaded: {len(audio)/1000:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return

        # Pass 1: Calculate RMS energy for each 50ms window
        window_ms = 50
        energies = []
        for i in range(0, len(audio), window_ms):
            chunk = audio[i:i + window_ms]
            rms = chunk.rms
            energies.append({'time_ms': i, 'rms': rms})

        if not energies:
            return

        # Calculate energy statistics
        rms_values = [e['rms'] for e in energies]
        avg_rms = sum(rms_values) / len(rms_values)
        # Silence threshold: 30% of average RMS, but at least 100
        silence_threshold = max(avg_rms * 0.3, 100)

        logger.info(f"Audio energy: avg={avg_rms:.0f}, threshold={silence_threshold:.0f}")

        speech_segments_for_refinement: list[tuple[int, int]] = []
        should_run_speech_vad = len(audio) >= 30_000 and len(self.segments) >= 8
        if should_run_speech_vad:
            try:
                from subforge.core.asr.silero_vad import detect_speech_segments
                from subforge.core.asr.silero_vad import is_available as vad_available

                if vad_available():
                    speech_segments_for_refinement = detect_speech_segments(
                        audio_path,
                        threshold=0.5,
                        min_speech_ms=200,
                        min_silence_ms=350,
                        speech_pad_ms=120,
                    )
            except Exception as e:
                logger.debug("Speech VAD timing refinement skipped: %s", e, exc_info=True)

        # Pass 2: Restore natural pauses when an ASR engine emits a continuous
        # timeline. Whisper.cpp commonly returns adjacent segments with
        # end_time == next.start_time even when the audio contains silence.
        min_pause_ms = 250
        search_radius_ms = 1200
        min_segment_duration_ms = 200

        def _find_silent_run_near_boundary(
            boundary_ms: int,
            search_start: int,
            search_end: int,
        ) -> tuple[int, int] | None:
            best: tuple[int, int] | None = None
            best_score: float | None = None
            run_start: int | None = None
            run_end: int | None = None

            for e in energies:
                t = e["time_ms"]
                if t < search_start:
                    continue
                if t >= search_end:
                    break

                if e["rms"] < silence_threshold:
                    if run_start is None:
                        run_start = t
                    run_end = min(t + window_ms, search_end)
                    continue

                if run_start is not None and run_end is not None:
                    if run_end - run_start >= min_pause_ms:
                        center = (run_start + run_end) / 2
                        score = abs(center - boundary_ms)
                        if best_score is None or score < best_score:
                            best = (run_start, run_end)
                            best_score = score
                    run_start = None
                    run_end = None

            if run_start is not None and run_end is not None:
                if run_end - run_start >= min_pause_ms:
                    center = (run_start + run_end) / 2
                    score = abs(center - boundary_ms)
                    if best_score is None or score < best_score:
                        best = (run_start, run_end)

            return best

        for i, seg in enumerate(self.segments[:-1]):
            next_seg = self.segments[i + 1]
            gap_ms = next_seg.start_time - seg.end_time

            # Preserve real gaps and let the overlap normalizer handle overlaps.
            if gap_ms > 50 or gap_ms < 0:
                continue

            boundary_ms = seg.end_time
            search_start = max(seg.start_time, boundary_ms - search_radius_ms)
            search_end = min(next_seg.end_time, boundary_ms + search_radius_ms)
            if search_end - search_start < min_pause_ms:
                continue

            silent_run = _find_silent_run_near_boundary(
                boundary_ms,
                search_start,
                search_end,
            )
            if not silent_run:
                continue

            pause_start, pause_end = silent_run
            if speech_segments_for_refinement:
                overlaps_speech = any(
                    min(pause_end, speech_end) - max(pause_start, speech_start) >= min_pause_ms // 2
                    for speech_start, speech_end in speech_segments_for_refinement
                )
                if overlaps_speech:
                    continue

            if pause_start - seg.start_time < min_segment_duration_ms:
                continue
            if next_seg.end_time - pause_end < min_segment_duration_ms:
                continue

            if pause_end > pause_start:
                logger.debug(
                    "Restored pause %.2fs-%.2fs between segments %s and %s",
                    pause_start / 1000,
                    pause_end / 1000,
                    i + 1,
                    i + 2,
                )
                seg.end_time = pause_start
                next_seg.start_time = pause_end

        # Pass 3: Trim/split long segments that cover clear internal silence.
        # This catches a different failure mode from overlaps: a single short
        # subtitle line can be stretched over many seconds of silence or music.
        def _word_count(text: str) -> int:
            return len(re.findall(_WORD_SPLIT_PATTERN, text))

        def _ends_with_sentence_punctuation(text: str) -> bool:
            return bool(re.search(r"[.!?。！？]\s*$", text.strip()))

        def _has_internal_sentence_boundary(text: str) -> bool:
            return bool(re.search(r"[.!?。！？]\s+\S", text.strip()))

        def _silent_runs(start_ms: int, end_ms: int, min_run_ms: int) -> list[tuple[int, int]]:
            runs = []
            run_start = None
            run_end = None
            for e in energies:
                t = e["time_ms"]
                if t < start_ms:
                    continue
                if t >= end_ms:
                    break
                if e["rms"] < silence_threshold:
                    if run_start is None:
                        run_start = t
                    run_end = min(t + window_ms, end_ms)
                elif run_start is not None and run_end is not None:
                    if run_end - run_start >= min_run_ms:
                        runs.append((run_start, run_end))
                    run_start = None
                    run_end = None
            if run_start is not None and run_end is not None:
                if run_end - run_start >= min_run_ms:
                    runs.append((run_start, run_end))
            return runs

        def _active_clusters(start_ms: int, end_ms: int) -> list[tuple[int, int]]:
            clusters = []
            cluster_start = None
            cluster_end = None
            last_active = None
            max_merge_gap_ms = 700
            for e in energies:
                t = e["time_ms"]
                if t < start_ms:
                    continue
                if t >= end_ms:
                    break
                if e["rms"] >= silence_threshold:
                    if cluster_start is None:
                        cluster_start = t
                    elif last_active is not None and t - last_active > max_merge_gap_ms:
                        clusters.append((cluster_start, cluster_end or last_active + window_ms))
                        cluster_start = t
                    cluster_end = min(t + window_ms, end_ms)
                    last_active = t
            if cluster_start is not None and cluster_end is not None:
                clusters.append((cluster_start, cluster_end))
            return clusters

        def _split_text_at_pause(text: str, ratio: float) -> tuple[str, str] | None:
            words = text.strip().split()
            if len(words) < 2:
                return None
            split_at = max(1, min(len(words) - 1, round(len(words) * ratio)))

            # Prefer a punctuation boundary near the duration-derived split.
            best = None
            for idx in range(max(1, split_at - 3), min(len(words), split_at + 4)):
                if re.search(r"[,.;:!?，。！？；：]$", words[idx - 1]):
                    best = idx
                    break
            if best is not None:
                split_at = best

            left = " ".join(words[:split_at]).strip()
            right = " ".join(words[split_at:]).strip()
            if not left or not right:
                return None
            return left, right

        adjusted_segments = []
        for seg in self.segments:
            duration_ms = seg.end_time - seg.start_time
            words = _word_count(seg.text)
            if duration_ms <= 0 or words == 0:
                continue

            max_reasonable_duration_ms = max(3500, min(8000, words * 500 + 800))

            # Very short text over a long span is usually a hallucinated line
            # during music/road noise, not speech.
            if words <= 4 and duration_ms > max(8000, words * 2500):
                logger.debug(
                    "Removed overlong short segment as hallucination: %.2fs-%.2fs %r",
                    seg.start_time / 1000,
                    seg.end_time / 1000,
                    seg.text,
                )
                continue

            if duration_ms > max_reasonable_duration_ms:
                clusters = _active_clusters(seg.start_time, seg.end_time)
                if len(clusters) == 1:
                    cluster_start, cluster_end = clusters[0]
                    padded_start = max(seg.start_time, cluster_start - 150)
                    padded_end = min(seg.end_time, cluster_end + 150)
                    padded_duration = padded_end - padded_start
                    enough_text_capacity = (
                        words <= 8
                        or padded_duration >= min(max_reasonable_duration_ms * 0.6, words * 300 + 500)
                    )
                    if padded_duration >= 500 and enough_text_capacity:
                        logger.debug(
                            "Trimmed overlong segment %.2fs-%.2fs to %.2fs-%.2fs",
                            seg.start_time / 1000,
                            seg.end_time / 1000,
                            padded_start / 1000,
                            padded_end / 1000,
                        )
                        seg.start_time = padded_start
                        seg.end_time = padded_end
                        duration_ms = seg.end_time - seg.start_time

            split_done = False
            if words >= 2 and duration_ms >= 2500:
                for pause_start, pause_end in _silent_runs(seg.start_time, seg.end_time, 600):
                    left_duration = pause_start - seg.start_time
                    right_duration = seg.end_time - pause_end
                    if left_duration < 500 or right_duration < 500:
                        continue
                    split_text = _split_text_at_pause(seg.text, left_duration / duration_ms)
                    if not split_text:
                        continue
                    left_text, right_text = split_text
                    adjusted_segments.append(
                        ASRDataSeg(
                            left_text,
                            seg.start_time,
                            pause_start,
                            translated_text=seg.translated_text,
                            speaker_id=seg.speaker_id,
                        )
                    )
                    adjusted_segments.append(
                        ASRDataSeg(
                            right_text,
                            pause_end,
                            seg.end_time,
                            translated_text="",
                            speaker_id=seg.speaker_id,
                        )
                    )
                    split_done = True
                    logger.debug(
                        "Split segment at silence %.2fs-%.2fs: %r | %r",
                        pause_start / 1000,
                        pause_end / 1000,
                        left_text,
                        right_text,
                    )
                    break
            if not split_done:
                duration_ms = seg.end_time - seg.start_time
                short_multi_sentence = words <= 6 and _has_internal_sentence_boundary(seg.text)
                if duration_ms > max_reasonable_duration_ms and not short_multi_sentence:
                    capped_end = seg.start_time + max_reasonable_duration_ms
                    logger.debug(
                        "Capped overlong segment %.2fs-%.2fs to %.2fs-%.2fs",
                        seg.start_time / 1000,
                        seg.end_time / 1000,
                        seg.start_time / 1000,
                        capped_end / 1000,
                    )
                    seg.end_time = capped_end
                adjusted_segments.append(seg)

        for seg in adjusted_segments:
            words = _word_count(seg.text)
            if words == 0:
                continue
            max_duration_ms = max(3500, min(8000, words * 500 + 800))
            duration_ms = seg.end_time - seg.start_time
            short_multi_sentence = words <= 6 and _has_internal_sentence_boundary(seg.text)
            if duration_ms > max_duration_ms and not short_multi_sentence:
                capped_end = seg.start_time + max_duration_ms
                logger.debug(
                    "Capped adjusted segment %.2fs-%.2fs to %.2fs-%.2fs",
                    seg.start_time / 1000,
                    seg.end_time / 1000,
                    seg.start_time / 1000,
                    capped_end / 1000,
                )
                seg.end_time = capped_end

            # Road noise can keep RMS high even after speech has ended, so pure
            # energy gating misses some subtitle tails. For complete sentences,
            # apply a conservative speech-rate cap to trim only the trailing end.
            duration_ms = seg.end_time - seg.start_time
            if (
                words >= 6
                and duration_ms >= 3500
                and _ends_with_sentence_punctuation(seg.text)
            ):
                sentence_tail_cap_ms = max(3000, min(7000, words * 320 + 600))
                if duration_ms > sentence_tail_cap_ms:
                    min_duration_ms = max(2500, words * 250)
                    trim_ms = min(duration_ms - sentence_tail_cap_ms, 800)
                    capped_end = max(seg.start_time + min_duration_ms, seg.end_time - trim_ms)
                    if capped_end < seg.end_time:
                        logger.debug(
                            "Trimmed sentence-final tail %.2fs-%.2fs to %.2fs-%.2fs",
                            seg.start_time / 1000,
                            seg.end_time / 1000,
                            seg.start_time / 1000,
                            capped_end / 1000,
                        )
                        seg.end_time = capped_end

        self.segments = adjusted_segments

        if speech_segments_for_refinement:
            self.refine_timing_with_speech_segments(speech_segments_for_refinement)

        # Pass 4: Remove segments that are entirely in silent regions
        original_count = len(self.segments)
        filtered = []
        for seg in self.segments:
            # Check average energy during this segment
            start_ms = max(0, seg.start_time)
            end_ms = min(len(audio), seg.end_time)

            if start_ms >= end_ms:
                continue

            # Sample energy in the segment
            segment_energies = []
            for e in energies:
                if start_ms <= e['time_ms'] < end_ms:
                    segment_energies.append(e['rms'])

            if segment_energies:
                avg_segment_rms = sum(segment_energies) / len(segment_energies)
                # Keep segment if it has enough energy (likely speech)
                if avg_segment_rms > silence_threshold * 0.5:
                    filtered.append(seg)
                else:
                    logger.debug(f"Removed silent segment: {seg.start_time/1000:.1f}-{seg.end_time/1000:.1f}s")
            else:
                filtered.append(seg)

        logger.info(f"Filtered segments: {original_count} -> {len(filtered)} (removed {original_count - len(filtered)})")
        self.segments = filtered

    def _filter_by_text_heuristics(self) -> None:
        """Filter segments using text-based heuristics (fallback)."""
        # Common whisper hallucination patterns
        hallucination_patterns = {
            "thank you", "thanks", "you", "the", "a", "an", "is", "it", "we",
            "um", "uh", "hmm", "ah", "oh", "like", "so", "well", "yeah",
            "okay", "ok", "right", "yes", "no", "and", "but", "or",
            "字幕", "字幕由", "感谢", "谢谢", "订阅", "观看",
        }

        filtered = []
        for seg in self.segments:
            text = seg.text.strip().lower().rstrip(".,!?。，！？")
            duration = seg.end_time - seg.start_time

            if duration >= 500:
                filtered.append(seg)
                continue

            if len(text) >= 3 and text not in hallucination_patterns:
                filtered.append(seg)

        self.segments = filtered

    def cap_abnormal_word_durations(
        self,
        default_max_ms: int = 900,
        numeric_max_ms: int = 1400,
        min_trim_ms: int = 250,
    ) -> "ASRData":
        """Cap obviously overlong word-level timestamps.

        WhisperX alignment can occasionally assign a whole silent region to one
        token (for example a number). This pass only applies to word-level data
        and only moves an abnormal token end earlier; it never deletes text or
        shifts following tokens.
        """
        import logging
        logger = logging.getLogger(__name__)

        if not self.segments or not self.is_word_timestamp():
            return self

        def _max_duration(text: str) -> int:
            stripped = text.strip().strip(".,;:!?()[]{}，。！？；：")
            if re.search(r"\d", stripped):
                return numeric_max_ms
            return min(default_max_ms, reasonable_word_duration_ms(stripped))

        capped = 0
        for seg in self.segments:
            duration = seg.end_time - seg.start_time
            if duration <= 0:
                continue
            max_duration = _max_duration(seg.text)
            if duration <= max_duration + min_trim_ms:
                continue
            candidate_end = seg.start_time + max_duration
            if candidate_end < seg.end_time - min_trim_ms:
                logger.debug(
                    "Capped abnormal word duration %.2fs -> %.2fs: %r",
                    duration / 1000,
                    (candidate_end - seg.start_time) / 1000,
                    seg.text,
                )
                seg.end_time = max(seg.start_time + 20, candidate_end)
                capped += 1

        if capped:
            logger.info("Capped abnormal word durations: %s", capped)
        return self

    def extend_sentence_tails_conservatively(
        self,
        min_gap_ms: int = 600,
        safety_gap_ms: int = 90,
        min_extension_ms: int = 250,
    ) -> "ASRData":
        """Extend short final subtitle tails into a following gap.

        This is a display/timing correction for sentence-level subtitles built
        from word timestamps. It deliberately avoids global padding: subtitles
        are only extended when they are shorter than a conservative readable
        target and a real gap exists before the next segment.
        """
        import logging
        logger = logging.getLogger(__name__)

        if len(self.segments) < 2 or self.is_word_timestamp():
            return self

        def _word_count(text: str) -> int:
            return len(re.findall(_WORD_SPLIT_PATTERN, text))

        def _cjk_count(text: str) -> int:
            return len(re.findall(r"[\u4e00-\u9fff]", text))

        def _has_timing_anchor(text: str) -> bool:
            return bool(
                re.search(r"\d", text)
                or re.search(r"\b[A-Z]\d+[A-Za-z0-9.-]*\b", text)
                or re.search(r"\b(?:RPM|U\.?S\.?)\b", text, flags=re.IGNORECASE)
            )

        def _target_duration(seg: ASRDataSeg) -> int:
            words = _word_count(seg.text)
            cjk = _cjk_count(seg.translated_text)
            if words <= 3:
                word_target = 1700
            elif words <= 6:
                word_target = 2400
            elif words <= 10:
                word_target = 3600
            else:
                word_target = min(5000, words * 310 + 700)
            if _has_timing_anchor(seg.text) and words >= 7:
                word_target = max(word_target, min(5200, words * 390 + 1_000))
            cjk_target = min(4600, cjk * 125 + 900) if cjk else 0
            return max(word_target, cjk_target)

        def _extension_cap(duration_ms: int) -> int:
            if duration_ms < 1200:
                return 2600
            if duration_ms < 2500:
                return 2400
            if duration_ms < 3500:
                return 1500
            if duration_ms < 4500:
                return 900
            return 0

        self.segments.sort(key=lambda s: (s.start_time, s.end_time))
        extended = 0

        for i, seg in enumerate(self.segments[:-1]):
            next_seg = self.segments[i + 1]
            gap_ms = next_seg.start_time - seg.end_time
            if gap_ms < min_gap_ms:
                continue

            duration_ms = seg.end_time - seg.start_time
            if gap_ms < 900 or duration_ms >= 4500:
                continue
            if not _has_timing_anchor(seg.text):
                continue
            if duration_ms >= 3500 and re.search(r"[.!?。！？]\s*$", seg.text.strip()):
                continue

            cap_ms = _extension_cap(duration_ms)
            if cap_ms <= 0:
                continue

            target_ms = _target_duration(seg)
            needed_ms = target_ms - duration_ms
            if needed_ms < min_extension_ms:
                continue

            extension_ms = min(needed_ms, cap_ms, gap_ms - safety_gap_ms)
            if extension_ms < min_extension_ms:
                continue

            old_end = seg.end_time
            seg.end_time += int(extension_ms)
            extended += 1
            logger.debug(
                "Extended subtitle tail %.2fs -> %.2fs (+%.2fs): %r",
                old_end / 1000,
                seg.end_time / 1000,
                extension_ms / 1000,
                seg.text,
            )

        if extended:
            logger.info("Extended conservative subtitle tails: %s", extended)
        self.fix_boundary_overlaps()
        return self

    def optimize_timing(self, threshold_ms: int = 100) -> "ASRData":
        """Optimize subtitle display timing by adjusting adjacent segment boundaries.

        Only adjusts very small gaps (< threshold) to reduce flicker.
        Larger gaps are preserved as natural speech pauses.

        Args:
            threshold_ms: Time gap threshold in milliseconds (default 100ms)

        Returns:
            Self for method chaining
        """
        if self.is_word_timestamp() or not self.segments:
            return self

        for i in range(len(self.segments) - 1):
            current_seg = self.segments[i]
            next_seg = self.segments[i + 1]
            time_gap = next_seg.start_time - current_seg.end_time

            # Only adjust very small gaps (micro-flicker), preserve larger pauses
            if 0 < time_gap < threshold_ms:
                mid_time = (
                    current_seg.end_time + next_seg.start_time
                ) // 2 + time_gap // 4
                current_seg.end_time = mid_time
                next_seg.start_time = mid_time

        return self

    def fix_boundary_overlaps(self, min_duration_ms: int = 1) -> "ASRData":
        """Fix remaining timestamp overlaps after VAD boundary clipping.

        Splits ordinary overlaps at a midpoint and clamps extreme overlaps so
        the final timeline is monotonically non-overlapping.

        Returns:
            Self for method chaining
        """
        if not self.segments:
            return self

        min_duration_ms = max(0, min_duration_ms)
        self.segments.sort(key=lambda s: (s.start_time, s.end_time))

        for seg in self.segments:
            if seg.end_time < seg.start_time:
                seg.end_time = seg.start_time

        for i in range(1, len(self.segments)):
            prev = self.segments[i - 1]
            curr = self.segments[i]
            if prev.end_time <= curr.start_time:
                continue

            prev_min_end = prev.start_time + min_duration_ms
            curr_max_start = curr.end_time - min_duration_ms

            if prev_min_end <= curr_max_start:
                split = (prev.end_time + curr.start_time) // 2
                split = max(prev_min_end, min(split, curr_max_start))
            else:
                split = max(prev.start_time, min(curr.start_time, curr.end_time))

            prev.end_time = min(prev.end_time, split)
            curr.start_time = max(curr.start_time, split)
            if curr.end_time < curr.start_time:
                curr.end_time = curr.start_time

        return self

    def __str__(self):
        return self.to_txt()

    @staticmethod
    def from_subtitle_file(file_path: str) -> "ASRData":
        """Load ASRData from subtitle file.

        Args:
            file_path: Subtitle file path (supports .srt, .vtt, .ass, .json)

        Returns:
            Parsed ASRData instance

        Raises:
            FileNotFoundError: File does not exist
            ValueError: Unsupported file format
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path_obj}")

        try:
            content = file_path_obj.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path_obj.read_text(encoding="gbk")

        suffix = file_path_obj.suffix.lower()

        if suffix == ".srt":
            return ASRData.from_srt(content)
        elif suffix == ".vtt":
            if "<c>" in content:
                return ASRData.from_youtube_vtt(content)
            return ASRData.from_vtt(content)
        elif suffix == ".ass":
            return ASRData.from_ass(content)
        elif suffix == ".json":
            return ASRData.from_json(json.loads(content))
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def from_json(json_data: dict) -> "ASRData":
        """Create ASRData from JSON data"""
        segments = []
        for i in sorted(json_data.keys(), key=int):
            segment_data = json_data[i]
            segment = ASRDataSeg(
                text=segment_data["original_subtitle"],
                translated_text=segment_data["translated_subtitle"],
                start_time=segment_data["start_time"],
                end_time=segment_data["end_time"],
            )
            segments.append(segment)
        return ASRData(segments)

    @staticmethod
    def from_srt(srt_str: str) -> "ASRData":
        """Create ASRData from SRT format string.

        Detect bilingual subtitles block-by-block. This supports both
        source-above and target-above layouts and preserves multiline source or
        target text when the two language groups are clearly separable.

        Args:
            srt_str: SRT format subtitle string

        Returns:
            Parsed ASRData instance
        """
        segments = []
        srt_time_pattern = re.compile(
            r"(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})\s-->\s(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})"
        )
        speaker_pattern = re.compile(r"^\[(说话人\d+|Speaker \d+)\]\s*")
        blocks = re.split(r"\n\s*\n", srt_str.strip())

        def _line_family(text: str) -> str:
            """Classify a subtitle text line without being fooled by model names.

            A Chinese translation often contains Latin tokens such as W126,
            AMG, Mercedes-Benz, or email addresses. Presence of meaningful CJK
            text therefore wins over embedded Latin identifiers.
            """
            stripped = text.strip()
            if not stripped:
                return "empty"
            cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", stripped))
            latin_count = len(re.findall(r"[A-Za-z]", stripped))
            if cjk_count >= 2:
                return "cjk"
            if latin_count >= 2 and cjk_count == 0:
                return "latin"
            if cjk_count:
                return "cjk"
            return "other"

        def _group_family(lines: list[str]) -> str:
            joined = " ".join(line.strip() for line in lines if line.strip())
            family = _line_family(joined)
            if family in {"cjk", "latin"}:
                return family
            families = [_line_family(line) for line in lines if line.strip()]
            cjk = families.count("cjk")
            latin = families.count("latin")
            if cjk > latin and cjk > 0:
                return "cjk"
            if latin > cjk and latin > 0:
                return "latin"
            return "other"

        def _fallback_different_language(left: str, right: str) -> bool:
            try:
                return detect(left) != detect(right)
            except LangDetectException:
                return False

        def _split_bilingual_lines(text_lines: list[str]) -> tuple[str, str] | None:
            non_empty = [line for line in text_lines if line.strip()]
            if len(non_empty) < 2:
                return None

            best: tuple[int, int, str, str] | None = None
            for split_index in range(1, len(non_empty)):
                left = non_empty[:split_index]
                right = non_empty[split_index:]
                left_family = _group_family(left)
                right_family = _group_family(right)

                score = 0
                if {left_family, right_family} == {"cjk", "latin"}:
                    score = 100
                elif left_family != right_family and left_family != "other" and right_family != "other":
                    score = 60
                elif (
                    left_family != right_family
                    and len(left) == 1
                    and len(right) == 1
                    and _fallback_different_language(left[0], right[0])
                ):
                    score = 40

                if score <= 0:
                    continue
                # Prefer balanced split points when multiple options look valid.
                balance_penalty = abs(len(left) - len(right))
                candidate = (score - balance_penalty, split_index, left_family, right_family)
                if best is None or candidate > best:
                    best = candidate

            if best is None:
                return None

            _, split_index, left_family, right_family = best
            left_text = "\n".join(non_empty[:split_index]).strip()
            right_text = "\n".join(non_empty[split_index:]).strip()

            if {left_family, right_family} == {"cjk", "latin"}:
                if left_family == "latin":
                    return left_text, right_text
                return right_text, left_text

            return left_text, right_text

        # Process all blocks based on detected mode
        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 3:
                continue

            match = srt_time_pattern.match(lines[1])
            if not match:
                continue

            time_parts = list(map(int, match.groups()))
            start_time = sum(
                [
                    time_parts[0] * 3600000,
                    time_parts[1] * 60000,
                    time_parts[2] * 1000,
                    time_parts[3],
                ]
            )
            end_time = sum(
                [
                    time_parts[4] * 3600000,
                    time_parts[5] * 60000,
                    time_parts[6] * 1000,
                    time_parts[7],
                ]
            )

            text_lines = lines[2:]

            # Extract speaker_id from the first text line if present
            speaker_id = ""
            if text_lines:
                speaker_match = speaker_pattern.match(text_lines[0])
                if speaker_match:
                    speaker_id = speaker_match.group(1)
                    text_lines[0] = text_lines[0][speaker_match.end():]

            bilingual = _split_bilingual_lines(text_lines)
            if bilingual:
                original, translated = bilingual
                segments.append(
                    ASRDataSeg(
                        original,
                        start_time,
                        end_time,
                        translated,
                        speaker_id=speaker_id,
                    )
                )
            elif len(text_lines) == 1:
                segments.append(ASRDataSeg(text_lines[0], start_time, end_time, speaker_id=speaker_id))
            else:
                # Multi-line subtitle: preserve line breaks with \n
                segments.append(ASRDataSeg("\n".join(text_lines), start_time, end_time, speaker_id=speaker_id))

        return ASRData(segments)

    @staticmethod
    def from_vtt(vtt_str: str) -> "ASRData":
        """Create ASRData from VTT format string.

        Args:
            vtt_str: VTT format subtitle string

        Returns:
            ASRData instance
        """
        segments = []
        # Split by blank lines, skip the WEBVTT header block
        blocks = vtt_str.strip().split("\n\n")
        # Find first block after header (skip WEBVTT line and any NOTE/STYLE blocks)
        content = []
        header_done = False
        for block in blocks:
            stripped = block.strip()
            if not header_done:
                if stripped.startswith("WEBVTT") or stripped.startswith("NOTE") or stripped.startswith("STYLE"):
                    continue
                header_done = True
            if stripped:
                content.append(stripped)

        # Support both HH:MM:SS.mmm and MM:SS.mmm (VTT allows omitting hours)
        timestamp_pattern = re.compile(
            r"(?:(\d{2}):)?(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(?:(\d{2}):)?(\d{2}):(\d{2})\.(\d{3})"
        )

        for block in content:
            lines = block.split("\n")
            if not lines:
                continue

            # Find the timestamp line (could be first line or second if cue ID present)
            timestamp_line = None
            text_start = 0
            for i, line in enumerate(lines):
                if "-->" in line:
                    timestamp_line = line
                    text_start = i + 1
                    break

            if not timestamp_line:
                continue
            match = timestamp_pattern.match(timestamp_line.strip())
            if not match:
                continue

            groups = match.groups()
            time_parts = [int(g) if g is not None else 0 for g in groups]
            start_time = (
                time_parts[0] * 3600000 + time_parts[1] * 60000 +
                time_parts[2] * 1000 + time_parts[3]
            )
            end_time = (
                time_parts[4] * 3600000 + time_parts[5] * 60000 +
                time_parts[6] * 1000 + time_parts[7]
            )

            text_line = "\n".join(lines[text_start:])
            # Remove VTT inline tags: timestamps, <c>, <b>, <i>, <u>, <ruby>, etc.
            cleaned_text = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", text_line)
            cleaned_text = re.sub(r"</?[a-zA-Z][^>]*>", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

            if cleaned_text and cleaned_text != " ":
                segments.append(ASRDataSeg(cleaned_text, start_time, end_time))

        return ASRData(segments)

    @staticmethod
    def from_youtube_vtt(vtt_str: str) -> "ASRData":
        """Create ASRData from YouTube VTT format with word-level timestamps.

        Args:
            vtt_str: YouTube VTT format subtitle string (contains <c> tags)

        Returns:
            Parsed ASRData with word-level segments
        """

        def parse_timestamp(ts: str) -> int:
            """Convert timestamp string to milliseconds"""
            h, m, s = ts.split(":")
            return int(float(h) * 3600000 + float(m) * 60000 + float(s) * 1000)

        def split_timestamped_text(text: str) -> List[ASRDataSeg]:
            """Extract word segments from timestamped text"""
            pattern = re.compile(r"<(\d{2}:\d{2}:\d{2}\.\d{3})>([^<]*)")
            matches = list(pattern.finditer(text))
            word_segments = []

            for i in range(len(matches) - 1):
                current_match = matches[i]
                next_match = matches[i + 1]

                start_time = parse_timestamp(current_match.group(1))
                end_time = parse_timestamp(next_match.group(1))
                word = current_match.group(2).strip()

                if word:
                    word_segments.append(ASRDataSeg(word, start_time, end_time))

            return word_segments

        segments = []
        blocks = re.split(r"\n\n+", vtt_str.strip())

        timestamp_pattern = re.compile(
            r"(\d{2}):(\d{2}):(\d{2}\.\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}\.\d{3})"
        )
        for block in blocks:
            lines = block.strip().split("\n")
            if not lines:
                continue

            match = timestamp_pattern.match(lines[0])
            if not match:
                continue

            text = "\n".join(lines)

            timestamp_row = re.search(r"\n(.*?<c>.*?</c>.*)", block)
            if timestamp_row:
                text = re.sub(r"<c>|</c>", "", timestamp_row.group(1))
                block_start_time_string = (
                    f"{match.group(1)}:{match.group(2)}:{match.group(3)}"
                )
                block_end_time_string = (
                    f"{match.group(4)}:{match.group(5)}:{match.group(6)}"
                )
                text = f"<{block_start_time_string}>{text}<{block_end_time_string}>"

                word_segments = split_timestamped_text(text)
                segments.extend(word_segments)

        return ASRData(segments)

    @staticmethod
    def from_ass(ass_str: str) -> "ASRData":
        """Create ASRData from ASS format string.

        Args:
            ass_str: ASS format subtitle string

        Returns:
            ASRData instance
        """
        segments = []
        ass_time_pattern = re.compile(
            r"Dialogue: \d+,(\d+:\d{2}:\d{2}\.\d{2}),(\d+:\d{2}:\d{2}\.\d{2}),(.*?),.*?,\d+,\d+,\d+,.*?,(.*?)$"
        )

        def parse_ass_time(time_str: str) -> int:
            """Convert ASS timestamp to milliseconds"""
            hours, minutes, seconds = time_str.split(":")
            seconds, centiseconds = seconds.split(".")
            return (
                int(hours) * 3600000
                + int(minutes) * 60000
                + int(seconds) * 1000
                + int(centiseconds) * 10
            )

        # 检查是否有翻译: 同时存在Default和Secondary样式
        has_default = "Dialogue:" in ass_str and ",Default," in ass_str
        has_secondary = ",Secondary," in ass_str
        has_translation = has_default and has_secondary
        temp_segments = {}

        for line in ass_str.splitlines():
            if line.startswith("Dialogue:"):
                match = ass_time_pattern.match(line)
                if match:
                    start_time = parse_ass_time(match.group(1))
                    end_time = parse_ass_time(match.group(2))
                    style = match.group(3).strip()
                    text = match.group(4)

                    text = re.sub(r"\{[^}]*\}", "", text)
                    text = text.replace("\\N", "\n")
                    text = text.strip()

                    if not text:
                        continue

                    if has_translation:
                        time_key = f"{start_time}-{end_time}"
                        if time_key in temp_segments:
                            # Default style = original text, Secondary = translated
                            if style == "Default":
                                temp_segments[time_key].text = text
                            else:
                                temp_segments[time_key].translated_text = text
                            segments.append(temp_segments[time_key])
                            del temp_segments[time_key]
                        else:
                            segment = ASRDataSeg(
                                text="", start_time=start_time, end_time=end_time
                            )
                            if style == "Default":
                                segment.text = text
                            else:
                                segment.translated_text = text
                            temp_segments[time_key] = segment
                    else:
                        segments.append(ASRDataSeg(text, start_time, end_time))

        for segment in temp_segments.values():
            segments.append(segment)

        return ASRData(segments)
