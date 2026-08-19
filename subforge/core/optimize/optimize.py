"""字幕优化模块

使用LLM优化字幕内容，支持agent loop自动验证和修正。
"""

import difflib
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..asr.asr_data import ASRData, ASRDataSeg
from ..entities import SubtitleProcessData
from ..llm import call_llm, get_response_text, parse_json_object
from ..llm.client import LLMRequestCancelled
from ..prompts import get_prompt
from ..split.alignment import SubtitleAligner
from ..utils.logger import setup_logger
from ..utils.text_utils import count_words

logger = setup_logger("subtitle_optimizer")

MAX_STEPS = 3
MIN_CROSS_KEY_COPY_TOKENS = 4
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def _ownership_tokens(text: str) -> List[str]:
    """Tokenize text for conservative adjacent-key ownership checks."""
    return re.findall(
        r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?|"
        r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u9fff\uac00-\ud7af]",
        str(text or "").lower(),
    )


def _contains_tokens(haystack: List[str], needle: List[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def _cross_key_ownership_violations(
    original_chunk: Dict[str, str], optimized_chunk: Dict[str, str]
) -> List[Tuple[str, str]]:
    """Detect a neighboring subtitle copied across a key boundary."""
    ordered_keys = sorted(original_chunk, key=lambda value: int(value))
    violations: List[Tuple[str, str]] = []
    for position, key in enumerate(ordered_keys):
        original = _ownership_tokens(original_chunk[key])
        optimized = _ownership_tokens(optimized_chunk[key])
        if optimized == original:
            continue

        if position + 1 < len(ordered_keys):
            next_key = ordered_keys[position + 1]
            next_tokens = _ownership_tokens(original_chunk[next_key])
            overlap = min(len(next_tokens), len(optimized), 12)
            while overlap >= MIN_CROSS_KEY_COPY_TOKENS:
                phrase = next_tokens[:overlap]
                if optimized[-overlap:] == phrase and not _contains_tokens(original, phrase):
                    violations.append(
                        (
                            key,
                            f"Key '{key}' copied the start of adjacent key '{next_key}': "
                            f"{' '.join(phrase)!r}",
                        )
                    )
                    break
                overlap -= 1

        if position > 0:
            previous_key = ordered_keys[position - 1]
            previous_tokens = _ownership_tokens(original_chunk[previous_key])
            overlap = min(len(previous_tokens), len(optimized), 12)
            while overlap >= MIN_CROSS_KEY_COPY_TOKENS:
                phrase = previous_tokens[-overlap:]
                if optimized[:overlap] == phrase and not _contains_tokens(original, phrase):
                    violations.append(
                        (
                            key,
                            f"Key '{key}' copied the end of adjacent key '{previous_key}': "
                            f"{' '.join(phrase)!r}",
                        )
                    )
                    break
                overlap -= 1
    return violations


def _cross_key_ownership_errors(
    original_chunk: Dict[str, str], optimized_chunk: Dict[str, str]
) -> List[str]:
    return [
        message
        for _, message in _cross_key_ownership_violations(
            original_chunk,
            optimized_chunk,
        )
    ]


def _is_duplicate_variant(left: str, right: str) -> bool:
    if left == right:
        return True
    return len(left) >= 3 and len(right) >= 3 and (
        left.rstrip("s") == right.rstrip("s")
    )


def _allowed_deleted_tokens(tokens: List[str], start: int, end: int) -> bool:
    deleted = tokens[start:end]
    if not deleted:
        return True
    if deleted == ["you", "know"] or all(token in {"um", "uh", "ah", "er"} for token in deleted):
        return True
    previous = tokens[start - 1] if start > 0 else ""
    following = tokens[end] if end < len(tokens) else ""
    return all(
        (previous and _is_duplicate_variant(token, previous))
        or (following and _is_duplicate_variant(token, following))
        for token in deleted
    )


def _allowed_replacement(original: List[str], optimized: List[str]) -> bool:
    # Joining or separating the same letters is a formatting/spelling repair,
    # not a lexical rewrite (for example ``Black berry`` -> ``Blackberry``).
    if "".join(original) == "".join(optimized) and len("".join(original)) >= 5:
        return True
    if len(original) != len(optimized):
        return False
    for source, target in zip(original, optimized):
        if source == target:
            continue
        if _NUMBER_WORDS.get(source) == target or _NUMBER_WORDS.get(target) == source:
            continue
        if difflib.SequenceMatcher(None, source, target).ratio() >= 0.72:
            continue
        return False
    return True


def _is_safe_phrase_correction(
    source_tokens: List[str],
    target_tokens: List[str],
    source_start: int,
    source_end: int,
    target_start: int,
    target_end: int,
) -> bool:
    """Allow narrowly defined grammar repairs without opening semantic rewrites."""
    source = source_tokens[source_start:source_end]
    target = target_tokens[target_start:target_end]
    if source != ["at"] or target != ["in"]:
        return False
    following = source_tokens[source_end : source_end + 2]
    return len(following) == 2 and following[0] == "the" and following[1] in {
        "last",
        "past",
    }


def _lexical_edit_violations(original: str, optimized: str) -> List[str]:
    """Reject semantic token edits while allowing explicit cleanup operations."""
    source_tokens = _ownership_tokens(original)
    target_tokens = _ownership_tokens(optimized)
    violations: List[str] = []
    for opcode, a0, a1, b0, b1 in difflib.SequenceMatcher(
        None, source_tokens, target_tokens
    ).get_opcodes():
        if opcode == "equal":
            continue
        if opcode == "delete" and _allowed_deleted_tokens(source_tokens, a0, a1):
            continue
        if opcode == "replace" and _allowed_replacement(
            source_tokens[a0:a1], target_tokens[b0:b1]
        ):
            continue
        if opcode == "replace" and _is_safe_phrase_correction(
            source_tokens,
            target_tokens,
            a0,
            a1,
            b0,
            b1,
        ):
            continue
        violations.append(
            f"{opcode}: {' '.join(source_tokens[a0:a1])!r} -> "
            f"{' '.join(target_tokens[b0:b1])!r}"
        )
    return violations


class SubtitleOptimizer:
    """字幕优化器

    使用LLM优化字幕内容，支持:
    - Agent loop自动验证和修正
    - 并发批量处理
    - 自动对齐修复
    """

    def __init__(
        self,
        thread_num: int,
        batch_num: int,
        model: str,
        custom_prompt: str,
        update_callback: Optional[Callable] = None,
        use_cache: bool = True,
        llm_client: Any = None,
    ):
        """初始化优化器

        Args:
            thread_num: 并发线程数
            batch_num: 每批处理的字幕数量
            model: LLM模型名称
            custom_prompt: 自定义优化提示词
            temperature: LLM温度参数
            update_callback: 进度更新回调函数
        """
        self.thread_num = thread_num
        self.batch_num = batch_num
        self.model = model
        self.custom_prompt = custom_prompt
        self.update_callback = update_callback
        self.use_cache = use_cache
        self.llm_client = llm_client

        self.is_running = True
        self.failed_batch_count = 0
        self._failure_lock = threading.Lock()
        self._executor_lock = threading.Lock()
        self.executor: Optional[ThreadPoolExecutor] = None
        self._init_thread_pool()

    def _init_thread_pool(self) -> None:
        """初始化线程池"""
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)

    def optimize_subtitle(self, subtitle_data: Union[str, ASRData]) -> ASRData:
        """优化字幕

        Args:
            subtitle_data: 字幕文件路径或ASRData对象

        Returns:
            优化后的ASRData对象
        """
        try:
            # Reading字幕
            if isinstance(subtitle_data, str):
                asr_data = ASRData.from_subtitle_file(subtitle_data)
            else:
                asr_data = subtitle_data

            # 转换为字典格式
            subtitle_dict = {
                str(i): seg.text for i, seg in enumerate(asr_data.segments, 1)
            }

            # 分批处理
            chunks = self._split_chunks(subtitle_dict)

            # 并行优化
            optimized_dict = self._parallel_optimize(chunks)

            # Chunks are processed independently, so also protect the boundaries
            # between batches. Only the offending key is restored.
            violations = _cross_key_ownership_violations(
                subtitle_dict,
                optimized_dict,
            )
            for key, message in violations:
                logger.warning("Rejected cross-key optimization: %s", message)
                optimized_dict[key] = subtitle_dict[key]

            # 创建新segments
            new_segments = self._create_segments(asr_data.segments, optimized_dict)

            return ASRData(new_segments)

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise RuntimeError(f"Optimization failed: {str(e)}")
        finally:
            self.stop()

    def _split_chunks(self, subtitle_dict: Dict[str, str]) -> List[Dict[str, str]]:
        """将字幕字典分割成批次

        Args:
            subtitle_dict: 字幕字典 {index: text}

        Returns:
            批次列表
        """
        items = list(subtitle_dict.items())
        return [
            dict(items[i : i + self.batch_num])
            for i in range(0, len(items), self.batch_num)
        ]

    def _parallel_optimize(self, chunks: List[Dict[str, str]]) -> Dict[str, str]:
        """并行优化All批次

        Args:
            chunks: 字幕批次列表

        Returns:
            优化后的字幕字典
        """
        if not self.executor:
            raise ValueError("Thread pool not initialized")

        futures = []
        optimized_dict: Dict[str, str] = {}

        # 提交All任务
        for chunk in chunks:
            future = self.executor.submit(self._optimize_chunk, chunk)
            futures.append((future, chunk))

        # 收集结果
        for future, chunk in futures:
            if not self.is_running:
                break

            try:
                result = future.result()
                optimized_dict.update(result)
            except Exception as e:
                logger.error(f"Optimization batch failed: {str(e)}")
                optimized_dict.update(chunk)  # 失败时保留原文

        return optimized_dict

    def _optimize_chunk(self, subtitle_chunk: Dict[str, str]) -> Dict[str, str]:
        """优化单个字幕批次

        Args:
            subtitle_chunk: 字幕批次字典

        Returns:
            优化后的字幕批次
        """
        start_idx = next(iter(subtitle_chunk))
        end_idx = next(reversed(subtitle_chunk))
        logger.debug(f"[+]Optimizing subtitles: {start_idx} - {end_idx}")

        try:
            result = self.agent_loop(subtitle_chunk)

            if self.update_callback:
                callback_data = [
                    SubtitleProcessData(
                        index=int(idx),
                        original_text=subtitle_chunk[idx],
                        optimized_text=result[idx],
                    )
                    for idx in sorted(result.keys(), key=int)
                ]
                self.update_callback(callback_data)

            return result

        except Exception as e:
            if isinstance(e, LLMRequestCancelled) or not self.is_running:
                raise
            with self._failure_lock:
                self.failed_batch_count += 1
            logger.error(f"Optimization failed: {str(e)}")
            return subtitle_chunk

    def agent_loop(self, subtitle_chunk: Dict[str, str]) -> Dict[str, str]:
        """使用agent loop优化字幕

        LLM → 验证 → 反馈 → 重试 (最多MAX_STEPS次)

        Args:
            subtitle_chunk: 字幕批次字典

        Returns:
            优化后的字幕批次

        Raises:
            ValueError: LLM returned empty result
        """
        # 构建提示词
        user_prompt = (
            f"Correct the following subtitles. Keep the original language, do not translate:\n"
            f"<input_subtitle>{str(subtitle_chunk)}</input_subtitle>"
        )

        if self.custom_prompt:
            user_prompt += (
                f"\nReference content:\n<reference>{self.custom_prompt}</reference>"
            )

        messages = [
            {"role": "system", "content": get_prompt("optimize/subtitle")},
            {"role": "user", "content": user_prompt},
        ]

        # Agent loop
        for step in range(MAX_STEPS):
            # 调用LLM
            response = call_llm(
                messages=messages,
                model=self.model,
                temperature=0.2,
                use_cache=self.use_cache,
                client=self.llm_client,
                # This pass is a tightly constrained source-preserving edit. Internal
                # reasoning frequently consumes DeepSeek's output budget before the
                # required JSON, while the validator already supplies targeted feedback.
                reasoning_mode="disabled",
                max_output_tokens=4096,
            )

            try:
                result_text = get_response_text(response)
                parsed_result = parse_json_object(result_text)
                result_dict: Dict[str, str] = parsed_result
                is_valid, error_message = self._validate_optimization_result(
                    original_chunk=subtitle_chunk, optimized_chunk=result_dict
                )
            except ValueError as error:
                result_text = ""
                is_valid = False
                error_message = str(error)

            if is_valid:
                return self._repair_subtitle(subtitle_chunk, result_dict)

            # 验证失败，添加反馈
            logger.warning(
                f"优化验证失败，开始反馈循环 (第{step + 1}次尝试): {error_message}"
            )
            if result_text:
                messages.append({"role": "assistant", "content": result_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Validation failed: {error_message}\n"
                        f"Please fix the errors and output ONLY a valid JSON dictionary."
                    ),
                }
            )

        # 达到最大步数
        logger.warning(
            "Max attempts reached(%s); preserving the original batch because every "
            "optimization response failed validation",
            MAX_STEPS,
        )
        return dict(subtitle_chunk)

    def _validate_optimization_result(
        self, original_chunk: Dict[str, str], optimized_chunk: Dict[str, str]
    ) -> Tuple[bool, str]:
        """验证优化结果

        检查:
        1. 键是否完全匹配
        2. 改动是否过大（相似度 < 0.7）

        Args:
            original_chunk: 原始字幕批次
            optimized_chunk: 优化后字幕批次

        Returns:
            (是否有效, Error反馈)
        """
        expected_keys = set(original_chunk.keys())
        actual_keys = set(optimized_chunk.keys())

        # 检查键匹配
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys

            error_parts = []
            if missing:
                error_parts.append(f"Missing keys: {sorted(missing)}")
            if extra:
                error_parts.append(f"Extra keys: {sorted(extra)}")

            error_msg = (
                "\n".join(error_parts) + f"\nRequired keys: {sorted(expected_keys)}\n"
                f"Please return the COMPLETE optimized dictionary with ALL {len(expected_keys)} keys."
            )
            return False, error_msg

        # 检查改动是否过大（逐条比较相似度）
        excessive_changes = []
        for key in expected_keys:
            original_text = original_chunk[key]
            optimized_text = optimized_chunk[key]

            # 清理文本用于比较
            original_cleaned = re.sub(r"\s+", " ", original_text).strip()
            optimized_cleaned = re.sub(r"\s+", " ", optimized_text).strip()

            # 计算相似度
            matcher = difflib.SequenceMatcher(None, original_cleaned, optimized_cleaned)
            similarity = matcher.ratio()
            similarity_threshold = 0.3 if count_words(original_text) <= 10 else 0.7

            # 相似度过低
            if similarity < similarity_threshold:
                excessive_changes.append(
                    f"Key '{key}': similarity {similarity:.1%} < {similarity_threshold:.0%}. "
                    f"Original: '{original_text}' → Optimized: '{optimized_text}' "
                )

        if excessive_changes:
            error_msg = ";\n".join(excessive_changes)
            error_msg += (
                "\n\nYour optimizations changed the text too much. "
                "Keep high similarity (≥70% for normal text) by making MINIMAL changes: "
                "only fix recognition errors and improve clarity, "
                "but preserve the original wording, length and structure as much as possible."
            )
            return False, error_msg

        ownership_errors = _cross_key_ownership_errors(
            original_chunk,
            optimized_chunk,
        )
        if ownership_errors:
            return False, (
                ";\n".join(ownership_errors)
                + "\nKeep every phrase in its original subtitle key. Do not copy text "
                "from the previous or next key."
            )

        lexical_changes = []
        for key in expected_keys:
            violations = _lexical_edit_violations(
                original_chunk[key], optimized_chunk[key]
            )
            if violations:
                lexical_changes.append(
                    f"Key '{key}' changed protected source words: {violations[:3]}"
                )
        if lexical_changes:
            return False, (
                ";\n".join(lexical_changes)
                + "\nOnly punctuation, capitalization, explicit filler removal, adjacent "
                "duplicate cleanup, number notation, and high-confidence spelling fixes are allowed."
            )

        return True, ""

    @staticmethod
    def _repair_subtitle(
        original: Dict[str, str], optimized: Dict[str, str]
    ) -> Dict[str, str]:
        """修复字幕对齐

        使用SubtitleAligner对齐原文和优化后的文本，
        处理优化过程中可能产生的段落合并或拆分。

        Args:
            original: 原始字幕字典
            optimized: 优化后字幕字典

        Returns:
            对齐后的字幕字典
        """
        if not all(isinstance(value, str) and value.strip() for value in optimized.values()):
            logger.warning("Optimized subtitle contains empty or non-string values, returning original")
            return dict(original)

        direct_repaired = {
            key: optimized[key] if key in optimized and optimized[key].strip() else text
            for key, text in original.items()
        }
        if any(key in optimized for key in original):
            return direct_repaired

        try:
            aligner = SubtitleAligner()
            original_list = list(original.values())
            optimized_list = list(optimized.values())

            aligned_source, aligned_target = aligner.align_texts(
                original_list, optimized_list
            )

            if len(aligned_source) != len(aligned_target):
                logger.warning("Alignment length mismatch, returning original")
                return dict(original)

            repaired: Dict[str, str] = {}
            original_keys = list(original.keys())
            for key, text in zip(original_keys, aligned_target):
                repaired[key] = text if isinstance(text, str) and text.strip() else original[key]
            for key in original_keys[len(repaired):]:
                repaired[key] = original[key]
            return repaired

        except Exception as e:
            logger.error(f"Alignment failed: {str(e)}, returning original")
            return dict(original)

    @staticmethod
    def _create_segments(
        original_segments: List[ASRDataSeg],
        optimized_dict: Dict[str, str],
    ) -> List[ASRDataSeg]:
        """从优化字典创建新的ASRDataSeg列表

        Args:
            original_segments: 原始Subtitle segment列表
            optimized_dict: 优化后字幕字典

        Returns:
            新的Subtitle segment列表
        """
        return [
            ASRDataSeg(
                text=optimized_dict.get(str(i), seg.text),
                start_time=seg.start_time,
                end_time=seg.end_time,
                translated_text=seg.translated_text,
                speaker_id=seg.speaker_id,
                words=list(seg.words),
                timestamp_granularity=seg.timestamp_granularity,
                timing_source=seg.timing_source,
                language_code=seg.language_code,
            )
            for i, seg in enumerate(original_segments, 1)
        ]

    def cancel(self) -> None:
        """Request prompt cancellation without blocking the caller thread."""
        self.is_running = False
        lock = getattr(self, "_executor_lock", None)
        if lock is None:
            executor = self.executor
        else:
            with lock:
                executor = self.executor
        if executor:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def stop(self) -> None:
        """Stop the optimizer and wait until its worker threads have exited."""
        self.is_running = False
        lock = getattr(self, "_executor_lock", None)
        if lock is None:
            executor = self.executor
            self.executor = None
        else:
            with lock:
                executor = self.executor
                self.executor = None
        if executor:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
