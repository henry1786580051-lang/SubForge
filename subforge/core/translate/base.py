"""翻译器基类"""

import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.types import TargetLanguage
from subforge.core.utils.cache import generate_cache_key, get_translate_cache, is_cache_enabled
from subforge.core.utils.logger import setup_logger

logger = setup_logger("subtitle_translator")


class PartialTranslationError(RuntimeError):
    """A batch failed after producing validated translations for some items."""

    def __init__(
        self,
        message: str,
        completed: List[SubtitleProcessData],
        failed_indices: List[int],
    ):
        super().__init__(message)
        self.completed = completed
        self.failed_indices = failed_indices


class BaseTranslator(ABC):
    """翻译器基类"""

    def __init__(
        self,
        thread_num: int,
        batch_num: int,
        target_language: TargetLanguage,
        update_callback: Optional[Callable],
        use_cache: bool = True,
    ):
        if thread_num <= 0:
            raise ValueError("thread_num must be positive")
        if batch_num <= 0:
            raise ValueError("batch_num must be positive")

        self.thread_num = thread_num
        self.batch_num = batch_num
        self.target_language = target_language
        self.is_running = True
        self.update_callback = update_callback
        self.use_cache = use_cache
        self.executor = None
        self._cache = get_translate_cache()

        self._init_thread_pool()

    def _init_thread_pool(self):
        """初始化线程池"""
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)

    def _ensure_thread_pool(self) -> None:
        """Allow a translator instance to be reused after a completed run."""
        if self.executor is None:
            self.is_running = True
            self._init_thread_pool()

    def translate_subtitle(self, subtitle_data: ASRData) -> ASRData:
        """翻译字幕文件"""
        try:
            self._ensure_thread_pool()
            asr_data = subtitle_data

            # 将ASRData转换为SubtitleProcessData列表
            translate_data_list = [
                SubtitleProcessData(index=i, original_text=seg.text)
                for i, seg in enumerate(asr_data.segments, 1)
            ]

            # 分批处理字幕
            chunks = self._split_chunks(translate_data_list)

            # 多线程翻译
            translated_list = self._parallel_translate(chunks)
            self._validate_translated_list(translate_data_list, translated_list)

            # 设置Subtitle segment的翻译文本
            new_segments = self._set_segments_translated_text(asr_data.segments, translated_list)

            return ASRData(new_segments)
        except RuntimeError:
            logger.exception("Translation failed")
            raise
        except Exception as e:
            logger.exception("Translation failed")
            raise RuntimeError(f"Translation failed: {str(e)}") from e
        finally:
            self.stop()

    def _split_chunks(
        self, translate_data_list: List[SubtitleProcessData]
    ) -> List[List[SubtitleProcessData]]:
        """将字幕分割成块"""
        return [
            translate_data_list[i : i + self.batch_num]
            for i in range(0, len(translate_data_list), self.batch_num)
        ]

    def _parallel_translate(
        self, chunks: List[List[SubtitleProcessData]]
    ) -> List[SubtitleProcessData]:
        """并行翻译All块"""
        future_to_chunk = {}
        translated_list = []
        failed_count = 0
        failed_errors: list[str] = []
        total_segments = sum(len(c) for c in chunks)
        if self.executor is None:
            raise RuntimeError("Translation thread pool is not initialized")

        for chunk in chunks:
            future = self.executor.submit(self._safe_translate_chunk, chunk)
            future_to_chunk[future] = chunk

        for future in as_completed(future_to_chunk):
            if not self.is_running:
                break
            try:
                result = future.result()
                translated_list.extend(result)
            except Exception as e:
                logger.error(f"Translation chunk failed: {e}")
                failed_errors.append(str(e))
                if isinstance(e, PartialTranslationError):
                    translated_list.extend(e.completed)
                    failed_count += len(e.failed_indices)
                    if self.update_callback and e.completed:
                        self.update_callback(e.completed)
                else:
                    failed_count += len(future_to_chunk[future])

        # Never return a mixed-language result as complete. Recovery output is
        # handled by the API layer, but distinguish provider errors from local
        # quality-gate rejection in the user-facing failure reason.
        if failed_count > 0 and total_segments > 0:
            fail_rate = failed_count / total_segments
            detail = f" First error: {failed_errors[0]}" if failed_errors else ""
            quality_markers = (
                "failed validation",
                "translation incomplete",
                "missing keys",
                "untranslated",
                "placeholder",
                "single item translation failed",
                "boundary",
                "preserve model names",
            )
            quality_failure = any(
                marker in error.lower()
                for error in failed_errors
                for marker in quality_markers
            )
            guidance = (
                "The translation provider responded, but the result did not pass subtitle "
                "quality validation."
                if quality_failure
                else "Check your API key, model limits, and network connection."
            )
            raise RuntimeError(
                f"Translation failed: {failed_count}/{total_segments} segments failed "
                f"({fail_rate:.0%}). {guidance}"
                + detail
            )

        return translated_list

    def _validate_translated_list(
        self,
        source_list: List[SubtitleProcessData],
        translated_list: List[SubtitleProcessData],
    ) -> None:
        """Reject incomplete translation results before writing subtitles."""
        translated_by_index = {}
        duplicates: list[str] = []
        for item in translated_list:
            if item.index in translated_by_index:
                duplicates.append(str(item.index))
            translated_by_index[item.index] = item

        missing: list[str] = []
        empty: list[str] = []
        placeholders: list[str] = []
        untranslated: list[str] = []
        for source in source_list:
            translated = translated_by_index.get(source.index)
            if translated is None:
                missing.append(str(source.index))
            elif not translated.translated_text.strip():
                empty.append(str(source.index))
            else:
                output = translated.translated_text.strip()
                if self._looks_like_placeholder_translation(output):
                    placeholders.append(str(source.index))
                if self._is_untranslated_output(output, source.original_text):
                    untranslated.append(str(source.index))

        if not missing and not empty and not duplicates and not placeholders and not untranslated:
            return

        parts = []
        if missing:
            parts.append(f"missing indices: {missing[:20]}")
        if empty:
            parts.append(f"empty translations: {empty[:20]}")
        if duplicates:
            parts.append(f"duplicate indices: {duplicates[:20]}")
        if placeholders:
            parts.append(f"placeholder translations: {placeholders[:20]}")
        if untranslated:
            parts.append(f"untranslated indices: {untranslated[:20]}")
        raise RuntimeError(
            "Translation incomplete; refusing to save mixed source/target subtitles ("
            + "; ".join(parts)
            + ")"
        )

    @staticmethod
    def _looks_like_placeholder_translation(text: str) -> bool:
        """Detect LLM notes that are not actual translations."""
        text = str(text or "").strip()
        if not text:
            return True
        compact = re.sub(r"\s+", "", text).strip("()（）[]【】<>《》“”\"'。，、；;：:！!?")
        previous_refs = r"上一句|上句|上一条|上条|前一句|前一条|前文|前面"
        placeholder_patterns = [
            r"(?:此|本)句.*(?:合并|并入|省略|略去|无需翻译|不单独翻译).*",
            rf"(?:已)?(?:合并|并入|接上|延续|已译|包含).*(?:{previous_refs})",
            rf"(?:{previous_refs}).*(?:合并|包含|已译|并入|已经翻译)",
            r"(?:最终版本|最终字幕).*(?:合并|省略)",
            r"(?:内容)?(?:同上|见上|略|省略|无需翻译|不单独翻译)",
            r"merged(?:with|into)?(?:the)?(?:previous|above)",
            r"sameasabove",
            r"omitted",
        ]
        if any(
            re.fullmatch(pattern, compact, flags=re.IGNORECASE) for pattern in placeholder_patterns
        ):
            return True
        meta_note = re.compile(
            r"(?:\(|（|\[|【)\s*(?:应为|疑似|译注|注\s*[:：]|原文(?:应为)?|可能是)"
            r"[^\)）\]】]*(?:\)|）|\]|】)",
            flags=re.IGNORECASE,
        )
        return bool(meta_note.search(text))

    def _is_untranslated_output(self, output: str, source: str) -> bool:
        target_patterns = {
            TargetLanguage.SIMPLIFIED_CHINESE: r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
            TargetLanguage.TRADITIONAL_CHINESE: r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
            TargetLanguage.CANTONESE: r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
            TargetLanguage.JAPANESE: r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff]",
            TargetLanguage.KOREAN: r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff]",
        }
        target_pattern = target_patterns.get(self.target_language)
        if target_pattern is None:
            return False
        if re.search(target_pattern, output):
            return False

        # Broad CJK presence is insufficient: unchanged Korean is not a valid
        # Simplified Chinese translation, and vice versa.
        if re.search(
            r"[\u3040-\u30ff\u31f0-\u31ff\u1100-\u11ff\u3130-\u318f"
            r"\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff\u3400-\u4dbf"
            r"\u4e00-\u9fff\uf900-\ufaff]",
            source,
        ):
            return True

        source_words = re.findall(r"[A-Za-z]+", source)
        if not source_words:
            # Numbers, symbols, and punctuation can legitimately be identical.
            return False
        source_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+#&/-]*", source)

        def is_identifier_like(token: str) -> bool:
            token = token.strip(".")
            letters = re.sub(r"[^A-Za-z]", "", token)
            return bool(
                re.search(r"\d", token)
                or (len(letters) >= 2 and letters.isupper())
                or re.search(r"[a-z][A-Z]", letters)
                or re.search(r"[.+#&/-]", token)
            )

        # Acronyms and product identifiers can legitimately remain in Latin
        # script. Merely title-cased words such as "Area" or "Okay" cannot.
        if source_tokens and len(source_tokens) <= 3 and all(
            is_identifier_like(token) for token in source_tokens
        ):
            return False
        return bool(source_words)

    def _get_cache_key(self, chunk: List[SubtitleProcessData]) -> str:
        """生成缓存键"""
        class_name = self.__class__.__name__
        chunk_key = generate_cache_key(chunk)
        lang = self.target_language.value
        return f"{class_name}:{chunk_key}:{lang}"

    def _safe_translate_chunk(self, chunk: List[SubtitleProcessData]) -> List[SubtitleProcessData]:
        """安全的翻译块"""
        try:
            cache_key = self._get_cache_key(chunk)
            if self.use_cache and is_cache_enabled():
                try:
                    cached_result = self._cache.get(cache_key, default=None)
                except Exception:
                    # Graceful degradation: corrupted cache (e.g. old pickle from app rename)
                    cached_result = None
                    self._cache.delete(cache_key)
                if cached_result is not None:
                    if isinstance(cached_result, list) and all(
                        isinstance(item, SubtitleProcessData) for item in cached_result
                    ):
                        try:
                            self._validate_translated_list(chunk, cached_result)
                        except RuntimeError:
                            logger.warning(
                                "Discarding invalid translation cache entry: %s", cache_key
                            )
                            self._cache.delete(cache_key)
                        else:
                            return cached_result
                    else:
                        logger.warning("Discarding invalid translation cache entry: %s", cache_key)
                        self._cache.delete(cache_key)

            result = self._translate_chunk(chunk)
            self._validate_translated_list(chunk, result)

            if self.update_callback:
                self.update_callback(result)

            if self.use_cache and is_cache_enabled():
                self._cache.set(cache_key, result, expire=86400 * 7)
            return result

        except Exception as e:
            logger.error(f"Translation chunk failed with error: {type(e).__name__}: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    @staticmethod
    def _set_segments_translated_text(
        original_segments: List[ASRDataSeg], translated_list: List[SubtitleProcessData]
    ) -> List[ASRDataSeg]:
        """设置Subtitle segment的翻译文本"""
        # 创建索引到翻译文本的映射
        translation_map = {data.index: data.translated_text for data in translated_list}

        for i, seg in enumerate(original_segments, 1):
            if i not in translation_map:
                logger.error(f"Subtitle segment {i} has no translation")
                continue
            seg.translated_text = translation_map[i]

        return original_segments

    @abstractmethod
    def _translate_chunk(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """翻译字幕块"""
        pass

    def stop(self):
        """停止翻译器"""
        if not self.is_running:
            return

        self.is_running = False
        if hasattr(self, "executor") and self.executor is not None:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.error(f"Error closing thread pool: {str(e)}")
            finally:
                self.executor = None
