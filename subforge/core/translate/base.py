"""翻译器基类"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

from subforge.core.asr.asr_data import ASRData, ASRDataSeg
from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.quality import (
    inspect_translation_batch,
    is_placeholder_translation,
    is_untranslated_output,
)
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
            translated_list = self._finalize_translated_list(
                translate_data_list,
                translated_list,
            )
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

    def _finalize_translated_list(
        self,
        source_list: List[SubtitleProcessData],
        translated_list: List[SubtitleProcessData],
    ) -> List[SubtitleProcessData]:
        """Allow translators to run whole-document consistency checks."""
        return translated_list

    def _is_chunk_result_stable(self, translated_list: List[SubtitleProcessData]) -> bool:
        """Return whether a provisional chunk is safe to preview and cache."""
        return True

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
                    if (
                        self.update_callback
                        and e.completed
                        and self._is_chunk_result_stable(e.completed)
                    ):
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
                marker in error.lower() for error in failed_errors for marker in quality_markers
            )
            guidance = (
                "The translation provider responded, but the result did not pass subtitle "
                "quality validation."
                if quality_failure
                else "Check your API key, model limits, and network connection."
            )
            raise RuntimeError(
                f"Translation failed: {failed_count}/{total_segments} segments failed "
                f"({fail_rate:.0%}). {guidance}" + detail
            )

        return translated_list

    def _validate_translated_list(
        self,
        source_list: List[SubtitleProcessData],
        translated_list: List[SubtitleProcessData],
    ) -> None:
        """Reject incomplete translation results before writing subtitles."""
        report = inspect_translation_batch(
            source_list,
            translated_list,
            self.target_language,
        )
        if report.valid:
            return
        raise RuntimeError(
            "Translation incomplete; refusing to save mixed source/target subtitles ("
            + report.error_detail()
            + ")"
        )

    @staticmethod
    def _looks_like_placeholder_translation(text: str) -> bool:
        """Detect LLM notes that are not actual translations."""
        return is_placeholder_translation(text)

    def _is_untranslated_output(self, output: str, source: str) -> bool:
        return is_untranslated_output(output, source, self.target_language)

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

            result_is_stable = self._is_chunk_result_stable(result)
            if self.update_callback and result_is_stable:
                self.update_callback(result)

            if self.use_cache and result_is_stable and is_cache_enabled():
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
