import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.blocking import run_blocking
from app.core.task_manager import TaskResourceBusyError, task_manager
from app.security import validate_path
from app.services.task_runtime import create_pipeline_context, schedule_background_task
from subforge.application import subtitle_preview_segments

router = APIRouter()
logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task] = set()


def _result_path(input_file: str, suffix: str, configured_work_dir: str = "") -> Path:
    """Choose a durable output path without changing the existing filename contract."""
    source = Path(input_file).resolve()
    output_dir = source.parent
    if configured_work_dir.strip():
        try:
            output_dir = validate_path(configured_work_dir.strip())
        except ValueError as exc:
            raise RuntimeError("Configured output folder is outside allowed roots") from exc
        if not output_dir.is_dir():
            raise RuntimeError("Configured output folder does not exist")
    else:
        from app.api.files import UPLOAD_ROOT

        if source.is_relative_to(UPLOAD_ROOT.resolve()):
            from subforge.config import WORK_PATH

            output_dir = WORK_PATH.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{source.stem}{suffix}.srt"


class SubtitleRequest(BaseModel):
    subtitle_file: str = Field(max_length=4096)
    media_file: str | None = Field(default=None, max_length=4096)
    target_language: Literal[
        "chinese",
        "english",
        "japanese",
        "korean",
        "french",
        "german",
        "spanish",
        "portuguese",
        "russian",
        "cantonese",
        "thai",
        "vietnamese",
        "indonesian",
        "malay",
        "tagalog",
        "italian",
        "dutch",
        "polish",
        "turkish",
        "swedish",
        "ukrainian",
        "arabic",
    ] = "chinese"
    translator: Literal["llm", "bing", "google", "deeplx"] = "bing"
    need_optimize: bool = True
    need_translate: bool = True
    need_reflect: bool = False
    llm_provider: str = Field(default="", max_length=64)
    llm_model: str = Field(default="", max_length=256)
    custom_prompt: str | None = Field(default=None, max_length=100_000)


def _validate_expected_llm_config(req: SubtitleRequest, runtime) -> None:
    uses_llm = req.need_optimize or (req.need_translate and req.translator == "llm")
    if req.llm_provider.strip() and req.llm_provider.strip() != runtime.provider:
        raise ValueError(
            f"LLM 服务已从 {req.llm_provider.strip()} 切换为 {runtime.provider}，"
            "请刷新页面后重新开始任务。"
        )
    if uses_llm and (not runtime.base_url or not runtime.api_key or not runtime.model):
        raise ValueError("LLM 配置不完整，请先在设置中选择模型并完成连接测试。")
    if req.llm_model.strip() and req.llm_model.strip() != runtime.model:
        raise ValueError(
            f"LLM 模型已从 {req.llm_model.strip()} 切换为 {runtime.model}，"
            "请刷新页面后重新开始任务。"
        )


def _resolve_custom_prompt(request_prompt: str | None, persisted_prompt: str) -> str:
    """Treat an explicitly cleared task prompt as authoritative."""
    return persisted_prompt if request_prompt is None else request_prompt


def _apply_translation_preview(asr_data, result, translated_indices: set[int]) -> int:
    """Apply usable translations and return the unique completed-item count."""
    from subforge.core.translate.base import BaseTranslator

    for item in result:
        idx = int(item.index) - 1
        if (
            0 <= idx < len(asr_data.segments)
            and item.translated_text
            and not BaseTranslator._looks_like_placeholder_translation(item.translated_text)
        ):
            asr_data.segments[idx].translated_text = item.translated_text
            translated_indices.add(idx)
    return len(translated_indices)


@router.post("/start")
async def start_subtitle_processing(req: SubtitleRequest):
    try:
        file_path = validate_path(req.subtitle_file)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=400, detail="Subtitle file not found")
    updates = {"subtitle_file": str(file_path)}
    if req.media_file:
        try:
            media_path = validate_path(req.media_file)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        if not media_path.is_file():
            raise HTTPException(status_code=400, detail="Media file not found")
        updates["media_file"] = str(media_path)
    req = req.model_copy(update=updates)

    from app.api.config import get_llm_runtime_config

    try:
        _validate_expected_llm_config(req, get_llm_runtime_config())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        task_id = schedule_background_task(
            task_type="subtitle",
            resource_key=f"subtitle:{file_path.resolve()}",
            runner=lambda current_task_id: _run_subtitle(current_task_id, req),
            background_tasks=_background_tasks,
        )
    except TaskResourceBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"task_id": task_id, "status": "started"}


async def _run_subtitle(task_id: str, req: SubtitleRequest):
    import tempfile
    from pathlib import Path as PathLib

    partial_srt_path = None
    asr_data = None
    llm_client = None
    llm_cancel_callback = None
    timing_speech_segments: list[tuple[int, int]] = []
    timing_media_duration_ms: int | None = None
    pipeline_warnings: list[str] = []
    context = create_pipeline_context(task_id)
    try:
        from app.api.config import get_config_value, get_llm_runtime_config

        custom_prompt = _resolve_custom_prompt(
            req.custom_prompt,
            get_config_value("custom_prompt", ""),
        )

        # Snapshot the complete provider profile again inside the worker. This
        # closes the small race where settings change after task scheduling.
        llm_runtime = get_llm_runtime_config()
        _validate_expected_llm_config(req, llm_runtime)
        llm_model = llm_runtime.model
        logger.info(
            "Subtitle task %s using LLM provider=%s model=%s",
            task_id,
            llm_runtime.provider,
            llm_model,
        )

        # Build an explicit LLM client for this task. This avoids global
        # OPENAI_* environment mutation and prevents concurrent subtitle tasks
        # from using each other's credentials or base URLs.
        api_key = llm_runtime.api_key
        base_url = llm_runtime.base_url
        if api_key and base_url:
            from subforge.core.llm import cancel_client_requests, create_client
            from subforge.core.llm.client import set_client_log_context
            from subforge.core.llm.request_logger import set_llm_log_level

            set_llm_log_level(get_config_value("llm_log_level", "summary"))
            llm_client = create_client(base_url=base_url, api_key=api_key)
            set_client_log_context(
                llm_client,
                task_id=task_id,
                file_name=Path(req.subtitle_file).name,
            )
            def _cancel_llm_requests() -> None:
                cancel_client_requests(llm_client)

            llm_cancel_callback = _cancel_llm_requests
            task_manager.register_cancel_callback(task_id, llm_cancel_callback)

        from subforge.core.asr.asr_data import ASRData
        from subforge.core.optimize.optimize import SubtitleOptimizer
        from subforge.core.split.split import SubtitleSplitter
        from subforge.core.subtitle.validation import (
            lock_source_segments,
            validate_bilingual_result,
        )
        from subforge.core.translate.context import TranslationContext, build_translation_context
        from subforge.core.translate.factory import TranslatorFactory
        from subforge.core.translate.types import TargetLanguage

        context.report(5, "Loading subtitle file...")
        context.checkpoint()

        # Partial SRT file for real-time preview
        partial_srt = tempfile.NamedTemporaryFile(suffix="_partial.srt", delete=False)
        partial_srt_path = partial_srt.name
        partial_srt.close()
        preview_lock = threading.RLock()
        last_preview_save = [0.0]
        last_recovery_save = [0.0]

        def _save_partial(data, msg="", *, force=False):
            """Save current ASRData to partial SRT and notify frontend."""
            try:
                with preview_lock:
                    now = time.monotonic()
                    if not force and now - last_preview_save[0] < 0.25:
                        return
                    last_preview_save[0] = now
                    snapshot_due = force or now - last_recovery_save[0] >= 5.0
                    snapshot_path = None
                    if snapshot_due:
                        from subforge.core.entities import SubtitleLayoutEnum

                        data.save(
                            partial_srt_path,
                            layout=SubtitleLayoutEnum.TRANSLATE_ON_TOP,
                            speaker_style="none",
                        )
                        last_recovery_save[0] = now
                        snapshot_path = partial_srt_path
                    context.publish_preview(
                        subtitle_preview_segments(data),
                        subtitle_file=snapshot_path,
                        message=msg or None,
                    )
            except Exception as e:
                logger.warning(f"Failed to save partial result: {e}")

        # Load subtitle into ASRData
        asr_data = ASRData.from_subtitle_file(req.subtitle_file)
        timing_speech_segments, timing_media_duration_ms = ASRData.load_timing_metadata(
            req.subtitle_file
        )
        if any(segment.speaker_id for segment in asr_data.segments):
            from subforge.core.asr.speaker_diarization import smooth_speaker_assignments

            smooth_speaker_assignments(asr_data)
        context.report(10, f"Loaded {len(asr_data.segments)} segments")
        context.checkpoint()

        # Split long segments
        context.report(15, "Splitting subtitle segments...")
        thread_num = get_config_value("thread_num", 3)
        batch_size = get_config_value("batch_size", 10)
        max_word_count_cjk = get_config_value("max_word_count_cjk", 25)
        max_word_count_english = get_config_value("max_word_count_english", 18)

        async def _run_stage(instance, function, *args):
            cancel = getattr(instance, "cancel", None)
            if not callable(cancel):
                cancel = getattr(instance, "stop", None)
            with context.cancellation_scope(cancel if callable(cancel) else None):
                return await run_blocking(
                    function,
                    *args,
                    on_cancel=cancel if callable(cancel) else None,
                )

        def _on_split_progress(segments):
            partial = ASRData(list(segments))
            _save_partial(partial, f"Splitting subtitles... {len(segments)} ready")

        splitter = SubtitleSplitter(
            thread_num=thread_num,
            model=llm_model,
            llm_client=llm_client,
            target_language=req.target_language if req.need_translate else "",
            max_word_count_cjk=max_word_count_cjk,
            max_word_count_english=max_word_count_english,
            update_callback=_on_split_progress,
        )
        asr_data = await _run_stage(splitter, splitter.split_subtitle, asr_data)
        context.checkpoint()
        split_fallback_count = int(getattr(splitter, "fallback_count", 0) or 0)
        if split_fallback_count:
            pipeline_warnings.append(
                f"Smart splitting used rule fallback for {split_fallback_count} batch(es)."
            )
        _save_partial(
            asr_data,
            f"Split into {len(asr_data.segments)} segments",
            force=True,
        )
        context.report(25, f"Split into {len(asr_data.segments)} segments")

        # Optimize subtitles
        if req.need_optimize:
            context.report(30, "Optimizing subtitles...")

            optimize_count = [0]

            def _on_optimize_progress(result):
                with preview_lock:
                    for item in result:
                        idx = int(item.index) - 1
                        if 0 <= idx < len(asr_data.segments) and item.optimized_text:
                            asr_data.segments[idx].text = item.optimized_text
                    optimize_count[0] = min(
                        len(asr_data.segments),
                        optimize_count[0] + len(result),
                    )
                    pct = (
                        30 + int(30 * optimize_count[0] / len(asr_data.segments))
                        if len(asr_data.segments) > 0
                        else 30
                    )
                    context.report(
                        min(pct, 60),
                        f"Optimized {optimize_count[0]}/{len(asr_data.segments)}...",
                    )
                    _save_partial(asr_data)

            optimizer = SubtitleOptimizer(
                thread_num=thread_num,
                batch_num=batch_size,
                model=llm_model,
                custom_prompt=custom_prompt,
                update_callback=_on_optimize_progress,
                use_cache=False,
                llm_client=llm_client,
            )
            asr_data = await _run_stage(optimizer, optimizer.optimize_subtitle, asr_data)
            context.checkpoint()
            optimize_failure_count = int(
                getattr(optimizer, "failed_batch_count", 0) or 0
            )
            if optimize_failure_count:
                pipeline_warnings.append(
                    f"Subtitle optimization kept the source for "
                    f"{optimize_failure_count} failed batch(es)."
                )
            _save_partial(asr_data, "Optimization complete", force=True)
            context.report(60, "Optimization complete")

        # Translate subtitles
        if req.need_translate:
            context.report(65, "Translating subtitles...")

            # Map frontend language values to TargetLanguage enum
            lang_map = {
                "chinese": TargetLanguage.SIMPLIFIED_CHINESE,
                "english": TargetLanguage.ENGLISH,
                "japanese": TargetLanguage.JAPANESE,
                "korean": TargetLanguage.KOREAN,
                "french": TargetLanguage.FRENCH,
                "german": TargetLanguage.GERMAN,
                "spanish": TargetLanguage.SPANISH,
                "portuguese": TargetLanguage.PORTUGUESE,
                "russian": TargetLanguage.RUSSIAN,
                "cantonese": TargetLanguage.CANTONESE,
                "thai": TargetLanguage.THAI,
                "vietnamese": TargetLanguage.VIETNAMESE,
                "indonesian": TargetLanguage.INDONESIAN,
                "malay": TargetLanguage.MALAY,
                "tagalog": TargetLanguage.TAGALOG,
                "italian": TargetLanguage.ITALIAN,
                "dutch": TargetLanguage.DUTCH,
                "polish": TargetLanguage.POLISH,
                "turkish": TargetLanguage.TURKISH,
                "swedish": TargetLanguage.SWEDISH,
                "ukrainian": TargetLanguage.UKRAINIAN,
                "arabic": TargetLanguage.ARABIC,
            }
            target_lang = lang_map[req.target_language]

            translated_indices: set[int] = set()

            def _on_translate_progress(result):
                with preview_lock:
                    translate_count = _apply_translation_preview(
                        asr_data,
                        result,
                        translated_indices,
                    )
                    pct = (
                        65 + int(25 * translate_count / len(asr_data.segments))
                        if len(asr_data.segments) > 0
                        else 65
                    )
                    context.report(
                        min(pct, 90),
                        f"Translated {translate_count}/{len(asr_data.segments)}...",
                    )
                    _save_partial(asr_data)

            from subforge.core.translate.types import TranslatorType

            type_map = {
                "llm": TranslatorType.OPENAI,
                "bing": TranslatorType.BING,
                "google": TranslatorType.GOOGLE,
                "deeplx": TranslatorType.DEEPLX,
            }
            translator_type = type_map[req.translator]
            translation_context = TranslationContext(custom_prompt=custom_prompt)
            if translator_type == TranslatorType.OPENAI:
                context.report(63, "Generating translation context...")
                if asr_data is None:
                    raise RuntimeError("Subtitle data is unavailable before translation")
                translation_source: ASRData = asr_data
                translation_context = await run_blocking(
                    lambda: build_translation_context(
                        translation_source,
                        model=llm_model,
                        target_language=target_lang,
                        custom_prompt=custom_prompt,
                        use_cache=False,
                        llm_client=llm_client,
                    ),
                )
                context.checkpoint()
                context.report(65, "Translating subtitles...")
            source_lock = lock_source_segments(asr_data)
            translator = TranslatorFactory.create_translator(
                translator_type=translator_type,
                thread_num=thread_num,
                batch_num=batch_size,
                target_language=target_lang,
                model=llm_model,
                custom_prompt=custom_prompt,
                is_reflect=req.need_reflect,
                update_callback=_on_translate_progress,
                use_cache=False,
                translation_context=translation_context,
                llm_client=llm_client,
                azure_translator_key=get_config_value("azure_translator_key", ""),
                azure_translator_region=get_config_value("azure_translator_region", ""),
                azure_translator_endpoint=get_config_value(
                    "azure_translator_endpoint",
                    "https://api.cognitive.microsofttranslator.com",
                ),
            )
            asr_data = await _run_stage(translator, translator.translate_subtitle, asr_data)
            context.checkpoint()
            context.report(90, "Translation complete")

            context.report(95, "Validating bilingual subtitles...")
            validate_bilingual_result(asr_data, source_lock)
            if req.target_language.lower() in {"chinese", "cantonese"} and bool(
                get_config_value("replace_chinese_punctuation", True)
            ):
                context.report(96, "Finalizing subtitle formatting...")
                asr_data.replace_chinese_translation_punctuation()

        # Sentence cues are assembled from exact word boundaries. Preserve the
        # atomic timing data, but add a small display tail and use high-confidence
        # VAD only when the matching source media is still available.
        context.report(96, "Refining subtitle timing...")
        if req.media_file and not timing_speech_segments:
            subtitle_stem = Path(req.subtitle_file).stem
            for suffix in ("_processed", "_recovery"):
                if subtitle_stem.endswith(suffix):
                    subtitle_stem = subtitle_stem[: -len(suffix)]
            if Path(req.media_file).stem == subtitle_stem:
                try:
                    from subforge.core.asr.audio_analysis import AudioAnalysisContext

                    analysis = AudioAnalysisContext(req.media_file)
                    def _analyze_timing():
                        return (
                            analysis.speech_segments(
                                threshold=0.75,
                                min_speech_ms=120,
                                min_silence_ms=200,
                                speech_pad_ms=0,
                            ),
                            len(analysis.audio_segment()),
                        )

                    timing_speech_segments, timing_media_duration_ms = await run_blocking(
                        _analyze_timing
                    )
                except Exception as exc:
                    logger.warning(
                        "Subtitle timing VAD failed for %s; using safe tail padding only: %s",
                        req.media_file,
                        exc,
                    )
        asr_data.extend_sentence_tails_conservatively(
            timing_speech_segments,
            media_duration_ms=timing_media_duration_ms,
        )

        # Save result
        context.report(97, "Saving subtitle file...")
        output_path = _result_path(
            req.subtitle_file,
            "_processed",
            str(get_config_value("work_dir", "") or ""),
        )

        from subforge.core.entities import SubtitleLayoutEnum

        layout = SubtitleLayoutEnum.TRANSLATE_ON_TOP  # Chinese on top, English on bottom
        context.checkpoint()
        await run_blocking(
            lambda: asr_data.save(
                str(output_path),
                layout=layout,
                speaker_style="none",
            )
        )

        context.report(99, "Updating subtitle editor...")
        final_preview = subtitle_preview_segments(asr_data)
        context.publish_preview(
            final_preview,
            subtitle_file=str(output_path),
            message="Subtitle file saved",
        )
        final_preview_revision = task_manager.get_preview_revision(task_id)
        completion_message = "Done with processing warnings" if pipeline_warnings else "Done"
        context.report(100, completion_message)
        task_manager.complete_task(
            task_id,
            {
                "subtitle_file": str(output_path),
                "preview_revision": final_preview_revision,
                "segments": final_preview,
                "warnings": pipeline_warnings,
            },
        )

    except asyncio.CancelledError:
        logger.info("Subtitle task %s cancelled", task_id)
        raise
    except Exception as e:
        logger.exception("Subtitle task %s failed", task_id)
        recovery_path = None
        if asr_data is not None and any(
            (segment.translated_text or "").strip() for segment in asr_data.segments
        ):
            try:
                if req.target_language.lower() in {"chinese", "cantonese"} and bool(
                    get_config_value("replace_chinese_punctuation", True)
                ):
                    asr_data.replace_chinese_translation_punctuation()
                asr_data.extend_sentence_tails_conservatively(
                    timing_speech_segments,
                    media_duration_ms=timing_media_duration_ms,
                )
                recovery_path = _result_path(
                    req.subtitle_file,
                    "_recovery",
                    str(get_config_value("work_dir", "") or ""),
                )
                from subforge.core.entities import SubtitleLayoutEnum

                await run_blocking(
                    lambda: asr_data.save(
                        str(recovery_path),
                        layout=SubtitleLayoutEnum.TRANSLATE_ON_TOP,
                        speaker_style="none",
                    )
                )
                task = task_manager.get_task(task_id)
                context.report(
                    task.progress if task else 0,
                    "Translation failed; partial result saved",
                    subtitle_file=str(recovery_path),
                )
                context.publish_preview(
                    subtitle_preview_segments(asr_data),
                    subtitle_file=str(recovery_path),
                    message="Translation failed; partial result saved",
                )
                logger.warning(
                    "Saved recoverable subtitle result for failed task %s: %s",
                    task_id,
                    recovery_path,
                )
            except Exception:
                logger.exception("Failed to save subtitle recovery file for task %s", task_id)
                recovery_path = None
        failure_result = {}
        if recovery_path:
            failure_result["recovery_file"] = str(recovery_path)
        if pipeline_warnings:
            failure_result["warnings"] = pipeline_warnings
        task_manager.fail_task(task_id, str(e), failure_result or None)
    finally:
        if llm_cancel_callback is not None:
            task_manager.unregister_cancel_callback(task_id, llm_cancel_callback)
        if llm_client is not None:
            from subforge.core.llm import close_client

            close_client(llm_client)
        if partial_srt_path:
            try:
                PathLib(partial_srt_path).unlink(missing_ok=True)
            except Exception:
                pass
