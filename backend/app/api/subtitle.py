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


def _preview_segments(data) -> list[dict]:
    from subforge.core.translate.base import BaseTranslator

    return [
        {
            "id": index,
            "start": segment._ms_to_srt_time(segment.start_time),
            "end": segment._ms_to_srt_time(segment.end_time),
            "text": segment.text,
            "translated": ""
            if BaseTranslator._looks_like_placeholder_translation(segment.translated_text or "")
            else segment.translated_text or "",
        }
        for index, segment in enumerate(data.segments, 1)
    ]


def _raise_if_cancelled(task_id: str) -> None:
    if task_manager.is_cancelled(task_id):
        raise asyncio.CancelledError()


class SubtitleRequest(BaseModel):
    subtitle_file: str = Field(max_length=4096)
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
    llm_model: str = Field(default="", max_length=256)
    custom_prompt: str = Field(default="", max_length=100_000)


@router.post("/start")
async def start_subtitle_processing(req: SubtitleRequest):
    try:
        file_path = validate_path(req.subtitle_file)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=400, detail="Subtitle file not found")
    req = req.model_copy(update={"subtitle_file": str(file_path)})

    try:
        task = task_manager.create_task(
            "subtitle",
            resource_key=f"subtitle:{file_path.resolve()}",
        )
    except TaskResourceBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    task_obj = asyncio.create_task(_run_subtitle(task.id, req))
    task_manager.register_running_task(task.id, task_obj)
    _background_tasks.add(task_obj)
    task_obj.add_done_callback(_background_tasks.discard)
    task_obj.add_done_callback(lambda _task: task_manager.unregister_running_task(task.id))
    return {"task_id": task.id, "status": "started"}


async def _run_subtitle(task_id: str, req: SubtitleRequest):
    import tempfile
    from pathlib import Path as PathLib

    partial_srt_path = None
    asr_data = None
    try:
        from app.api.config import get_config_value

        custom_prompt = req.custom_prompt or get_config_value("custom_prompt", "")

        # Use current config's model unless explicitly overridden by request.
        llm_model = (req.llm_model or "").strip() or get_config_value("llm_model", "gpt-4o-mini")

        # Build an explicit LLM client for this task. This avoids global
        # OPENAI_* environment mutation and prevents concurrent subtitle tasks
        # from using each other's credentials or base URLs.
        api_key = get_config_value("llm_api_key", "")
        base_url = get_config_value("llm_base_url", "")
        llm_client = None
        if api_key and base_url:
            from subforge.core.llm import create_client
            from subforge.core.llm.client import set_client_log_context
            from subforge.core.llm.request_logger import set_llm_log_level

            set_llm_log_level(get_config_value("llm_log_level", "summary"))
            llm_client = create_client(base_url=base_url, api_key=api_key)
            set_client_log_context(
                llm_client,
                task_id=task_id,
                file_name=Path(req.subtitle_file).name,
            )

        from subforge.core.asr.asr_data import ASRData
        from subforge.core.optimize.optimize import SubtitleOptimizer
        from subforge.core.split.split import SubtitleSplitter
        from subforge.core.subtitle.validation import (
            lock_source_segments,
            validate_bilingual_result,
        )
        from subforge.core.translate.base import BaseTranslator
        from subforge.core.translate.context import TranslationContext, build_translation_context
        from subforge.core.translate.factory import TranslatorFactory
        from subforge.core.translate.types import TargetLanguage

        task_manager.update_progress(task_id, 5, "Loading subtitle file...")
        _raise_if_cancelled(task_id)

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
                    task_manager.publish_preview(
                        task_id,
                        _preview_segments(data),
                        subtitle_file=snapshot_path,
                        message=msg or None,
                    )
            except Exception as e:
                logger.warning(f"Failed to save partial result: {e}")

        # Load subtitle into ASRData
        asr_data = ASRData.from_subtitle_file(req.subtitle_file)
        task_manager.update_progress(task_id, 10, f"Loaded {len(asr_data.segments)} segments")
        _raise_if_cancelled(task_id)

        # Split long segments
        task_manager.update_progress(task_id, 15, "Splitting subtitle segments...")
        thread_num = get_config_value("thread_num", 3)
        batch_size = get_config_value("batch_size", 10)

        async def _run_stage(instance, function, *args):
            stop = getattr(instance, "stop", None)
            if callable(stop):
                task_manager.register_cancel_callback(task_id, stop)
            try:
                return await run_blocking(
                    function,
                    *args,
                    on_cancel=stop if callable(stop) else None,
                )
            finally:
                if callable(stop):
                    task_manager.unregister_cancel_callback(task_id, stop)

        def _on_split_progress(segments):
            partial = ASRData(list(segments))
            _save_partial(partial, f"Splitting subtitles... {len(segments)} ready")

        splitter = SubtitleSplitter(
            thread_num=thread_num,
            model=llm_model,
            llm_client=llm_client,
            update_callback=_on_split_progress,
        )
        asr_data = await _run_stage(splitter, splitter.split_subtitle, asr_data)
        _raise_if_cancelled(task_id)
        _save_partial(
            asr_data,
            f"Split into {len(asr_data.segments)} segments",
            force=True,
        )
        task_manager.update_progress(task_id, 25, f"Split into {len(asr_data.segments)} segments")

        # Optimize subtitles
        if req.need_optimize:
            task_manager.update_progress(task_id, 30, "Optimizing subtitles...")

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
                    task_manager.update_progress(
                        task_id,
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
            _raise_if_cancelled(task_id)
            _save_partial(asr_data, "Optimization complete", force=True)
            task_manager.update_progress(task_id, 60, "Optimization complete")

        # Translate subtitles
        if req.need_translate:
            task_manager.update_progress(task_id, 65, "Translating subtitles...")

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

            translate_count = [0]

            def _on_translate_progress(result):
                with preview_lock:
                    for item in result:
                        idx = int(item.index) - 1
                        if (
                            0 <= idx < len(asr_data.segments)
                            and item.translated_text
                            and not BaseTranslator._looks_like_placeholder_translation(
                                item.translated_text
                            )
                        ):
                            asr_data.segments[idx].translated_text = item.translated_text
                    translate_count[0] = min(
                        len(asr_data.segments),
                        translate_count[0] + len(result),
                    )
                    pct = (
                        65 + int(25 * translate_count[0] / len(asr_data.segments))
                        if len(asr_data.segments) > 0
                        else 65
                    )
                    task_manager.update_progress(
                        task_id,
                        min(pct, 90),
                        f"Translated {translate_count[0]}/{len(asr_data.segments)}...",
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
                task_manager.update_progress(task_id, 63, "Generating translation context...")
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
                _raise_if_cancelled(task_id)
                task_manager.update_progress(task_id, 65, "Translating subtitles...")
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
            )
            asr_data = await _run_stage(translator, translator.translate_subtitle, asr_data)
            _raise_if_cancelled(task_id)
            task_manager.update_progress(task_id, 90, "Translation complete")

            task_manager.update_progress(task_id, 95, "Validating bilingual subtitles...")
            validate_bilingual_result(asr_data, source_lock)
            if req.target_language.lower() in {"chinese", "cantonese"} and bool(
                get_config_value("replace_chinese_punctuation", True)
            ):
                task_manager.update_progress(
                    task_id, 96, "Cleaning Chinese subtitle punctuation..."
                )
                asr_data.replace_chinese_translation_punctuation()
            _save_partial(asr_data, "Bilingual subtitles validated", force=True)

        # Save result
        task_manager.update_progress(task_id, 97, "Saving result...")
        output_path = _result_path(
            req.subtitle_file,
            "_processed",
            str(get_config_value("work_dir", "") or ""),
        )

        from subforge.core.entities import SubtitleLayoutEnum

        layout = SubtitleLayoutEnum.TRANSLATE_ON_TOP  # Chinese on top, English on bottom
        _raise_if_cancelled(task_id)
        await run_blocking(
            lambda: asr_data.save(
                str(output_path),
                layout=layout,
                speaker_style="none",
            )
        )

        task_manager.update_progress(task_id, 100, "Done")
        task_manager.complete_task(
            task_id,
            {
                "subtitle_file": str(output_path),
                "segments": _preview_segments(asr_data),
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
                task_manager.update_progress(
                    task_id,
                    task.progress if task else 0,
                    "Translation failed; partial result saved",
                    subtitle_file=str(recovery_path),
                    preview_segments=_preview_segments(asr_data),
                )
                logger.warning(
                    "Saved recoverable subtitle result for failed task %s: %s",
                    task_id,
                    recovery_path,
                )
            except Exception:
                logger.exception("Failed to save subtitle recovery file for task %s", task_id)
                recovery_path = None
        task_manager.fail_task(
            task_id,
            str(e),
            {"recovery_file": str(recovery_path)} if recovery_path else None,
        )
    finally:
        if partial_srt_path:
            try:
                PathLib(partial_srt_path).unlink(missing_ok=True)
            except Exception:
                pass
