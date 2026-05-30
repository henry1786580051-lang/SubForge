import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.task_manager import task_manager
from app.security import validate_path

router = APIRouter()

_background_tasks: set[asyncio.Task] = set()


class SubtitleRequest(BaseModel):
    subtitle_file: str
    target_language: str = "english"
    translator: str = "bing"
    need_optimize: bool = True
    need_translate: bool = True
    need_reflect: bool = False
    llm_model: str = "mimo-v2.5-pro"
    custom_prompt: str = ""


@router.post("/start")
async def start_subtitle_processing(req: SubtitleRequest):
    try:
        file_path = validate_path(req.subtitle_file)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=400, detail="Subtitle file not found")

    task = task_manager.create_task("subtitle")
    task_obj = asyncio.create_task(_run_subtitle(task.id, req))
    task_manager.register_running_task(task.id, task_obj)
    _background_tasks.add(task_obj)
    task_obj.add_done_callback(_background_tasks.discard)
    return {"task_id": task.id, "status": "started"}


async def _run_subtitle(task_id: str, req: SubtitleRequest):
    import tempfile
    from pathlib import Path as PathLib

    partial_srt_path = None
    try:
        from app.api.config import get_config_value
        from subforge.core.utils.logger import setup_logger
        logger = setup_logger("subtitle_pipeline")

        custom_prompt = req.custom_prompt or get_config_value("custom_prompt", "")

        # Use config's llm_model if not specified in request
        llm_model = req.llm_model or get_config_value("llm_model", "mimo-v2.5-pro")

        # Set env vars for global LLM singleton (used by splitter/optimizer/translator)
        api_key = get_config_value("llm_api_key", "")
        base_url = get_config_value("llm_base_url", "")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url

        # Reset global LLM client to pick up new env vars
        try:
            import subforge.core.llm.client as llm_client_module
            with llm_client_module._client_lock:
                llm_client_module._global_client = None
        except Exception:
            pass

        from subforge.core.asr.asr_data import ASRData
        from subforge.core.optimize.optimize import SubtitleOptimizer
        from subforge.core.split.split import SubtitleSplitter
        from subforge.core.translate.factory import TranslatorFactory
        from subforge.core.translate.types import TargetLanguage

        task_manager.update_progress(task_id, 5, "Loading subtitle file...")

        # Partial SRT file for real-time preview
        partial_srt = tempfile.NamedTemporaryFile(suffix="_partial.srt", delete=False)
        partial_srt_path = partial_srt.name
        partial_srt.close()

        def _save_partial(data, msg=""):
            """Save current ASRData to partial SRT and notify frontend."""
            try:
                from subforge.core.asr.asr_data import SubtitleLayoutEnum
                data.save(partial_srt_path, layout=SubtitleLayoutEnum.TRANSLATE_ON_TOP)
                task = task_manager.get_task(task_id)
                if task:
                    task_manager.update_progress(task_id, task.progress, msg, subtitle_file=partial_srt_path)
            except Exception as e:
                logger.warning(f"Failed to save partial result: {e}")

        # Load subtitle into ASRData
        asr_data = ASRData.from_subtitle_file(req.subtitle_file)
        task_manager.update_progress(task_id, 10, f"Loaded {len(asr_data.segments)} segments")

        # Split long segments
        task_manager.update_progress(task_id, 15, "Splitting subtitle segments...")
        loop = asyncio.get_event_loop()
        thread_num = get_config_value("thread_num", 3)
        batch_size = get_config_value("batch_size", 10)
        splitter = SubtitleSplitter(thread_num=thread_num, model=req.llm_model)
        asr_data = await loop.run_in_executor(None, splitter.split_subtitle, asr_data)
        _save_partial(asr_data, f"Split into {len(asr_data.segments)} segments")
        task_manager.update_progress(task_id, 25, f"Split into {len(asr_data.segments)} segments")

        # Optimize subtitles
        if req.need_optimize:
            task_manager.update_progress(task_id, 30, "Optimizing subtitles...")

            optimize_count = [0]
            def _on_optimize_progress(result):
                optimize_count[0] += len(result)
                pct = 30 + int(30 * optimize_count[0] / len(asr_data.segments)) if len(asr_data.segments) > 0 else 30
                task_manager.update_progress(task_id, min(pct, 60), f"Optimized {optimize_count[0]}/{len(asr_data.segments)}...")
                _save_partial(asr_data)

            optimizer = SubtitleOptimizer(
                thread_num=thread_num,
                batch_num=batch_size,
                model=llm_model,
                custom_prompt=custom_prompt,
                update_callback=_on_optimize_progress,
            )
            asr_data = await loop.run_in_executor(None, optimizer.optimize_subtitle, asr_data)
            _save_partial(asr_data, "Optimization complete")
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
            }
            target_lang = lang_map.get(req.target_language.lower(), TargetLanguage.SIMPLIFIED_CHINESE)

            translate_count = [0]
            def _on_translate_progress(result):
                translate_count[0] += len(result)
                pct = 65 + int(25 * translate_count[0] / len(asr_data.segments)) if len(asr_data.segments) > 0 else 65
                task_manager.update_progress(task_id, min(pct, 90), f"Translated {translate_count[0]}/{len(asr_data.segments)}...")
                _save_partial(asr_data)

            from subforge.core.translate.types import TranslatorType
            type_map = {"llm": TranslatorType.OPENAI, "bing": TranslatorType.BING, "google": TranslatorType.GOOGLE, "deeplx": TranslatorType.DEEPLX}
            translator = TranslatorFactory.create_translator(
                translator_type=type_map.get(req.translator, TranslatorType.OPENAI),
                thread_num=thread_num,
                batch_num=batch_size,
                target_language=target_lang,
                model=llm_model,
                custom_prompt=custom_prompt,
                is_reflect=req.need_reflect,
                update_callback=_on_translate_progress,
            )
            asr_data = await loop.run_in_executor(None, translator.translate_subtitle, asr_data)
            task_manager.update_progress(task_id, 90, "Translation complete")

            # Resegment (split long translated subtitles)
            task_manager.update_progress(task_id, 92, "Resegmenting subtitles...")
            from subforge.core.subtitle.resegment import resegment_subtitles
            asr_data = resegment_subtitles(asr_data)
            task_manager.update_progress(task_id, 95, "Resegmentation complete")

        # Save result
        task_manager.update_progress(task_id, 95, "Saving result...")
        output_path = Path(req.subtitle_file).with_stem(
            Path(req.subtitle_file).stem + "_processed"
        ).with_suffix(".srt")

        from subforge.core.asr.asr_data import SubtitleLayoutEnum
        layout = SubtitleLayoutEnum.TRANSLATE_ON_TOP  # Chinese on top, English on bottom
        await loop.run_in_executor(None, lambda: asr_data.save(str(output_path), layout=layout))

        task_manager.update_progress(task_id, 100, "Done")
        task_manager.complete_task(task_id, {"subtitle_file": str(output_path)})

    except Exception as e:
        task_manager.fail_task(task_id, str(e))
    finally:
        if partial_srt_path:
            try:
                PathLib(partial_srt_path).unlink(missing_ok=True)
            except Exception:
                pass
