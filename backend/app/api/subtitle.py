import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.task_manager import task_manager

# Set LLM environment variables from config at module load time
try:
    from app.api.config import get_config_value
    _api_key = get_config_value("llm_api_key", "")
    _base_url = get_config_value("llm_base_url", "")
    if _api_key:
        os.environ["OPENAI_API_KEY"] = _api_key
    if _base_url:
        os.environ["OPENAI_BASE_URL"] = _base_url
except Exception:
    pass

router = APIRouter()


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
    file_path = Path(req.subtitle_file)
    if not file_path.exists():
        raise HTTPException(status_code=400, detail="Subtitle file not found")

    task = task_manager.create_task("subtitle")
    asyncio.create_task(_run_subtitle(task.id, req))
    return {"task_id": task.id, "status": "started"}


async def _run_subtitle(task_id: str, req: SubtitleRequest):
    try:
        from app.api.config import get_config_value
        custom_prompt = req.custom_prompt or get_config_value("custom_prompt", "")

        # Use config's llm_model if not specified in request
        llm_model = req.llm_model or get_config_value("llm_model", "mimo-v2.5-pro")

        # Set environment variables for LLM client
        api_key = get_config_value("llm_api_key", "")
        base_url = get_config_value("llm_base_url", "")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url

        # Reset LLM client singleton to pick up new env vars
        try:
            import subforge.core.llm.client as llm_client_module
            llm_client_module._global_client = None
        except Exception:
            pass

        from subforge.core.asr.asr_data import ASRData
        from subforge.core.optimize.optimize import SubtitleOptimizer
        from subforge.core.split.split import SubtitleSplitter
        from subforge.core.translate.factory import TranslatorFactory
        from subforge.core.translate.types import TargetLanguage

        task_manager.update_progress(task_id, 5, "Loading subtitle file...")

        # Load subtitle into ASRData
        asr_data = ASRData.from_subtitle_file(req.subtitle_file)
        task_manager.update_progress(task_id, 10, f"Loaded {len(asr_data.segments)} segments")

        # Split long segments
        task_manager.update_progress(task_id, 15, "Splitting subtitle segments...")
        loop = asyncio.get_event_loop()
        splitter = SubtitleSplitter(thread_num=3, model=req.llm_model)
        asr_data = await loop.run_in_executor(None, splitter.split_subtitle, asr_data)
        task_manager.update_progress(task_id, 25, f"Split into {len(asr_data.segments)} segments")

        # Optimize subtitles
        if req.need_optimize:
            task_manager.update_progress(task_id, 30, "Optimizing subtitles...")

            optimize_count = [0]
            def _on_optimize_progress(result):
                optimize_count[0] += len(result)
                pct = 30 + int(30 * optimize_count[0] / len(asr_data.segments)) if len(asr_data.segments) > 0 else 30
                task_manager.update_progress(task_id, min(pct, 60), f"Optimized {optimize_count[0]}/{len(asr_data.segments)}...")

            optimizer = SubtitleOptimizer(
                thread_num=3,
                batch_num=10,
                model=llm_model,
                custom_prompt=custom_prompt,
                update_callback=_on_optimize_progress,
            )
            asr_data = await loop.run_in_executor(None, optimizer.optimize_subtitle, asr_data)
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

            from subforge.core.translate.types import TranslatorType
            type_map = {"llm": TranslatorType.OPENAI, "bing": TranslatorType.BING, "google": TranslatorType.GOOGLE, "deeplx": TranslatorType.DEEPLX}
            translator = TranslatorFactory.create_translator(
                translator_type=type_map.get(req.translator, TranslatorType.OPENAI),
                thread_num=3,
                batch_num=10,
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
