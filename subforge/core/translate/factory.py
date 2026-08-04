"""翻译器工厂"""

from typing import Any, Callable, Optional

from subforge.core.translate.base import BaseTranslator
from subforge.core.translate.bing_translator import BingTranslator
from subforge.core.translate.context import TranslationContext
from subforge.core.translate.deeplx_translator import DeepLXTranslator
from subforge.core.translate.google_translator import GoogleTranslator
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.types import TargetLanguage, TranslatorType
from subforge.core.utils.logger import setup_logger

logger = setup_logger("translator_factory")


class TranslatorFactory:
    """翻译器工厂类"""

    @staticmethod
    def create_translator(
        translator_type: TranslatorType,
        thread_num: int = 5,
        batch_num: int = 10,
        target_language: Optional[TargetLanguage] = None,
        model: str = "gpt-4o-mini",
        custom_prompt: str = "",
        is_reflect: bool = False,
        update_callback: Optional[Callable] = None,
        use_cache: bool = True,
        translation_context: Optional[TranslationContext] = None,
        llm_client: Any = None,
        azure_translator_key: str = "",
        azure_translator_region: str = "",
        azure_translator_endpoint: str = "",
    ) -> BaseTranslator:
        """创建翻译器实例"""
        try:
            # 如果没有指定目标语言，使用默认值
            if target_language is None:
                target_language = TargetLanguage.SIMPLIFIED_CHINESE

            if translator_type == TranslatorType.OPENAI:
                return LLMTranslator(
                    thread_num=thread_num,
                    batch_num=batch_num,
                    target_language=target_language,
                    model=model,
                    custom_prompt=custom_prompt,
                    is_reflect=is_reflect,
                    update_callback=update_callback,
                    use_cache=use_cache,
                    translation_context=translation_context,
                    llm_client=llm_client,
                )
            elif translator_type == TranslatorType.GOOGLE:
                batch_num = 5
                return GoogleTranslator(
                    thread_num=thread_num,
                    batch_num=batch_num,
                    target_language=target_language,
                    timeout=20,
                    update_callback=update_callback,
                    use_cache=use_cache,
                )
            elif translator_type == TranslatorType.BING:
                batch_num = 10
                return BingTranslator(
                    thread_num=thread_num,
                    batch_num=batch_num,
                    target_language=target_language,
                    update_callback=update_callback,
                    use_cache=use_cache,
                    api_key=azure_translator_key,
                    region=azure_translator_region,
                    endpoint=azure_translator_endpoint,
                )
            elif translator_type == TranslatorType.DEEPLX:
                batch_num = 5
                return DeepLXTranslator(
                    thread_num=thread_num,
                    batch_num=batch_num,
                    target_language=target_language,
                    timeout=20,
                    update_callback=update_callback,
                    use_cache=use_cache,
                )
        except Exception as e:
            logger.error(f"Failed to create translator: {str(e)}")
            raise
