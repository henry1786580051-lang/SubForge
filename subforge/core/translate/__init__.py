"""
翻译模块

提供多种翻译服务: OpenAI LLM、Google、Bing、DeepLX
"""

from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.base import BaseTranslator
from subforge.core.translate.bing_translator import BingTranslator
from subforge.core.translate.deeplx_translator import DeepLXTranslator
from subforge.core.translate.factory import TranslatorFactory
from subforge.core.translate.google_translator import GoogleTranslator
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.types import TargetLanguage, TranslatorType

__all__ = [
    "BaseTranslator",
    "SubtitleProcessData",
    "TranslatorFactory",
    "TranslatorType",
    "TargetLanguage",
    "BingTranslator",
    "DeepLXTranslator",
    "GoogleTranslator",
    "LLMTranslator",
]
