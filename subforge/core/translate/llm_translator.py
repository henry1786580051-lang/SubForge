"""LLM 翻译器（使用 OpenAI）"""

import json
import re
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import json_repair
import openai

from subforge.core.llm import call_llm
from subforge.core.prompts import get_prompt
from subforge.core.translate.base import BaseTranslator, SubtitleProcessData, logger
from subforge.core.translate.context import TranslationContext
from subforge.core.translate.types import TargetLanguage
from subforge.core.utils.cache import generate_cache_key


class LLMTranslator(BaseTranslator):
    """LLM 翻译器（OpenAI兼容API）"""

    MAX_STEPS = 3
    CONTEXT_BEFORE = 3
    CONTEXT_AFTER = 2

    def __init__(
        self,
        thread_num: int,
        batch_num: int,
        target_language: TargetLanguage,
        model: str,
        custom_prompt: str,
        is_reflect: bool,
        update_callback: Optional[Callable],
        use_cache: bool = True,
        translation_context: Optional[TranslationContext] = None,
        llm_client: Any = None,
    ):
        super().__init__(
            thread_num=thread_num,
            batch_num=batch_num,
            target_language=target_language,
            update_callback=update_callback,
            use_cache=use_cache,
        )

        self.model = model
        self.custom_prompt = custom_prompt
        self.is_reflect = is_reflect
        self.translation_context = translation_context or TranslationContext(custom_prompt=custom_prompt)
        self._all_source_by_index: Dict[int, str] = {}
        self.llm_client = llm_client

    def translate_subtitle(self, subtitle_data):
        self._all_source_by_index = {
            i: seg.text for i, seg in enumerate(subtitle_data.segments, 1)
        }
        try:
            return super().translate_subtitle(subtitle_data)
        finally:
            self._all_source_by_index = {}

    def _translate_chunk(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """翻译字幕块"""
        logger.debug(
            f"[+]正在翻译字幕: {subtitle_chunk[0].index} - {subtitle_chunk[-1].index}"
        )

        # 转换为字典格式用于API调用
        subtitle_dict = {str(data.index): data.original_text for data in subtitle_chunk}

        # 获取提示词
        if self.is_reflect:
            prompt = get_prompt(
                "translate/reflect",
                target_language=self.target_language.value,
                custom_prompt=self.custom_prompt,
            )
        else:
            prompt = get_prompt(
                "translate/standard",
                target_language=self.target_language.value,
                custom_prompt=self.custom_prompt,
            )

        try:
            # 使用agent loop进行翻译，自动验证和修正
            result_dict = self._agent_loop(prompt, subtitle_dict)

            # 处理反思翻译模式的结果
            if self.is_reflect and isinstance(result_dict, dict):
                processed_result = {
                    k: f"{v.get('native_translation', v) if isinstance(v, dict) else v}"
                    for k, v in result_dict.items()
                }
            else:
                processed_result = {k: f"{v}" for k, v in result_dict.items()}

            # 将结果填充回SubtitleProcessData
            missing_keys = []
            for data in subtitle_chunk:
                key = str(data.index)
                translated_text = processed_result.get(key)
                if not translated_text:
                    missing_keys.append(key)
                    continue
                data.translated_text = translated_text
            if missing_keys:
                raise RuntimeError(f"LLM response missing translations for keys: {missing_keys}")
            return subtitle_chunk
        except openai.RateLimitError as e:
            logger.error(f"OpenAI Rate Limit Error: {str(e)}")
            raise
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI Authentication Error: {str(e)}")
            raise
        except openai.NotFoundError as e:
            logger.error(f"OpenAI NotFound Error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"LLM translation error: {e}")
            return self._translate_chunk_single(subtitle_chunk)

    def _agent_loop(
        self, system_prompt: str, subtitle_dict: Dict[str, str]
    ) -> Dict[str, str]:
        """Agent loop翻译字幕块"""
        context_text = self.translation_context.render()
        if context_text:
            system_prompt = (
                f"{system_prompt}\n\n"
                "<global_context>\n"
                f"{context_text}\n"
                "</global_context>\n\n"
                "Use the global context and terminology consistently. "
                "Translate ONLY the current_subtitles keys. "
                "previous_context and next_context are context only; do not output them."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "previous_context": self._neighbor_context(subtitle_dict, before=True),
                        "current_subtitles": subtitle_dict,
                        "next_context": self._neighbor_context(subtitle_dict, before=False),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        last_response_dict = None
        # llm 反馈循环
        for step in range(self.MAX_STEPS):
            response = call_llm(
                messages=messages,
                model=self.model,
                use_cache=self.use_cache,
                client=self.llm_client,
            )
            content = response.choices[0].message.content
            if not content:
                logger.warning(f"LLM returned empty content, step {step + 1}/{self.MAX_STEPS}")
                continue
            response_dict = json_repair.loads(content.strip())
            last_response_dict = response_dict
            is_valid, error_message = self._validate_llm_response(
                response_dict, subtitle_dict
            )
            if is_valid:
                return response_dict
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(response_dict, ensure_ascii=False),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                    "content": (
                        f"Error: {error_message}\n\n"
                        f"Fix the errors above and output ONLY a valid JSON dictionary "
                        f"with ALL {len(subtitle_dict)} current_subtitles keys. "
                        "Do not include context keys."
                    ),
                }
            )

        if last_response_dict is None:
            raise RuntimeError("LLM translation failed after all retry attempts")
        # Validate last attempt before returning
        is_valid, error_msg = self._validate_llm_response(last_response_dict, subtitle_dict)
        if not is_valid:
            logger.warning(f"LLM translation failed validation after {self.MAX_STEPS} retries: {error_msg}")
            raise RuntimeError(f"LLM translation failed validation: {error_msg}")
        return last_response_dict

    def _neighbor_context(self, subtitle_dict: Dict[str, str], before: bool) -> List[Dict[str, str]]:
        if not self._all_source_by_index:
            return []
        numeric_keys = sorted(int(k) for k in subtitle_dict if str(k).isdigit())
        if not numeric_keys:
            return []
        if before:
            start = max(1, numeric_keys[0] - self.CONTEXT_BEFORE)
            indices = range(start, numeric_keys[0])
        else:
            end = numeric_keys[-1] + self.CONTEXT_AFTER
            indices = range(numeric_keys[-1] + 1, end + 1)
        return [
            {"index": str(index), "source": self._all_source_by_index[index]}
            for index in indices
            if index in self._all_source_by_index
        ]

    def _validate_llm_response(
        self, response_dict: Any, subtitle_dict: Dict[str, str]
    ) -> Tuple[bool, str]:
        """验证LLM翻译结果（支持普通和反思模式）

        Returns: (is_valid, error_feedback)
        """
        if not isinstance(response_dict, dict):
            return (
                False,
                f"Output must be a dict, got {type(response_dict).__name__}. Use format: {{'0': 'text', '1': 'text'}}",
            )

        expected_keys = set(subtitle_dict.keys())
        actual_keys = set(response_dict.keys())

        def sort_keys(keys):
            return sorted(keys, key=lambda x: int(x) if x.isdigit() else x)

        # 检查键是否匹配
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            error_parts = []

            if missing:
                error_parts.append(
                    f"Missing keys {sort_keys(missing)} - you must translate these items"
                )
            if extra:
                error_parts.append(
                    f"Extra keys {sort_keys(extra)} - these keys are not in input, remove them"
                )

            return (False, "; ".join(error_parts))

        # Helper: extract translated text from a response value
        def _extract_text(val):
            if isinstance(val, dict):
                return val.get("native_translation", val.get("initial_translation", ""))
            return str(val)

        # Check if translated text is actually in the target language
        _cjk_langs = {"简体中文", "繁體中文", "日本語", "한국어"}
        _is_cjk_target = self.target_language.value in _cjk_langs
        if _is_cjk_target:
            untranslated = []
            for key in sort_keys(actual_keys):
                if key not in expected_keys:
                    continue
                text = _extract_text(response_dict[key])
                original = subtitle_dict.get(key, "")
                if self._looks_untranslated_for_cjk(text, original):
                    untranslated.append(key)
            if untranslated:
                return (
                    False,
                    f"Translation to {self.target_language.value} failed: {len(untranslated)}/{len(expected_keys)} entries are still in the source language. "
                    f"You MUST translate ALL entries to {self.target_language.value}. Output target-language characters, not English. "
                    f"Untranslated keys: {untranslated[:20]}",
                )

        preserved_ok, preserved_error = self._validate_preserved_tokens(
            response_dict, subtitle_dict, _extract_text
        )
        if not preserved_ok:
            return False, preserved_error

        placeholder_ok, placeholder_error = self._validate_no_placeholder_translations(
            response_dict, subtitle_dict, _extract_text
        )
        if not placeholder_ok:
            return False, placeholder_error

        # 如果是反思模式，检查嵌套结构
        if self.is_reflect:
            for key, value in response_dict.items():
                if not isinstance(value, dict):
                    return (
                        False,
                        f"Key '{key}': value must be a dict with 'native_translation' field. Got {type(value).__name__}.",
                    )

                if "native_translation" not in value:
                    available_keys = list(value.keys())
                    return (
                        False,
                        f"Key '{key}': missing 'native_translation' field. Found keys: {available_keys}. Must include 'native_translation'.",
                    )

        return True, ""

    @staticmethod
    def _looks_like_placeholder_translation(text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return True
        normalized = re.sub(r"\s+", "", text)
        placeholder_patterns = [
            r"(?:此|本)?句.*(?:合并|并入|省略|略去)",
            r"(?:合并|并入|接上|延续).*(?:上一句|上句|前一句|前文)",
            r"(?:上一句|上句|前一句|前文).*(?:合并|包含|已译|并入)",
            r"(?:最终版本|最终字幕).*(?:合并|省略)",
            r"(?:内容)?(?:同上|见上|略|省略|无需翻译|不单独翻译)",
            r"merged(?:with|into)?(?:previous|above)",
            r"sameasabove",
            r"omitted",
        ]
        return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in placeholder_patterns)

    def _validate_no_placeholder_translations(
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        placeholders: list[str] = []
        for key in subtitle_dict:
            translated = extract_text(response_dict.get(key, ""))
            if self._looks_like_placeholder_translation(translated):
                placeholders.append(key)
        if placeholders:
            return (
                False,
                "Placeholder translations are not allowed. Every key must contain a real translation of its own source text. "
                f"Placeholder keys: {placeholders[:20]}",
            )
        return True, ""

    @staticmethod
    def _looks_untranslated_for_cjk(text: str, original: str) -> bool:
        text = str(text or "").strip()
        original = str(original or "").strip()
        if not text:
            return True
        if re.search(r"[一-鿿぀-ヿ가-힯]", text):
            return False
        if not re.search(r"[A-Za-z]", original):
            return False
        return not LLMTranslator._is_cjk_no_script_exempt(original, text)

    @staticmethod
    def _is_cjk_no_script_exempt(original: str, translated: str) -> bool:
        """Allow brand/model-only captions such as BMW M2 CS to remain Latin."""
        if re.search(r"[一-鿿぀-ヿ가-힯]", translated):
            return True
        source_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+#&/-]*", original)
        if not source_tokens:
            return True
        if len(source_tokens) > 3:
            return False

        stopwords = {
            "a", "an", "and", "are", "as", "at", "but", "for", "from", "in",
            "is", "it", "its", "it's", "of", "on", "or", "so", "that", "the",
            "this", "to", "today", "tomorrow", "was", "well", "with", "you",
        }

        def is_name_like(token: str) -> bool:
            stripped = token.strip(".,;:!?()[]{}")
            if not stripped:
                return False
            lower = stripped.lower()
            if lower in stopwords:
                return False
            if re.search(r"\d", stripped):
                return True
            if re.fullmatch(r"[A-Z]{2,}", stripped):
                return True
            if stripped != lower and stripped[0].isupper():
                return True
            return False

        return all(is_name_like(token) for token in source_tokens)

    def _validate_preserved_tokens(self, response_dict: Dict[str, Any], subtitle_dict: Dict[str, str], extract_text) -> Tuple[bool, str]:
        """Catch likely dropped model names, years, specs, and alphanumeric terms."""
        missing: list[str] = []

        def important_tokens(text: str) -> set[str]:
            tokens = set()
            collapsed_large_numbers = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
            pattern = (
                r"\b[A-Za-z]+\d+[A-Za-z0-9.-]*\b"
                r"|\b\d+[A-Za-z]+[A-Za-z0-9.-]*\b"
                r"|\b(?:19|20)\d{2}\b"
                r"|\b\d{2,3}\b"
                r"|\b[A-Z]{2,}\b"
            )
            for match in re.finditer(pattern, collapsed_large_numbers):
                token = match.group().strip(".,;:!?()[]{}")
                if len(token) >= 2:
                    tokens.add(token)
            return tokens

        def normalized_text(text: str) -> str:
            return re.sub(r"[\s,，.。-]+", "", text).lower()

        token_equivalents = {
            "BMW": {"宝马"},
            "Mercedes": {"奔驰", "梅赛德斯"},
            "Mercedes-Benz": {"奔驰", "梅赛德斯奔驰", "梅赛德斯-奔驰"},
            "Lexus": {"雷克萨斯"},
            "Honda": {"本田"},
            "Acura": {"讴歌"},
        }

        def _is_decade_token(token: str) -> bool:
            return bool(re.fullmatch(r"(?:\d{2}|\d{4})s", token, flags=re.IGNORECASE))

        def _is_ordinal_token(token: str) -> bool:
            return bool(re.fullmatch(r"\d+(?:st|nd|rd|th)", token, flags=re.IGNORECASE))

        def _ordinal_preserved(token: str, translated_norm: str) -> bool:
            if not _is_ordinal_token(token):
                return False
            number = re.match(r"\d+", token)
            if not number:
                return False
            digits = number.group()
            candidates = {digits, f"第{digits}"}
            return any(normalized_text(candidate) in translated_norm for candidate in candidates)

        def _decade_preserved(token: str, translated: str, translated_norm: str) -> bool:
            if not _is_decade_token(token):
                return False
            digits = token[:-1]
            if normalized_text(token) in translated_norm:
                return True
            if len(digits) == 4:
                century = digits[:2]
                decade = digits[2:]
                candidates = {
                    f"{digits}年代",
                    f"{century}世纪{decade}年代",
                    f"{decade}年代",
                    f"{int(decade)}年代",
                }
            else:
                candidates = {
                    f"{digits}年代",
                    f"{int(digits)}年代",
                }
            chinese_decades = {
                "00": "零零年代",
                "10": "一十年代",
                "20": "二十年代",
                "30": "三十年代",
                "40": "四十年代",
                "50": "五十年代",
                "60": "六十年代",
                "70": "七十年代",
                "80": "八十年代",
                "90": "九十年代",
            }
            decade_key = digits[-2:]
            if decade_key in chinese_decades:
                candidates.add(chinese_decades[decade_key])
            return any(normalized_text(candidate) in translated_norm for candidate in candidates)

        def _inflected_alnum_preserved(token: str, translated_norm: str) -> bool:
            if not re.search(r"\d", token):
                return False
            if not re.fullmatch(r"[A-Za-z0-9.-]+s", token):
                return False
            singular = token[:-1]
            return len(singular) >= 2 and normalized_text(singular) in translated_norm

        def _equivalent_token_preserved(token: str, translated_norm: str) -> bool:
            equivalents = token_equivalents.get(token)
            if not equivalents:
                equivalents = token_equivalents.get(token.strip(".,;:!?()[]{}"))
            if not equivalents:
                return False
            return any(normalized_text(equivalent) in translated_norm for equivalent in equivalents)

        for key, original in subtitle_dict.items():
            translated = extract_text(response_dict.get(key, ""))
            translated_norm = normalized_text(translated)
            for token in important_tokens(original):
                token_norm = normalized_text(token)
                if _decade_preserved(token, translated, translated_norm):
                    continue
                if _ordinal_preserved(token, translated_norm):
                    continue
                if _inflected_alnum_preserved(token, translated_norm):
                    continue
                if _equivalent_token_preserved(token, translated_norm):
                    continue
                if token_norm and token_norm not in translated_norm:
                    missing.append(f"{key}:{token}")

        if missing:
            return (
                False,
                "Likely dropped important source tokens. Preserve model names, years, specs, "
                f"and alphanumeric terms unless explicitly translated. Missing: {missing[:20]}",
            )
        return True, ""

    def _translate_chunk_single(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """单条翻译模式"""
        single_prompt = get_prompt(
            "translate/single", target_language=self.target_language.value
        )

        def _looks_untranslated(text: str, original: str) -> bool:
            if self.target_language.value not in {"简体中文", "繁體中文", "日本語", "한국어"}:
                return False
            import re

            if not text.strip():
                return True
            if text.strip() == original.strip():
                return True
            return not re.search(r"[一-鿿぀-ヿ가-힯]", text)

        failures: list[str] = []
        translated_items: list[SubtitleProcessData] = []

        for data in subtitle_chunk:
            try:
                response = call_llm(
                    messages=[
                        {"role": "system", "content": single_prompt},
                        {"role": "user", "content": data.original_text},
                    ],
                    model=self.model,
                    temperature=0.7,
                    use_cache=self.use_cache,
                    client=self.llm_client,
                )
                translated_text = response.choices[0].message.content.strip()
                if _looks_untranslated(translated_text, data.original_text):
                    raise RuntimeError(
                        f"Single item translation did not produce {self.target_language.value}: {translated_text!r}"
                    )
                if self._looks_like_placeholder_translation(translated_text):
                    raise RuntimeError(
                        f"Single item translation returned a placeholder instead of a translation: {translated_text!r}"
                    )
                translated_items.append(replace(data, translated_text=translated_text))
            except Exception as e:
                logger.error(f"Single item translation failed {data.index}: {str(e)}")
                failures.append(str(data.index))

        if failures:
            raise RuntimeError(
                f"Single item translation failed for {len(failures)}/{len(subtitle_chunk)} entries: {failures}"
            )
        return translated_items

    def _get_cache_key(self, chunk: List[SubtitleProcessData]) -> str:
        """生成缓存键"""
        class_name = self.__class__.__name__
        chunk_key = generate_cache_key(chunk)
        lang = self.target_language.value
        model = self.model
        prompt_key = generate_cache_key(
            {
                "custom_prompt": self.custom_prompt,
                "reflect": self.is_reflect,
                "context": self.translation_context.fingerprint(),
                "prompt_version": "context-v2-no-placeholders",
            }
        )
        return f"{class_name}:{chunk_key}:{lang}:{model}:{prompt_key}"
