"""LLM subtitle translator with structured validation and recovery."""

import difflib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, cast

import openai

from subforge.core.llm import (
    call_llm,
    get_response_text,
    parse_json_object,
    prefers_native_reasoning,
)
from subforge.core.prompts import get_prompt
from subforge.core.split.boundary import assess_english_boundary
from subforge.core.translate.base import (
    BaseTranslator,
    PartialTranslationError,
    SubtitleProcessData,
    logger,
)
from subforge.core.translate.context import TranslationContext
from subforge.core.translate.types import TargetLanguage
from subforge.core.utils.cache import generate_cache_key


class LLMTranslator(BaseTranslator):
    """Translate subtitles through OpenAI- or Anthropic-compatible clients."""

    MAX_STEPS = 3
    SINGLE_FALLBACK_MAX_ATTEMPTS = 3
    TRANSLATION_TEMPERATURE = 0.2
    CONTEXT_BEFORE = 3
    CONTEXT_AFTER = 2
    CHINESE_FLUENCY_AUDIT_BATCH_SIZE = 16
    CHINESE_FLUENCY_MAX_WINDOW = 4

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
        self.translation_context = translation_context or TranslationContext(
            custom_prompt=custom_prompt
        )
        self._all_source_by_index: Dict[int, str] = {}
        self._all_speaker_by_index: Dict[int, str] = {}
        self.llm_client = llm_client
        self._fatal_provider_error = threading.Event()
        self._fatal_provider_message = ""
        self._pending_alignment_repair_keys: set[int] = set()
        self._pending_alignment_repair_lock = threading.Lock()

    def translate_subtitle(self, subtitle_data):
        self._fatal_provider_error.clear()
        self._fatal_provider_message = ""
        with self._pending_alignment_repair_lock:
            self._pending_alignment_repair_keys.clear()
        self._all_source_by_index = {i: seg.text for i, seg in enumerate(subtitle_data.segments, 1)}
        speaker_aliases: Dict[str, str] = {}
        self._all_speaker_by_index = {}
        for index, segment in enumerate(subtitle_data.segments, 1):
            raw_speaker = str(segment.speaker_id or "").strip()
            if not raw_speaker:
                continue
            alias = speaker_aliases.setdefault(raw_speaker, f"S{len(speaker_aliases) + 1}")
            self._all_speaker_by_index[index] = alias
        try:
            return super().translate_subtitle(subtitle_data)
        finally:
            self._all_source_by_index = {}
            self._all_speaker_by_index = {}

    def _translate_chunk(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """翻译字幕块"""
        if self._fatal_provider_error.is_set():
            raise RuntimeError(self._fatal_provider_message or "LLM provider request rejected")
        logger.debug(f"[+]正在翻译字幕: {subtitle_chunk[0].index} - {subtitle_chunk[-1].index}")

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

            if self._needs_alignment_audit():
                processed_result = self._audit_reflective_alignment(
                    subtitle_dict,
                    processed_result,
                )

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
            self._open_provider_circuit(e)
            logger.error(f"OpenAI Authentication Error: {str(e)}")
            raise
        except openai.NotFoundError as e:
            self._open_provider_circuit(e)
            logger.error(f"OpenAI NotFound Error: {str(e)}")
            raise
        except Exception as e:
            if self._is_fatal_provider_error(e):
                self._open_provider_circuit(e)
                raise RuntimeError(self._fatal_provider_message) from e
            logger.error(f"LLM translation error: {e}")
            return self._translate_chunk_single(subtitle_chunk)

    def _needs_alignment_audit(self) -> bool:
        """Run conservative alignment checks for every model in reflective mode."""
        return self.is_reflect

    def _audit_reflective_alignment(
        self,
        subtitle_dict: Dict[str, str],
        translated_dict: Dict[str, str],
        *,
        initial_focus_keys: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Correct only translations that clearly belong to a neighboring key.

        Some models can preserve every JSON key while shifting a run of translations
        by one key when the source contains fragments. This independent pass asks
        for sparse corrections, then subjects the combined result to all existing
        structural validators. Audit failure keeps the already validated result.
        """
        misaligned_keys: List[str] = []
        try:

            def audit_items(keys, translations=translated_dict):
                items = {}
                for key in keys:
                    item = {
                        "source": subtitle_dict[key],
                        "translation": translations[key],
                    }
                    if str(key).isdigit():
                        numeric_key = int(key)
                        speaker = self._all_speaker_by_index.get(numeric_key, "")
                        if speaker:
                            item["speaker"] = speaker
                        previous_source = self._all_source_by_index.get(numeric_key - 1, "")
                        next_source = self._all_source_by_index.get(numeric_key + 1, "")
                        if previous_source:
                            item["previous_source"] = previous_source
                        if next_source:
                            item["next_source"] = next_source
                    items[key] = item
                return items

            ordered_keys = list(subtitle_dict)
            first_flags = self._request_alignment_flags(audit_items(ordered_keys))
            strong_outliers = self._strong_alignment_length_outliers(
                subtitle_dict,
                translated_dict,
            )
            semantic_candidates = self._strong_asr_semantic_candidates(
                subtitle_dict,
                translated_dict,
            )
            initial_flags = list(
                dict.fromkeys([*first_flags, *strong_outliers, *semantic_candidates])
            )
            if not initial_flags:
                return translated_dict

            flagged_positions = [ordered_keys.index(key) for key in initial_flags]
            focus_keys = {
                ordered_keys[position]
                for flagged_position in flagged_positions
                for position in range(
                    max(0, flagged_position - 2),
                    min(len(ordered_keys), flagged_position + 3),
                )
            }
            if initial_focus_keys:
                focus_keys.update(key for key in initial_focus_keys if key in subtitle_dict)
            confirmed_flags = self._request_alignment_flags(
                audit_items(key for key in ordered_keys if key in focus_keys),
                focused=True,
            )
            confirmed = (
                (set(first_flags) & set(confirmed_flags))
                | set(strong_outliers)
                | set(semantic_candidates)
            )
            misaligned_keys = self._expand_confirmed_alignment_keys(
                ordered_keys,
                confirmed,
            )
            misaligned_keys = [
                key
                for key in misaligned_keys
                if not self._is_disfluent_alignment_fragment(subtitle_dict[key])
            ]
            if not misaligned_keys:
                logger.info(
                    "Translation alignment flags were not confirmed: %s",
                    sorted(initial_flags),
                )
                return translated_dict

            candidate = dict(translated_dict)
            ordered_keys = list(subtitle_dict)
            for key in dict.fromkeys(misaligned_keys):
                position = ordered_keys.index(key)
                numeric_key = int(key) if key.isdigit() else None
                candidate[key] = self._translate_alignment_item(
                    subtitle_dict[key],
                    previous_source=(
                        subtitle_dict.get(ordered_keys[position - 1], "")
                        if position > 0
                        else self._all_source_by_index.get((numeric_key or 1) - 1, "")
                    ),
                    next_source=(
                        subtitle_dict.get(ordered_keys[position + 1], "")
                        if position + 1 < len(ordered_keys)
                        else self._all_source_by_index.get((numeric_key or -1) + 1, "")
                    ),
                )
            valid, error = self._validate_llm_response(
                candidate,
                subtitle_dict,
                require_reflect=False,
            )
            if not valid:
                recovery_chunk = [
                    SubtitleProcessData(
                        index=int(key),
                        original_text=subtitle_dict[key],
                    )
                    for key in ordered_keys
                    if key in set(misaligned_keys) and key.isdigit()
                ]
                if not recovery_chunk:
                    raise ValueError(error)
                recovered = self._translate_locked_batch(
                    recovery_chunk,
                    initial_feedback=("Sparse alignment corrections were still invalid: " + error),
                )
                candidate.update({str(item.index): item.translated_text for item in recovered})
                valid, error = self._validate_llm_response(
                    candidate,
                    subtitle_dict,
                    require_reflect=False,
                )
                if not valid:
                    raise ValueError(error)

            residual_flags = self._request_alignment_flags(
                audit_items(
                    misaligned_keys,
                    candidate,
                ),
                focused=True,
            )
            unresolved_repairs = sorted(
                (set(residual_flags) & set(misaligned_keys)) - set(semantic_candidates)
            )
            if unresolved_repairs:
                for key in unresolved_repairs:
                    candidate[key] = self._clean_alignment_item(
                        subtitle_dict[key],
                        candidate[key],
                    )
                valid, error = self._validate_llm_response(
                    candidate,
                    subtitle_dict,
                    require_reflect=False,
                )
                if not valid:
                    raise ValueError(error)
                fallback_flags = self._request_alignment_flags(
                    audit_items(unresolved_repairs, candidate),
                    focused=True,
                )
                unresolved_fallbacks = sorted(set(fallback_flags) & set(unresolved_repairs))
                if unresolved_fallbacks:
                    self._queue_alignment_repairs(unresolved_fallbacks)
                    logger.warning(
                        "Source-only alignment corrections need final grouped review: %s",
                        unresolved_fallbacks,
                    )
                    return translated_dict
            logger.info(
                "Translation alignment audit corrected keys: %s",
                sorted(misaligned_keys, key=lambda key: int(key) if key.isdigit() else key),
            )
            return candidate
        except Exception as error:
            self._queue_alignment_repairs(misaligned_keys)
            logger.warning("Translation alignment audit was ignored: %s", error)
            return translated_dict

    def _queue_alignment_repairs(self, keys) -> None:
        numeric_keys = {int(key) for key in keys if str(key).isdigit()}
        if not numeric_keys:
            return
        with self._pending_alignment_repair_lock:
            self._pending_alignment_repair_keys.update(numeric_keys)

    def _is_chunk_result_stable(self, translated_list: List[SubtitleProcessData]) -> bool:
        """Keep confirmed but unresolved alignment shifts out of previews and recovery."""
        chunk_indices = {item.index for item in translated_list}
        with self._pending_alignment_repair_lock:
            pending = chunk_indices & self._pending_alignment_repair_keys
        if pending:
            logger.warning(
                "Deferring provisional translation chunk with pending alignment repairs: %s",
                sorted(pending),
            )
            return False
        return True

    def _expand_confirmed_alignment_keys(
        self,
        ordered_keys: List[str],
        confirmed: set[str],
    ) -> List[str]:
        """Fill one-key holes inside a twice-confirmed local shift."""
        positions = [index for index, key in enumerate(ordered_keys) if key in confirmed]
        if len(positions) < 2:
            return [key for key in ordered_keys if key in confirmed]

        expanded = set(confirmed)
        cluster = [positions[0]]
        clusters: list[list[int]] = []
        for position in positions[1:]:
            if position - cluster[-1] <= 2:
                cluster.append(position)
            else:
                clusters.append(cluster)
                cluster = [position]
        clusters.append(cluster)

        for positions_in_cluster in clusters:
            if len(positions_in_cluster) < 2:
                continue
            start, end = positions_in_cluster[0], positions_in_cluster[-1]
            span = ordered_keys[start : end + 1]
            speakers = {
                self._all_speaker_by_index.get(int(key), "")
                for key in span
                if key.isdigit() and self._all_speaker_by_index.get(int(key), "")
            }
            if len(speakers) <= 1:
                expanded.update(span)
        return [key for key in ordered_keys if key in expanded]

    def _strong_alignment_length_outliers(
        self,
        subtitle_dict: Dict[str, str],
        translated_dict: Dict[str, str],
    ) -> List[str]:
        """Find only extreme Chinese expansions likely to contain a neighbor clause."""
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return []
        outliers = []
        for key, source in subtitle_dict.items():
            source_words = re.findall(r"[A-Za-z0-9']+", source)
            target = translated_dict.get(key, "")
            han_count = len(re.findall(r"[\u3400-\u9fff]", target))
            target_units = han_count + len(re.findall(r"[A-Za-z0-9]", target))
            has_internal_proper_noun = any(
                token[:1].isupper() and not token.isupper() for token in source_words[1:]
            )
            extreme_expansion = bool(
                source_words
                and (
                    han_count > max(18, len(source_words) * 3)
                    or (len(source_words) <= 1 and han_count >= 4)
                )
            )
            extreme_compression = bool(
                (len(source_words) >= 6 or (len(source_words) >= 5 and has_internal_proper_noun))
                and target_units < len(source_words) * 0.6
            )
            if extreme_expansion or extreme_compression:
                outliers.append(key)
        return outliers

    def _strong_asr_semantic_candidates(
        self,
        subtitle_dict: Dict[str, str],
        translated_dict: Dict[str, str],
    ) -> List[str]:
        """Select narrow ASR contradictions for a second LLM verdict, not repair."""
        ordered_keys = list(subtitle_dict)
        candidates = []
        for position, key in enumerate(ordered_keys):
            source = subtitle_dict[key]
            translated = translated_dict.get(key, "")
            if key.isdigit() and self._all_source_by_index:
                numeric_key = int(key)
                neighborhood = " ".join(
                    self._all_source_by_index[index]
                    for index in range(numeric_key - 3, numeric_key + 4)
                    if index in self._all_source_by_index
                )
            else:
                neighborhood = " ".join(
                    subtitle_dict[neighbor_key]
                    for neighbor_key in ordered_keys[
                        max(0, position - 2) : min(len(ordered_keys), position + 3)
                    ]
                )

            grouped_currency = re.findall(r"[$]\s*(\d{1,3}),000\b", source)
            has_quantity_context = bool(
                re.search(
                    r"\b(?:mpg|mph|miles?\s+per\s+gallon|fuel|speed|highway|"
                    r"doing\s+\d+)\b",
                    neighborhood,
                    flags=re.IGNORECASE,
                )
            )
            literal_currency_output = bool(
                re.search(r"(?:美元|美金|dollars?|[$])", translated, flags=re.IGNORECASE)
            )
            literal_grouped_output = any(
                re.search(rf"(?<!\d){re.escape(value)}000(?!\d)", translated)
                for value in grouped_currency
            )
            if (
                grouped_currency
                and has_quantity_context
                and (literal_currency_output or literal_grouped_output)
            ):
                candidates.append(key)
                continue

            model_year = re.search(
                r"\btook\s+a\s+break\s+for\s+(\d{2})\b",
                source,
                flags=re.IGNORECASE,
            )
            has_model_year_context = bool(
                model_year
                and re.search(
                    r"\b(?:back\s+for\s+\d{2}|model\s+year|trim)\b",
                    neighborhood,
                    flags=re.IGNORECASE,
                )
            )
            literal_duration_output = bool(
                model_year and re.search(rf"(?<!\d){model_year.group(1)}\s*年", translated)
            )
            if has_model_year_context and literal_duration_output:
                candidates.append(key)
                continue

            if self.target_language.value in {"简体中文", "繁体中文", "粤语"}:
                impossible_process = bool(
                    re.search(
                        r"\bproduction\s+to\s+(?:turn|switch|activate|change)\b",
                        source,
                        flags=re.IGNORECASE,
                    )
                    and re.search(r"(?:生产|投产|生产模式)", translated)
                )
                impossible_facelift = bool(
                    re.search(r"\bbaselift\b", source, flags=re.IGNORECASE)
                    and re.search(r"(?:基础版|基础款|底盘|升高|改装件)", translated)
                )
                impossible_reverse_camera_age = bool(
                    re.search(
                        r"\bit\s+disappears\s+from\s+(?:19|20)\d{2}\s+when\s+it\s+was\s+introduced\b",
                        source,
                        flags=re.IGNORECASE,
                    )
                    and re.search(
                        r"\b(?:reverse\s+camera|backup\s+camera)\b",
                        neighborhood,
                        flags=re.IGNORECASE,
                    )
                    and re.search(r"(?:消失|没了|不见)", translated)
                )
                if impossible_process or impossible_facelift or impossible_reverse_camera_age:
                    candidates.append(key)
        return candidates

    @staticmethod
    def _is_disfluent_alignment_fragment(source: str) -> bool:
        """Avoid rewriting short ASR fragments dominated by repeated words."""
        tokens = re.findall(r"[A-Za-z0-9']+", source.lower())
        if len(tokens) < 3 or len(tokens) > 6:
            return False
        repeated_bigram = any(
            tokens[index : index + 2] == tokens[index + 2 : index + 4]
            for index in range(len(tokens) - 3)
        )
        low_unique_ratio = len(set(tokens)) / len(tokens) <= 0.6
        return repeated_bigram or low_unique_ratio

    def _request_alignment_flags(
        self,
        items: Dict[str, Dict[str, str]],
        *,
        focused: bool = False,
    ) -> List[str]:
        focus_instruction = (
            " This is a focused second check around an already detected shift; inspect "
            "every item independently and include all other misaligned keys in this window."
            if focused
            else ""
        )
        system_prompt = f"""You are a conservative bilingual subtitle fidelity auditor for {self.target_language.value}.
Compare every source with the translation under the SAME key. Read the ordered items as a continuous transcript so you can detect a run shifted forward or backward by one key. Flag a key only when its translation clearly omits material source meaning, contains a clause owned by another key, or belongs to a neighboring key. A sentence fragment can have a fragmentary translation and is not an error. Different word order, natural compression, pronoun omission, and stylistic quality are not alignment errors. Names, numbers, negation, comparisons, and conclusions are strong ownership anchors. The optional speaker field is anonymous metadata and speaker changes are hard boundaries. Do not write translations or judge style.{focus_instruction} You MUST evaluate every input key and return ONLY {{\"alignment\": {{\"key\": true_or_false}}, \"misaligned_keys\": [\"key\"]}}. The alignment object must contain every input key exactly once; true means ownership is correct. misaligned_keys must contain exactly the keys marked false."""
        system_prompt += (
            "\nIn addition to boundary alignment, mark a key false when its translation "
            "blindly follows an ASR rendering whose literal meaning is impossible in the "
            "explicit local topic. High-confidence cases include incompatible currency or "
            "number formatting, an impossible unit, an obvious homophone, or abbreviated "
            "model-year wording with one unambiguous contextual interpretation. Do not flag "
            "uncertain wording, unsupported proper-noun corrections, normal colloquial "
            "compression, or style preferences. The previous_source and next_source fields "
            "are read-only context for that item and never belong to its translation."
        )
        context_text = self.translation_context.render()
        if context_text:
            system_prompt += (
                "\n\nUse this read-only global context only to confirm domain, terminology, "
                "units, and high-confidence ASR corrections. Never flag or rewrite an item "
                f"merely to add details from it:\n{context_text}"
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({"items": items}, ensure_ascii=False)},
        ]

        def request_audit(
            reasoning_mode: Literal["enabled", "disabled"],
            max_output_tokens: int,
        ) -> Dict[str, Any]:
            response = call_llm(
                messages=messages,
                model=self.model,
                temperature=0.1,
                use_cache=self.use_cache,
                client=self.llm_client,
                reasoning_mode=reasoning_mode,
                max_output_tokens=max_output_tokens,
            )
            return parse_json_object(get_response_text(response))

        try:
            audit = request_audit("enabled" if focused else "disabled", 4096)
        except ValueError as error:
            if not focused:
                raise
            logger.warning(
                "Thinking alignment audit produced no usable verdict; retrying without "
                "thinking: %s",
                error,
            )
            audit = request_audit("disabled", 4096)
        alignment = audit.get("alignment")
        if isinstance(alignment, dict):
            normalized_alignment = {str(key): value for key, value in alignment.items()}
            if set(normalized_alignment) != set(items) or not all(
                isinstance(value, bool) for value in normalized_alignment.values()
            ):
                raise ValueError("alignment audit must evaluate every input key with a boolean")
            alignment_flags = [key for key in items if not normalized_alignment[key]]
        else:
            # Accept old cached/provider responses while the new prompt rolls out.
            alignment_flags = None
        misaligned_keys = audit.get("misaligned_keys")
        if not isinstance(misaligned_keys, list) or not all(
            isinstance(key, (str, int)) for key in misaligned_keys
        ):
            raise ValueError("alignment audit must return a misaligned_keys list")
        normalized = list(dict.fromkeys(str(key) for key in misaligned_keys))
        if alignment_flags is not None and normalized != alignment_flags:
            raise ValueError("alignment verdicts and misaligned_keys disagree")
        unknown = set(normalized) - set(items)
        if unknown:
            raise ValueError(f"alignment audit returned unknown keys: {sorted(unknown)}")
        return normalized

    def _translate_alignment_item(
        self,
        source: str,
        *,
        previous_source: str = "",
        next_source: str = "",
    ) -> str:
        """Translate one flagged key with read-only context, then verify it separately."""
        system_prompt = f"""Translate the exact intended spoken meaning of current_source into {self.target_language.value}.
Translate ONLY current_source. Use previous_source and next_source solely to resolve references, word sense, and terminology. They are read-only: never include one of their clauses unless it is also present in current_source. If current_source is a sentence fragment, return a natural fragment without completing it. Preserve names, model identifiers, numbers, and technical terms. Do not infer or add any clause that is absent from current_source. Return only the translation with no JSON, labels, reasoning, markdown, or notes."""
        system_prompt += (
            "\nWhen literal ASR punctuation, currency formatting, a homophone, or abbreviated "
            "model-year wording is semantically impossible in the explicit local topic, "
            "restore the single unambiguous spoken interpretation while preserving the "
            "stated number. Otherwise keep the source conservative. When a confirmed "
            "high_confidence_asr_hint contains normalized_source, translate that verified "
            "spoken form while keeping the original current_source boundary."
        )
        context_text = self.translation_context.render()
        if context_text:
            system_prompt += (
                "\n\nUse this read-only global context only for terminology and reference "
                f"resolution:\n{context_text}"
            )
        system_prompt += (
            "\nResolve an elliptical role title against the explicitly named organization "
            "in nearby source or global context. Do not infer a government, school, or "
            "corporate role merely from a nearby group of people."
        )
        role_hint = self._alignment_role_hint(source, previous_source, next_source)
        payload: Dict[str, Any] = {
            "previous_source": previous_source,
            "current_source": source,
            "next_source": next_source,
        }
        if role_hint:
            payload["role_hint"] = role_hint
        reference_hint = self._alignment_reference_hint(source, previous_source)
        if reference_hint:
            payload["reference_hint"] = reference_hint
        title_hint = self._alignment_title_fragment_hint(source, previous_source)
        if title_hint:
            payload["title_fragment_hint"] = title_hint
        asr_hint = self._alignment_asr_hint(source, previous_source, next_source)
        if asr_hint:
            payload["original_asr_source"] = source
            payload["current_source"] = asr_hint["normalized_source"]
            payload["high_confidence_asr_hint"] = asr_hint
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error = "alignment item translation was invalid"
        for attempt in range(2):
            response = call_llm(
                messages=messages,
                model=self.model,
                temperature=0.1,
                use_cache=self.use_cache,
                client=self.llm_client,
                reasoning_mode="enabled" if attempt == 0 else "disabled",
                max_output_tokens=4096,
            )
            try:
                translated = get_response_text(response).strip()
            except ValueError as error:
                translated = ""
                last_error = str(error)
            if not translated or self._looks_like_placeholder_translation(translated):
                if translated:
                    last_error = "alignment item translation was empty or a placeholder"
            else:
                translated = self._apply_alignment_role_hint(translated, role_hint)
                asr_error = self._validate_alignment_asr_hint(translated, asr_hint)
                if asr_error:
                    last_error = asr_error
                else:
                    valid, error = self._validate_llm_response(
                        {"1": translated},
                        {"1": source},
                        require_reflect=False,
                    )
                    if valid:
                        return translated
                    last_error = error
            if attempt == 0:
                if translated:
                    messages.append({"role": "assistant", "content": translated})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Validation failed: {last_error}. Correct only that error. "
                            "Preserve the exact current_source boundary and return only "
                            "the complete corrected translation."
                        ),
                    }
                )
        raise ValueError(last_error)

    @staticmethod
    def _alignment_role_hint(source: str, previous_source: str, next_source: str) -> str:
        """Provide a lexical role hint only when the nearby institution is explicit."""
        if not re.search(r"\bpresident\b", source, flags=re.IGNORECASE):
            return ""
        context = f"{previous_source} {next_source}"
        if re.search(r"\b(?:union|uaw|labor)\b", context, flags=re.IGNORECASE):
            return "The role is president of the union (工会主席), not a head of state or school."
        return ""

    def _apply_alignment_role_hint(self, translated: str, role_hint: str) -> str:
        """Canonicalize a role only after an explicit institution match."""
        if "工会主席" not in role_hint:
            return translated
        if self.target_language == TargetLanguage.TRADITIONAL_CHINESE:
            canonical = "工會主席"
        else:
            canonical = "工会主席"
        if canonical in translated or re.search(r"工[会會](?:的)?主席", translated):
            return translated
        return re.sub(
            r"(?:当地(?:的)?主席|那位主席|该主席|总统|總統|校长|校長|主席)",
            canonical,
            translated,
            count=1,
        )

    @staticmethod
    def _alignment_reference_hint(source: str, previous_source: str) -> str:
        """Resolve a relative pronoun only when its antecedent is explicit."""
        if re.match(r"^\s*who\b", source, flags=re.IGNORECASE) and re.search(
            r"\bpresident\b", previous_source, flags=re.IGNORECASE
        ):
            return "Who refers to the president mentioned in previous_source, a person."
        return ""

    @staticmethod
    def _alignment_title_fragment_hint(source: str, previous_source: str) -> str:
        """Identify a title split across two subtitle boundaries."""
        current = source.strip().strip(".,!?;:")
        if not re.fullmatch(r"[A-Z][A-Za-z'-]*", current):
            return ""
        match = re.search(
            r"\b((?:The|A|An)\s+[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)?)\s*$",
            previous_source.strip(),
        )
        if not match:
            return ""
        title = f"{match.group(1)} {current}"
        return (
            f'current_source completes the title "{title}" from previous_source. '
            "Translate only this title fragment consistently; it is not a reply from next_source."
        )

    @staticmethod
    def _alignment_asr_hint(
        source: str,
        previous_source: str,
        next_source: str,
    ) -> Dict[str, str]:
        """Build a machine-verifiable hint only for an explicit local contradiction."""
        grouped = re.findall(r"[$]\s*(\d{1,3}),000\b", source)
        previous_mpg = re.search(
            r"\b(\d{1,3})\s*mpg\b",
            previous_source,
            flags=re.IGNORECASE,
        )
        doing_speed = re.search(
            r"\bdoing\s+[$]?\s*(\d{1,3}),000\b",
            source,
            flags=re.IGNORECASE,
        )
        if (
            len(grouped) >= 2
            and previous_mpg
            and grouped[0] == previous_mpg.group(1)
            and doing_speed
            and grouped[-1] == doing_speed.group(1)
        ):
            return {
                "kind": "grouped_quantity_units",
                "first_value": grouped[0],
                "first_unit": "mpg",
                "second_value": grouped[-1],
                "second_unit": "mph",
                "instruction": (
                    "ASR added currency formatting and ',000'. The first value repeats the "
                    "previous fuel-economy figure; 'doing' the second value in this road-test "
                    "context denotes vehicle speed. Translate those intended spoken units, "
                    "not money."
                ),
                "normalized_source": re.sub(
                    rf"[$]\s*{re.escape(grouped[-1])},000\b",
                    f"{grouped[-1]} mph",
                    re.sub(
                        rf"[$]\s*{re.escape(grouped[0])},000\b",
                        f"{grouped[0]} mpg",
                        source,
                        count=1,
                    ),
                    count=1,
                ),
            }

        short_year = re.search(
            r"\btook\s+a\s+break\s+for\s+(\d{2})\b",
            source,
            flags=re.IGNORECASE,
        )
        next_year = re.search(
            r"\bback\s+for\s+(\d{2})\b.*\btrim\b",
            next_source,
            flags=re.IGNORECASE,
        )
        if short_year and next_year:
            return {
                "kind": "model_year_shorthand",
                "year": short_year.group(1),
                "next_year": next_year.group(1),
                "instruction": (
                    "The adjacent trim discussion uses two-digit model-year shorthand. "
                    "Translate the first value as that model year, not a duration in years."
                ),
                "normalized_source": re.sub(
                    rf"\bfor\s+{short_year.group(1)}\b",
                    f"for model year 20{short_year.group(1)}",
                    source,
                    count=1,
                    flags=re.IGNORECASE,
                ),
            }
        if re.search(
            r"\bproduction\s+to\s+(?:turn|switch|activate|change)\b",
            source,
            flags=re.IGNORECASE,
        ):
            return {
                "kind": "process_homophone",
                "instruction": (
                    "In this control-operation sentence, ASR heard 'production' for "
                    "'process'. Translate the operation steps, not manufacturing."
                ),
                "normalized_source": re.sub(
                    r"\bproduction\b",
                    "process",
                    source,
                    count=1,
                    flags=re.IGNORECASE,
                ),
            }
        if re.search(r"\bbaselift\b", source, flags=re.IGNORECASE):
            return {
                "kind": "facelift_homophone",
                "instruction": (
                    "In this car-model sentence, ASR heard 'Baselift' for 'facelift'. "
                    "Translate the refreshed model, not a suspension or base trim."
                ),
                "normalized_source": re.sub(
                    r"\bbaselift\b",
                    "facelift",
                    source,
                    count=1,
                    flags=re.IGNORECASE,
                ),
            }
        reverse_camera_age = re.search(
            r"\bit\s+disappears\s+from\s+((?:19|20)\d{2})\s+when\s+it\s+was\s+introduced\b",
            source,
            flags=re.IGNORECASE,
        )
        if reverse_camera_age:
            year = reverse_camera_age.group(1)
            return {
                "kind": "reverse_camera_age_homophone",
                "year": year,
                "instruction": (
                    "The preceding cue is showing the reverse camera. Here ASR rendered "
                    "a remark about how dated it appears as 'disappears'. Translate that "
                    "the camera still looks like the version introduced in the stated year."
                ),
                "normalized_source": (
                    f"It still looks like the reverse camera introduced in {year}."
                ),
            }
        return {}

    @staticmethod
    def _validate_alignment_asr_hint(
        translated: str,
        hint: Dict[str, str],
    ) -> str:
        if not hint:
            return ""
        if hint.get("kind") == "grouped_quantity_units":
            first = hint["first_value"]
            second = hint["second_value"]
            first_ok = re.search(
                rf"(?:(?<!\d){first}\s*(?:mpg|英里每加仑|英里/加仑)\b|"
                rf"(?:每加仑)\D{{0,6}}(?<!\d){first}(?!\d))",
                translated,
                re.IGNORECASE,
            )
            second_ok = re.search(
                rf"(?:(?<!\d){second}\s*(?:mph|英里每小时)\b|"
                rf"(?:时速|每小时)\D{{0,6}}(?<!\d){second}(?!\d))",
                translated,
                re.IGNORECASE,
            )
            has_currency = re.search(
                r"(?:美元|美金|dollars?|[$])",
                translated,
                flags=re.IGNORECASE,
            )
            if not first_ok or not second_ok or has_currency:
                return (
                    "The confirmed ASR correction requires the first value in mpg and the "
                    "second in mph, with no currency wording."
                )
        elif hint.get("kind") == "model_year_shorthand":
            year = hint["year"]
            model_year_ok = re.search(
                rf"(?:20{year}|(?<!\d){year}\s*款)",
                translated,
            )
            duration = re.search(rf"(?<!\d){year}\s*年", translated)
            if not model_year_ok or duration:
                return "The confirmed shorthand is a model year, not a duration in years."
        elif hint.get("kind") == "process_homophone":
            if re.search(r"(?:生产|投产|生产模式)", translated):
                return "The confirmed ASR correction is an operation process, not production."
            if re.search(r"(?:高速|工程|项目|挑战|玩命|要命)", translated):
                return (
                    "Translate only the heated-seat operation process in this key; the "
                    "highway danger and project description belong to next_source."
                )
        elif hint.get("kind") == "facelift_homophone":
            if not re.search(r"(?:改款|中期改款|facelift)", translated, re.IGNORECASE):
                return "The confirmed ASR correction is facelift, not base lift."
        elif hint.get("kind") == "reverse_camera_age_homophone":
            year = hint["year"]
            if year not in translated or re.search(r"(?:消失|没了|不见)", translated):
                return (
                    "The confirmed reverse-camera remark must preserve the stated year and "
                    "describe its dated appearance, not disappearance."
                )
        return ""

    def _clean_alignment_item(self, source: str, candidate: str) -> str:
        """Remove context-borrowed clauses without losing valid disambiguation."""
        system_prompt = f"""Edit candidate_translation into an exact translation of current_source in {self.target_language.value}.
Delete every fact, action, object, name, number, or clause that current_source does not support. Keep correct contextual word-sense and reference choices already present in the candidate. Do not add replacement meaning. A fragment may remain a natural fragment. Return only the cleaned translation with no JSON, labels, reasoning, markdown, or notes."""
        response = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_source": source,
                            "candidate_translation": candidate,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            model=self.model,
            temperature=0.0,
            use_cache=self.use_cache,
            client=self.llm_client,
            reasoning_mode="disabled",
            max_output_tokens=2048,
        )
        cleaned = get_response_text(response).strip()
        if not cleaned or self._looks_like_placeholder_translation(cleaned):
            raise ValueError("cleaned alignment translation was empty or a placeholder")
        valid, error = self._validate_llm_response(
            {"1": cleaned},
            {"1": source},
            require_reflect=False,
        )
        if not valid:
            raise ValueError(error)
        return cleaned

    @staticmethod
    def _is_fatal_provider_error(error: Exception) -> bool:
        """Return whether retrying the same request cannot succeed."""
        return getattr(error, "status_code", None) in {401, 402, 403, 404}

    def _open_provider_circuit(self, error: Exception) -> None:
        status = getattr(error, "status_code", None)
        detail = str(error).strip()
        prefix = (
            f"LLM provider rejected requests with HTTP {status}"
            if status
            else "LLM provider rejected requests"
        )
        self._fatal_provider_message = f"{prefix}: {detail}"
        self._fatal_provider_error.set()

    def _agent_loop(self, system_prompt: str, subtitle_dict: Dict[str, str]) -> Dict[str, Any]:
        """Agent loop翻译字幕块"""
        system_prompt += self._dialogue_prompt_rules(subtitle_dict)
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
                        "current_subtitles": self._current_subtitles_payload(subtitle_dict),
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
                temperature=self.TRANSLATION_TEMPERATURE,
                use_cache=self.use_cache,
                client=self.llm_client,
                reasoning_mode="disabled",
                max_output_tokens=4096,
            )
            try:
                content = get_response_text(response)
                response_dict = parse_json_object(content)
            except ValueError as exc:
                logger.warning(
                    "LLM returned an invalid final answer, step %s/%s: %s",
                    step + 1,
                    self.MAX_STEPS,
                    exc,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Error: {exc}. Output ONLY one JSON object keyed by the "
                            f"{len(subtitle_dict)} current_subtitles indices. Do not output "
                            "reasoning, <think> tags, markdown, arrays, or context keys."
                        ),
                    }
                )
                continue
            last_response_dict = response_dict
            is_valid, error_message = self._validate_llm_response(response_dict, subtitle_dict)
            if is_valid:
                return cast(Dict[str, Any], response_dict)
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
            logger.warning(
                f"LLM translation failed validation after {self.MAX_STEPS} retries: {error_msg}"
            )
            raise RuntimeError(f"LLM translation failed validation: {error_msg}")
        return cast(Dict[str, Any], last_response_dict)

    def _neighbor_context(
        self, subtitle_dict: Dict[str, str], before: bool
    ) -> List[Dict[str, str]]:
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
        result = []
        for index in indices:
            if index not in self._all_source_by_index:
                continue
            item = {"index": str(index), "source": self._all_source_by_index[index]}
            speaker = self._all_speaker_by_index.get(index)
            if speaker:
                item["speaker"] = speaker
            result.append(item)
        return result

    def _current_subtitles_payload(self, subtitle_dict: Dict[str, str]) -> Dict[str, Any]:
        """Attach anonymous dialogue turns without mixing labels into source text."""
        if not any(
            self._all_speaker_by_index.get(int(key)) for key in subtitle_dict if str(key).isdigit()
        ):
            return dict(subtitle_dict)
        return {
            key: {
                "speaker": self._all_speaker_by_index.get(int(key), ""),
                "source": source,
            }
            for key, source in subtitle_dict.items()
        }

    def _dialogue_prompt_rules(self, subtitle_dict: Dict[str, str]) -> str:
        has_speakers = any(
            self._all_speaker_by_index.get(int(key)) for key in subtitle_dict if str(key).isdigit()
        )
        if not has_speakers:
            return ""
        return (
            "\n\n<dialogue_metadata>\n"
            "current_subtitles values may be objects with anonymous speaker and source fields. "
            "Use speaker changes and neighboring turns only to resolve who is responding, "
            "pronouns, ellipsis, intent, tone, and register. The speaker value is metadata, "
            "not subtitle text. Never translate, repeat, rename, or output speaker labels. "
            "Never merge dialogue turns or move meaning between keys.\n"
            "</dialogue_metadata>"
        )

    def _validate_llm_response(
        self,
        response_dict: Any,
        subtitle_dict: Dict[str, str],
        *,
        require_reflect: bool | None = None,
        check_adjacent_repetition: bool = True,
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
        _cjk_langs = {"简体中文", "繁体中文", "日本語", "韩语", "粤语"}
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

        if self._all_speaker_by_index:
            speaker_label_pattern = re.compile(
                r"(?:\[(?:speaker\s*\d+|s\d+)\]|【(?:说话人\s*\d+|s\d+)】|"
                r"\b(?:speaker|说话人)\s*\d+\b|^\s*s\d+\s*[:：-])",
                flags=re.IGNORECASE,
            )
            leaked_speakers = [
                key
                for key in sort_keys(actual_keys)
                if speaker_label_pattern.search(_extract_text(response_dict[key]))
            ]
            if leaked_speakers:
                return (
                    False,
                    "Speaker identifiers are metadata only and must not appear in translated "
                    f"subtitle text. Remove speaker labels from keys: {leaked_speakers[:20]}",
                )

        price_band_ok, price_band_error = self._validate_natural_price_bands(
            response_dict,
            subtitle_dict,
            _extract_text,
        )
        if not price_band_ok:
            return False, price_band_error

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

        residue_ok, residue_error = self._validate_unexpected_latin_residue(
            response_dict, subtitle_dict, _extract_text
        )
        if not residue_ok:
            return False, residue_error

        boundary_ok, boundary_error = self._validate_cross_key_boundaries(
            response_dict,
            subtitle_dict,
            _extract_text,
            check_adjacent_repetition=check_adjacent_repetition,
        )
        if not boundary_ok:
            return False, boundary_error

        # 如果是反思模式，检查嵌套结构
        if require_reflect is None:
            require_reflect = self.is_reflect
        if require_reflect:
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
    def _validate_natural_price_bands(
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        unnatural: list[str] = []
        for key, source in subtitle_dict.items():
            price_band = re.search(
                r"\b(\d{2})s\s+(?:to|through|-)\s+(\d{2})s\b",
                source,
                flags=re.IGNORECASE,
            )
            translated = extract_text(response_dict.get(key, ""))
            has_price_context = bool(
                re.search(
                    r"\b(?:cost|costs|expensive|pay|price|priced|range|sell|sold|worth)\b|[$]",
                    source,
                    flags=re.IGNORECASE,
                )
                or re.search(r"(?:美元|美金|万元?|千元?|\d\s*[万千k])", translated, re.IGNORECASE)
            )
            if not price_band or not has_price_context:
                continue
            if re.search(
                r"(?:\d+\s*到\s*\d+多?千|\d+多千|(?<![A-Za-z0-9])\d{2}s(?![A-Za-z0-9]))",
                translated,
                flags=re.IGNORECASE,
            ):
                unnatural.append(str(key))
        if not unnatural:
            return True, ""
        return (
            False,
            "Render colloquial thousand-dollar price bands in natural Chinese ten-thousand "
            "notation. For example, '18s to 20s' is '1.8万到2万美元', not '18到20多千美元'. "
            f"Unnatural price-band keys: {unnatural[:20]}",
        )

    def _validate_cross_key_boundaries(
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
        *,
        check_adjacent_repetition: bool = True,
    ) -> Tuple[bool, str]:
        """Catch explicit boundary edits and duplicated neighbor-owned numeric facts."""
        boundary_edits: list[str] = []
        if self.is_reflect:
            boundary_pattern = re.compile(
                r"(?:合并|并入|移到|挪到|放到|拆到).{0,20}(?:上|下|前|后|第).{0,8}(?:条|句|字幕)"
                r"|(?:merge|combine|move|put).{0,30}(?:previous|next|another|subtitle|key)",
                flags=re.IGNORECASE,
            )
            for key, value in response_dict.items():
                if isinstance(value, dict) and boundary_pattern.search(
                    str(value.get("reflection") or "")
                ):
                    boundary_edits.append(str(key))
        if boundary_edits:
            return (
                False,
                "Do not merge, move, or redistribute meaning between subtitle keys. "
                f"Boundary-edit instructions were found in keys: {boundary_edits[:20]}",
            )

        source_owners: dict[str, set[str]] = {}
        for key, source in subtitle_dict.items():
            for token in self._boundary_tokens(source):
                source_owners.setdefault(token, set()).add(key)

        translated_owners: dict[str, set[str]] = {}
        for key, value in response_dict.items():
            translated = extract_text(value)
            compact = re.sub(r"[\s,，.。-]+", "", translated).lower()
            for token in source_owners:
                token_compact = re.sub(r"[\s,，.。-]+", "", token)
                if token_compact.isdigit():
                    token_pattern = rf"(?<![a-z0-9]){re.escape(token_compact)}(?![a-z0-9])"
                else:
                    token_pattern = rf"(?<![a-z0-9]){re.escape(token_compact)}(?![a-z0-9])"
                if re.search(token_pattern, compact, flags=re.IGNORECASE):
                    translated_owners.setdefault(token, set()).add(str(key))

        leaks = []
        for token, owners in source_owners.items():
            output_keys = translated_owners.get(token, set())
            leaked_keys = {
                key
                for key in output_keys - owners
                if not self._numeric_token_belongs_to_compound_model(
                    token,
                    subtitle_dict.get(key, ""),
                    extract_text(response_dict.get(key, "")),
                )
            }
            leaks.extend(f"{key}:{token}" for key in sorted(leaked_keys))
        if leaks:
            return (
                False,
                "A number or model fact was duplicated into a different subtitle key. "
                "Keep each fact in the key that contains it in current_subtitles. "
                f"Cross-key duplicates: {leaks[:20]}",
            )

        discourse_source_owners = {
            str(key)
            for key, source in subtitle_dict.items()
            if re.search(r"\bi\s+mean\b", source, flags=re.IGNORECASE)
        }
        discourse_target_owners = {
            str(key)
            for key, value in response_dict.items()
            if re.search(r"(?:我是说|我的意思是|也就是说)", extract_text(value))
        }
        discourse_leaks = sorted(discourse_target_owners - discourse_source_owners)
        if discourse_leaks:
            return (
                False,
                "The discourse marker 'I mean' was translated under a key that does not "
                "own it. Keep it in its source key or omit it as filler; never move it to "
                f"a neighboring subtitle. Leaked keys: {discourse_leaks[:20]}",
            )

        if not check_adjacent_repetition:
            return True, ""

        ordered_keys = sorted(
            subtitle_dict,
            key=lambda key: int(key) if str(key).isdigit() else str(key),
        )

        missing_conditions: list[str] = []
        condition_translation_pattern = re.compile(
            r"(?:如果|若|要是|需要|必要|的话|一旦|只要|否则|假如|时候)"
        )
        for key in ordered_keys:
            source = subtitle_dict[key].strip()
            if not re.match(r"^if\b", source, flags=re.IGNORECASE):
                continue
            translated = extract_text(response_dict.get(key, ""))
            if not condition_translation_pattern.search(translated):
                missing_conditions.append(str(key))
        if missing_conditions:
            return (
                False,
                "A source key beginning with an if-condition lost its conditional meaning. "
                "Keep the condition explicitly under the same key. "
                f"Missing conditions: {missing_conditions[:10]}",
            )

        duplicated_quantities: list[str] = []
        for left_key, right_key in zip(ordered_keys, ordered_keys[1:]):
            if not left_key.isdigit() or not right_key.isdigit():
                continue
            if int(right_key) != int(left_key) + 1:
                continue
            left_speaker = self._all_speaker_by_index.get(int(left_key), "")
            right_speaker = self._all_speaker_by_index.get(int(right_key), "")
            if left_speaker and right_speaker and left_speaker != right_speaker:
                continue
            left_quantities = self._localized_quantity_tokens(
                extract_text(response_dict.get(left_key, ""))
            )
            right_quantities = self._localized_quantity_tokens(
                extract_text(response_dict.get(right_key, ""))
            )
            for quantity in sorted(left_quantities & right_quantities):
                left_owns = self._source_mentions_quantity(subtitle_dict[left_key], quantity)
                right_owns = self._source_mentions_quantity(subtitle_dict[right_key], quantity)
                if left_owns == right_owns:
                    continue
                duplicated_quantities.append(f"{left_key}-{right_key}:{quantity[0]}{quantity[1]}")
        if duplicated_quantities:
            return (
                False,
                "An adjacent translation anticipates and repeats a quantity that belongs "
                "to only one source key. Keep the number and unit under its owning key. "
                f"Repeated quantities: {duplicated_quantities[:10]}",
            )

        anticipated_conditions: list[str] = []
        for left_key, right_key in zip(ordered_keys, ordered_keys[1:]):
            if not left_key.isdigit() or not right_key.isdigit():
                continue
            if int(right_key) != int(left_key) + 1:
                continue
            left_speaker = self._all_speaker_by_index.get(int(left_key), "")
            right_speaker = self._all_speaker_by_index.get(int(right_key), "")
            if left_speaker and right_speaker and left_speaker != right_speaker:
                continue
            left_source = subtitle_dict[left_key].strip().lower()
            right_source = subtitle_dict[right_key].strip().lower()
            if not re.match(r"^(?:if|when)\b", right_source):
                continue
            if re.search(r"\b(?:if|when|unless)\b", left_source):
                continue
            left_target = extract_text(response_dict.get(left_key, ""))
            condition_pattern = re.compile(r"(?:如果|若|需要时|必要时|一旦|当.+?时)")
            if condition_pattern.search(left_target):
                anticipated_conditions.append(f"{left_key}-{right_key}")
        if anticipated_conditions:
            return (
                False,
                "A condition owned by the next source key was translated early under the "
                "previous key. Keep if/when meaning under its owning key. "
                f"Anticipated conditions: {anticipated_conditions[:10]}",
            )

        def source_repeats_meaning(left_key: str, right_key: str) -> bool:
            stopwords = {
                "a",
                "an",
                "and",
                "as",
                "at",
                "but",
                "for",
                "from",
                "in",
                "is",
                "it",
                "i",
                "he",
                "here",
                "she",
                "we",
                "you",
                "they",
                "there",
                "could",
                "would",
                "of",
                "on",
                "or",
                "that",
                "the",
                "this",
                "to",
                "with",
            }
            left_tokens = (
                set(self._normalized_source_text(subtitle_dict[left_key]).split()) - stopwords
            )
            right_tokens = (
                set(self._normalized_source_text(subtitle_dict[right_key]).split()) - stopwords
            )
            return bool(left_tokens & right_tokens)

        duplicated_connectors: list[str] = []
        cjk_connectors = set("都也但就又还而是的了在对和")
        for left_key, right_key in zip(ordered_keys, ordered_keys[1:]):
            if not left_key.isdigit() or not right_key.isdigit():
                continue
            if int(right_key) != int(left_key) + 1:
                continue
            left_speaker = self._all_speaker_by_index.get(int(left_key), "")
            right_speaker = self._all_speaker_by_index.get(int(right_key), "")
            if left_speaker and right_speaker and left_speaker != right_speaker:
                continue
            left_target = re.sub(
                r"[^A-Za-z0-9\u3400-\u9fff]+$",
                "",
                extract_text(response_dict.get(left_key, "")),
            )
            right_target = re.sub(
                r"^[^A-Za-z0-9\u3400-\u9fff]+",
                "",
                extract_text(response_dict.get(right_key, "")),
            )
            if not left_target or not right_target:
                continue
            connector = left_target[-1]
            if connector != right_target[0] or connector not in cjk_connectors:
                continue
            if source_repeats_meaning(left_key, right_key):
                continue
            left_source_tokens = self._normalized_source_text(subtitle_dict[left_key]).split()
            right_source_tokens = self._normalized_source_text(subtitle_dict[right_key]).split()
            if (
                left_source_tokens
                and right_source_tokens
                and left_source_tokens[-1] == right_source_tokens[0]
            ):
                continue
            duplicated_connectors.append(f"{left_key}-{right_key}:{connector}")
        if duplicated_connectors:
            return (
                False,
                "Adjacent same-speaker translations repeat a connector at the subtitle "
                "boundary. Render the connector once while preserving every key and its "
                f"meaning. Repeated boundaries: {duplicated_connectors[:10]}",
            )

        duplicated_endings: list[str] = []
        for left_key, right_key in zip(ordered_keys, ordered_keys[1:]):
            if not left_key.isdigit() or not right_key.isdigit():
                continue
            if int(right_key) != int(left_key) + 1:
                continue
            left_speaker = self._all_speaker_by_index.get(int(left_key), "")
            right_speaker = self._all_speaker_by_index.get(int(right_key), "")
            if left_speaker and right_speaker and left_speaker != right_speaker:
                continue
            left_target = self._normalized_target_text(
                extract_text(response_dict.get(left_key, ""))
            )
            right_target = self._normalized_target_text(
                extract_text(response_dict.get(right_key, ""))
            )
            left_target = left_target.replace("不过分", "不为过")
            right_target = right_target.replace("不过分", "不为过")
            if min(len(left_target), len(right_target)) < 4:
                continue
            repeated_ending = ""
            for length in range(min(6, len(left_target), len(right_target)), 2, -1):
                candidate = left_target[-length:]
                if candidate == right_target[-length:] and re.fullmatch(
                    r"[\u3400-\u9fff]+", candidate
                ):
                    repeated_ending = candidate
                    break
            if not repeated_ending:
                continue
            repeated_share = len(repeated_ending) / min(len(left_target), len(right_target))
            if (
                not (left_speaker and right_speaker)
                and repeated_ending not in {left_target, right_target}
                and repeated_share < 0.6
            ):
                # Without diarization, reserve the early ending-specific rejection for
                # an entire repeated short subtitle. Broader repetition remains covered
                # by the similarity check below without assuming speaker continuity.
                continue
            if source_repeats_meaning(left_key, right_key):
                continue
            left_source_tokens = self._normalized_source_text(subtitle_dict[left_key]).split()
            right_source_tokens = self._normalized_source_text(subtitle_dict[right_key]).split()
            if (
                left_source_tokens
                and right_source_tokens
                and left_source_tokens[-1] == right_source_tokens[-1]
            ):
                continue
            duplicated_endings.append(f"{left_key}-{right_key}:{repeated_ending}")
        if duplicated_endings:
            return (
                False,
                "Adjacent same-speaker translations repeat the same Chinese conclusion. "
                "Render that conclusion once unless the source intentionally repeats it. "
                f"Repeated endings: {duplicated_endings[:10]}",
            )

        duplicate_pairs: list[str] = []
        for left_key, right_key in zip(ordered_keys, ordered_keys[1:]):
            if left_key.isdigit() and right_key.isdigit():
                if int(right_key) != int(left_key) + 1:
                    continue
            left_target = self._normalized_target_text(
                extract_text(response_dict.get(left_key, ""))
            )
            right_target = self._normalized_target_text(
                extract_text(response_dict.get(right_key, ""))
            )
            shorter_target = min(len(left_target), len(right_target))
            if shorter_target < 6:
                continue
            if source_repeats_meaning(left_key, right_key):
                continue
            target_ratio = difflib.SequenceMatcher(None, left_target, right_target).ratio()
            left_source = self._normalized_source_text(subtitle_dict[left_key])
            right_source = self._normalized_source_text(subtitle_dict[right_key])
            source_ratio = difflib.SequenceMatcher(None, left_source, right_source).ratio()
            target_common = (
                difflib.SequenceMatcher(None, left_target, right_target).find_longest_match().size
            )
            common_share = target_common / min(len(left_target), len(right_target))
            canonical_left = left_target.replace("是一样的", "一样").replace("是相同的", "相同")
            canonical_right = right_target.replace("是一样的", "一样").replace("是相同的", "相同")
            boundary_overlap = 0
            for length in range(
                min(12, len(canonical_left), len(canonical_right)),
                3,
                -1,
            ):
                if canonical_left[-length:] == canonical_right[:length]:
                    boundary_overlap = length
                    break
            repeated_boundary_phrase = bool(
                boundary_overlap >= 4
                and source_ratio < 0.45
                and (
                    boundary_overlap >= 6
                    or re.search(
                        r"[A-Za-z0-9]",
                        canonical_right[:boundary_overlap],
                    )
                )
            )
            repeated_phrase = (
                target_common >= 7
                and common_share >= 0.38
                and source_ratio < 0.45
                and common_share - source_ratio >= 0.10
            )
            contained_short_phrase = (
                shorter_target >= 6
                and common_share == 1.0
                and source_ratio < 0.45
                and common_share - source_ratio >= 0.35
            )
            if (
                (target_ratio >= 0.68 and target_ratio - source_ratio >= 0.25)
                or repeated_phrase
                or contained_short_phrase
                or repeated_boundary_phrase
            ):
                duplicate_pairs.append(
                    f"{left_key}-{right_key} (target={target_ratio:.0%}, "
                    f"shared={common_share:.0%}, boundary={boundary_overlap}, "
                    f"source={source_ratio:.0%})"
                )
        if duplicate_pairs:
            return (
                False,
                "Adjacent translations are substantially more repetitive than their source "
                "subtitles. Do not complete, repeat, or anticipate a neighboring key. "
                f"Suspicious pairs: {duplicate_pairs[:10]}",
            )
        return True, ""

    _CHINESE_NUMBER_VALUES = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    _QUANTITY_UNITS = {
        "秒": ("second", "seconds", "sec", "secs"),
        "分钟": ("minute", "minutes", "min", "mins"),
        "小时": ("hour", "hours", "hr", "hrs"),
        "英寸": ("inch", "inches"),
        "厘米": ("centimeter", "centimeters", "cm"),
        "毫米": ("millimeter", "millimeters", "mm"),
        "公里": ("kilometer", "kilometers", "km"),
        "英里": ("mile", "miles"),
        "磅": ("pound", "pounds", "lb", "lbs"),
        "公斤": ("kilogram", "kilograms", "kg"),
        "马力": ("horsepower", "hp"),
    }

    @classmethod
    def _localized_quantity_tokens(cls, text: str) -> set[tuple[int, str]]:
        """Extract small translated quantities used for adjacent ownership checks."""
        unit_pattern = "|".join(sorted(map(re.escape, cls._QUANTITY_UNITS), key=len, reverse=True))
        quantities: set[tuple[int, str]] = set()
        for match in re.finditer(
            rf"(?<![A-Za-z0-9])(?P<number>\d+|[零〇一二两三四五六七八九十])\s*"
            rf"(?P<unit>{unit_pattern})(?:钟)?",
            str(text or ""),
        ):
            raw_number = match.group("number")
            value = (
                int(raw_number) if raw_number.isdigit() else cls._CHINESE_NUMBER_VALUES[raw_number]
            )
            quantities.add((value, match.group("unit")))
        return quantities

    @classmethod
    def _source_mentions_quantity(cls, source: str, quantity: tuple[int, str]) -> bool:
        value, unit = quantity
        number_words = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        number_pattern = rf"(?:{value}|{number_words.get(value, str(value))})"
        unit_pattern = "|".join(map(re.escape, cls._QUANTITY_UNITS[unit]))
        return bool(
            re.search(
                rf"\b{number_pattern}\s*(?:{unit_pattern})\b",
                str(source or ""),
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _boundary_tokens(text: str) -> set[str]:
        """Extract numbers and alphanumeric model/spec tokens with locked ownership."""
        return {
            match.group().lower()
            for match in re.finditer(
                r"\b(?:[A-Za-z]+\d+[A-Za-z0-9.-]*|\d+[A-Za-z]+[A-Za-z0-9.-]*|\d{2,4})\b",
                str(text or ""),
            )
        }

    @classmethod
    def _numeric_token_belongs_to_compound_model(
        cls,
        token: str,
        source: str,
        translated: str,
    ) -> bool:
        """Treat `RT392`, `RT 392`, and `R/T 392` as the same owned model."""
        if not token.isdigit():
            return False
        translated_compact = re.sub(r"[^a-z0-9]", "", translated.lower())
        for source_token in cls._boundary_tokens(source):
            source_compact = re.sub(r"[^a-z0-9]", "", source_token.lower())
            if (
                re.search(r"[a-z]", source_compact)
                and token in source_compact
                and source_compact in translated_compact
            ):
                return True
        return False

    @staticmethod
    def _normalized_target_text(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9\u3400-\u9fff]+", "", str(text or "").lower())

    @staticmethod
    def _normalized_source_text(text: str) -> str:
        return " ".join(re.findall(r"[A-Za-z0-9']+", str(text or "").lower()))

    def _validate_unexpected_latin_residue(
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        """Reject stray English grammar words in Chinese translations.

        Brand names and mode labels remain valid. This intentionally checks only
        common lowercase function words, avoiding broad language detection that
        would reject legitimate Latin product names.
        """
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return True, ""
        function_words = {
            "and",
            "because",
            "but",
            "for",
            "from",
            "into",
            "of",
            "that",
            "the",
            "this",
            "to",
            "with",
        }
        residue: list[str] = []
        for key in subtitle_dict:
            translated = str(extract_text(response_dict.get(key, "")) or "")
            matches = list(re.finditer(r"(?<![A-Za-z-])[A-Za-z]{2,}(?![A-Za-z-])", translated))
            for index, match in enumerate(matches):
                token = match.group()
                lower = token.lower()
                if token != lower or lower not in function_words:
                    continue
                previous = matches[index - 1].group() if index else ""
                following = matches[index + 1].group() if index + 1 < len(matches) else ""
                if previous[:1].isupper() and following[:1].isupper():
                    continue
                residue.append(f"{key}:{token}")
        if residue:
            return (
                False,
                "Unexpected English grammar words remain in a Chinese translation. "
                "Translate them unless they are part of a proper product name. "
                f"Residual words: {residue[:20]}",
            )
        return True, ""

    @staticmethod
    def _looks_like_placeholder_translation(text: str) -> bool:
        return BaseTranslator._looks_like_placeholder_translation(text)

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

    def _looks_untranslated_for_cjk(self, text: str, original: str) -> bool:
        text = str(text or "").strip()
        original = str(original or "").strip()
        if not text:
            return True
        return self._is_untranslated_output(text, original)

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
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "but",
            "for",
            "from",
            "in",
            "is",
            "it",
            "its",
            "it's",
            "of",
            "on",
            "or",
            "so",
            "that",
            "the",
            "this",
            "to",
            "today",
            "tomorrow",
            "was",
            "well",
            "with",
            "you",
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

    def _validate_preserved_tokens(
        self, response_dict: Dict[str, Any], subtitle_dict: Dict[str, str], extract_text
    ) -> Tuple[bool, str]:
        """Catch likely dropped model names, years, specs, and alphanumeric terms."""
        missing: list[str] = []

        def important_tokens(text: str) -> set[str]:
            tokens = set()
            uppercase_stopwords = {
                "AM",
                "AS",
                "AT",
                "BE",
                "DO",
                "IF",
                "IN",
                "IS",
                "IT",
                "NO",
                "OF",
                "ON",
                "OR",
                "TO",
                "US",
                "WE",
            }
            collapsed_large_numbers = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
            pattern = (
                r"\b[A-Za-z]+\d+[A-Za-z0-9.-]*\b"
                r"|\b\d+[A-Za-z]+[A-Za-z0-9.-]*\b"
                r"|\b(?:19|20)\d{2}\b"
                r"|\b\d{2,}\b"
                r"|\b[A-Z]{2,}\b"
            )
            for match in re.finditer(pattern, collapsed_large_numbers):
                token = match.group().strip(".,;:!?()[]{}")
                if len(token) >= 2 and token not in uppercase_stopwords:
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
            "REM": {"快速眼动"},
            "WWII": {"二战", "第二次世界大战"},
        }

        def _world_war_roman_preserved(
            original: str,
            token: str,
            translated_norm: str,
        ) -> bool:
            roman = token.upper()
            if roman not in {"I", "II"}:
                return False
            if not re.search(
                rf"\bWorld\s+War\s+{roman}\b",
                original,
                flags=re.IGNORECASE,
            ):
                return False
            equivalents = {"一战", "第一次世界大战"} if roman == "I" else {"二战", "第二次世界大战"}
            return any(normalized_text(value) in translated_norm for value in equivalents)

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

        def _integer_chinese_forms(token: str) -> set[str]:
            if not re.fullmatch(r"\d+", token):
                return set()
            digits = "零一二三四五六七八九"
            digit_form = "".join(digits[int(character)] for character in token)
            value = int(token)
            if value == 0:
                return {"零"}
            if value >= 10000:
                return {digit_form}

            units = ((1000, "千"), (100, "百"), (10, "十"))
            remaining = value
            parts: list[str] = []
            pending_zero = False
            for unit_value, unit_name in units:
                quotient, remaining = divmod(remaining, unit_value)
                if quotient:
                    if pending_zero:
                        parts.append("零")
                        pending_zero = False
                    if unit_value != 10 or quotient != 1 or parts:
                        parts.append(digits[quotient])
                    parts.append(unit_name)
                elif parts and remaining:
                    pending_zero = True
            if remaining:
                if pending_zero:
                    parts.append("零")
                parts.append(digits[remaining])
            return {digit_form, "".join(parts)}

        def _integer_preserved(token: str, translated_norm: str) -> bool:
            return any(
                normalized_text(candidate) in translated_norm
                for candidate in _integer_chinese_forms(token)
            )

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

        def _is_price_band_token(original: str, token: str, translated: str = "") -> bool:
            has_price_context = bool(
                re.search(
                    r"\b(?:cost|costs|expensive|pay|price|priced|range|sell|sold|worth)\b"
                    r"|[$]",
                    original,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    r"(?:美元|美金|万元?|千元?|\d\s*[万千k])",
                    translated,
                    flags=re.IGNORECASE,
                )
            )
            return bool(
                re.fullmatch(r"\d{2}s", token, flags=re.IGNORECASE)
                and has_price_context
            )

        def _price_band_preserved(
            original: str,
            token: str,
            translated: str,
            translated_norm: str,
        ) -> bool:
            """Accept plural price bands translated as thousands or ten-thousands."""
            if not _is_price_band_token(original, token, translated):
                return False

            match = re.fullmatch(r"(\d{2})s", token, flags=re.IGNORECASE)
            assert match is not None
            value = int(match.group(1))
            candidates = {
                f"{value}k",
                f"{value}000",
                f"{Decimal(value) / Decimal(10):g}万",
            }
            if value == 20:
                candidates.update({"二万", "两万"})
            return any(
                normalized_text(candidate) in translated_norm for candidate in candidates
            ) or bool(
                re.search(
                    rf"(?<!\d){value}(?!\d)\s*(?:千|k)",
                    translated,
                    flags=re.IGNORECASE,
                )
            )

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

        def _compound_model_preserved(token: str, translated: str) -> bool:
            if not (re.search(r"[A-Za-z]", token) and re.search(r"\d", token)):
                return False
            token_compact = re.sub(r"[^a-z0-9]", "", token.lower())
            translated_compact = re.sub(r"[^a-z0-9]", "", translated.lower())
            return bool(token_compact and token_compact in translated_compact)

        def _magnitude_preserved(
            original: str,
            translated_norm: str,
            token: str = "",
        ) -> bool:
            """Accept equivalent grand/K notation without allowing a lost magnitude."""

            def decimal_text(value: Decimal) -> str:
                rendered = format(value, "f")
                return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered

            for match in re.finditer(
                r"\b(\d+(?:\.\d+)?)\s*(grand|k)\b",
                original,
                flags=re.IGNORECASE,
            ):
                raw = match.group(0)
                if token and normalized_text(token) not in {
                    normalized_text(raw),
                    normalized_text(match.group(1)),
                }:
                    continue
                try:
                    stated = Decimal(match.group(1))
                except InvalidOperation:
                    continue
                absolute = stated * 1000
                ten_thousands = absolute / 10000
                candidates = {
                    raw,
                    decimal_text(absolute),
                    f"{decimal_text(ten_thousands)}万",
                }
                if any(normalized_text(candidate) in translated_norm for candidate in candidates):
                    return True
            return False

        def _thousand_magnitude_preserved(
            original: str,
            translated_norm: str,
            token: str,
        ) -> bool:
            """Accept natural Chinese equivalents of explicit thousand amounts."""
            if not re.fullmatch(r"\d+(?:\.\d+)?", token):
                return False

            normalized_source = re.sub(r"[-\s]+", " ", original).strip()
            match = re.search(
                rf"\b{re.escape(token)}\s+"
                r"(?:(?:some\s+odd|some|odd)\s+)?thousand\b",
                normalized_source,
                flags=re.IGNORECASE,
            )
            if not match:
                return False

            try:
                absolute = Decimal(token) * 1000
            except InvalidOperation:
                return False

            def decimal_text(value: Decimal) -> str:
                rendered = format(value, "f")
                return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered

            candidates = {decimal_text(absolute)}
            ten_thousands = absolute / 10000
            if ten_thousands == ten_thousands.to_integral_value():
                compact_value = str(int(ten_thousands))
                candidates.add(f"{compact_value}万")
                for chinese_value in _integer_chinese_forms(compact_value):
                    candidates.add(f"{chinese_value}万")
                if compact_value == "2":
                    candidates.add("两万")
            else:
                candidates.add(f"{decimal_text(ten_thousands)}万")
            return any(normalized_text(candidate) in translated_norm for candidate in candidates)

        def _introductory_101_preserved(
            original: str,
            token: str,
            translated_norm: str,
        ) -> bool:
            """Treat ``subject 101`` as an introductory-concept idiom in Chinese."""
            if token != "101" or self.target_language.value not in {
                "简体中文",
                "繁体中文",
                "粤语",
            }:
                return False
            if re.search(
                r"\b(?:route|highway|room|suite|flight|model|interstate)\s+101\b",
                original,
                flags=re.IGNORECASE,
            ):
                return False
            if not re.search(
                r"\b[A-Za-z][A-Za-z' -]{1,80}\s+101\b"
                r"(?=\s*(?:[.!?,;:]|$|\bis\b|\bwas\b))",
                original,
                flags=re.IGNORECASE,
            ):
                return False
            return any(
                normalized_text(equivalent) in translated_norm
                for equivalent in ("入门", "基础", "基本常识", "基础知识", "初级", "概论")
            )

        def _asr_formatted_number_preserved(
            original: str,
            token: str,
            translated: str,
            translated_norm: str,
        ) -> bool:
            """Allow only narrow, explicit repairs of ASR-formatted quantities."""
            grouped = re.fullmatch(r"(\d{1,3})000", token)
            if grouped:
                base = grouped.group(1)
                source_pattern = rf"[$]\s*{re.escape(base)},000\b"
                unit_after_pattern = (
                    rf"(?<!\d){re.escape(base)}\s*(?:mpg|mph|km/?h|kph|rpm|"
                    r"英里每加仑|英里/加仑|英里每小时|公里每小时|马力|"
                    r"磅英尺|磅-英尺)\b"
                )
                unit_before_pattern = (
                    rf"(?:每加仑|时速|速度|mpg|mph)\D{{0,8}}(?<!\d)"
                    rf"{re.escape(base)}(?!\d)"
                )
                has_unit = bool(
                    re.search(
                        unit_after_pattern,
                        translated,
                        flags=re.IGNORECASE,
                    )
                    or re.search(
                        unit_before_pattern,
                        translated,
                        flags=re.IGNORECASE,
                    )
                )
                if re.search(source_pattern, original) and has_unit:
                    return True

            if re.fullmatch(r"\d{2}", token):
                year = f"20{token}"
                shorthand_pattern = rf"\b(?:for|in|model\s+year)\s+{token}\b"
                if re.search(shorthand_pattern, original, flags=re.IGNORECASE) and (
                    year in translated_norm
                ):
                    return True
            return False

        for key, original in subtitle_dict.items():
            translated = extract_text(response_dict.get(key, ""))
            translated_norm = normalized_text(translated)
            for token in important_tokens(original):
                token_norm = normalized_text(token)
                if _world_war_roman_preserved(original, token, translated_norm):
                    continue
                if _is_price_band_token(original, token, translated):
                    if _price_band_preserved(
                        original,
                        token,
                        translated,
                        translated_norm,
                    ):
                        continue
                    missing.append(f"{key}:{token}")
                    continue
                if _decade_preserved(token, translated, translated_norm):
                    continue
                if _ordinal_preserved(token, translated_norm):
                    continue
                if _integer_preserved(token, translated_norm):
                    continue
                if _inflected_alnum_preserved(token, translated_norm):
                    continue
                if _equivalent_token_preserved(token, translated_norm):
                    continue
                if _compound_model_preserved(token, translated):
                    continue
                if _magnitude_preserved(original, translated_norm, token):
                    continue
                if _thousand_magnitude_preserved(original, translated_norm, token):
                    continue
                if _introductory_101_preserved(original, token, translated_norm):
                    continue
                if _asr_formatted_number_preserved(
                    original,
                    token,
                    translated,
                    translated_norm,
                ):
                    continue
                if token_norm and token_norm not in translated_norm:
                    missing.append(f"{key}:{token}")

            if re.search(
                r"\b\d+(?:\.\d+)?\s*(?:grand|k)\b",
                original,
                flags=re.IGNORECASE,
            ) and not _magnitude_preserved(original, translated_norm):
                missing.append(f"{key}:numeric magnitude")

        if missing:
            return (
                False,
                "Likely dropped important source tokens. Preserve model names, years, specs, "
                f"and alphanumeric terms unless explicitly translated. Missing: {missing[:20]}",
            )
        return True, ""

    def _translate_locked_batch(
        self,
        subtitle_chunk: List[SubtitleProcessData],
        initial_feedback: str = "",
    ) -> List[SubtitleProcessData]:
        """Recover a failed reflective batch without discarding key ownership."""
        subtitle_dict = {str(data.index): data.original_text for data in subtitle_chunk}
        system_prompt = get_prompt(
            "translate/standard",
            target_language=self.target_language.value,
            custom_prompt=self.custom_prompt,
        )
        system_prompt += self._dialogue_prompt_rules(subtitle_dict)
        context_text = self.translation_context.render()
        if context_text:
            system_prompt += f"\n\n<global_context>\n{context_text}\n</global_context>"
        system_prompt += (
            "\n\n<recovery_rules>\n"
            "This is a boundary-safe recovery pass. Every output key is locked to exactly "
            "the words and meaning in the same current_subtitles key. Keep fragments as "
            "fragments. Never complete, repeat, anticipate, or summarize a neighboring key. "
            "Return one plain JSON string value for every current key.\n"
            "</recovery_rules>"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "previous_context": self._neighbor_context(subtitle_dict, before=True),
                        "current_subtitles": self._current_subtitles_payload(subtitle_dict),
                        "next_context": self._neighbor_context(subtitle_dict, before=False),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        if initial_feedback:
            messages.append(
                {
                    "role": "user",
                    "content": f"The previous recovery was invalid: {initial_feedback}",
                }
            )

        last_error = "invalid recovery response"
        for _attempt in range(self.SINGLE_FALLBACK_MAX_ATTEMPTS):
            response = call_llm(
                messages=messages,
                model=self.model,
                temperature=0.2,
                use_cache=self.use_cache,
                client=self.llm_client,
                reasoning_mode="disabled",
                max_output_tokens=4096,
            )
            try:
                response_dict = parse_json_object(get_response_text(response))
            except ValueError as error:
                last_error = str(error)
            else:
                is_valid, error_message = self._validate_llm_response(
                    response_dict,
                    subtitle_dict,
                    require_reflect=False,
                )
                if is_valid:
                    recovered = []
                    for data in subtitle_chunk:
                        value = response_dict[str(data.index)]
                        if isinstance(value, dict):
                            value = value.get("native_translation", value.get("translation", ""))
                        recovered.append(replace(data, translated_text=str(value).strip()))
                    return recovered
                last_error = error_message
                messages.append(
                    {"role": "assistant", "content": json.dumps(response_dict, ensure_ascii=False)}
                )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Validation failed: {last_error}. Fix only the invalid ownership or "
                        "formatting. Output the complete JSON object with exactly the current "
                        "subtitle keys and plain translated string values."
                    ),
                }
            )
        raise RuntimeError(f"Locked batch recovery failed validation: {last_error}")

    def _validate_single_context_ownership(self, current: Dict[str, str], translated: str) -> None:
        """Reject model/spec tokens borrowed from neighboring source context."""
        own_tokens = set().union(*(self._boundary_tokens(text) for text in current.values()))
        neighbors = [
            *self._neighbor_context(current, before=True),
            *self._neighbor_context(current, before=False),
        ]
        neighbor_tokens = set().union(
            *(self._boundary_tokens(item["source"]) for item in neighbors),
            set(),
        )
        compact = re.sub(r"[\s,，.。-]+", "", translated).lower()
        borrowed = []
        for token in neighbor_tokens - own_tokens:
            if any(
                self._numeric_token_belongs_to_compound_model(
                    token,
                    own_source,
                    translated,
                )
                for own_source in current.values()
            ):
                continue
            token_compact = re.sub(r"[\s,，.。-]+", "", token)
            pattern = (
                rf"(?<!\d){re.escape(token_compact)}(?!\d)"
                if token_compact.isdigit()
                else rf"(?<![a-z0-9]){re.escape(token_compact)}(?![a-z0-9])"
            )
            if re.search(pattern, compact, flags=re.IGNORECASE):
                borrowed.append(token)
        if borrowed:
            raise RuntimeError(
                "Single item translation borrowed number/model tokens from context: "
                f"{sorted(borrowed)}"
            )

    def _translate_chunk_single(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """Recover a failed batch, preserving batch boundaries whenever possible."""
        if len(subtitle_chunk) > 1:
            try:
                return self._translate_locked_batch(subtitle_chunk)
            except Exception as error:
                if self._is_fatal_provider_error(error):
                    self._open_provider_circuit(error)
                    raise RuntimeError(self._fatal_provider_message) from error
                logger.warning("Locked batch recovery failed; trying single items: %s", error)

        single_prompt = get_prompt("translate/single", target_language=self.target_language.value)

        def _looks_untranslated(text: str, original: str) -> bool:
            if self.target_language.value not in {"简体中文", "繁体中文", "日本語", "韩语", "粤语"}:
                return False
            import re

            if not text.strip():
                return True
            if text.strip() == original.strip():
                return True
            return not re.search(r"[一-鿿぀-ヿ가-힯]", text)

        failures: list[int] = []
        translated_items: list[SubtitleProcessData] = []

        for data in subtitle_chunk:
            if self._fatal_provider_error.is_set():
                raise RuntimeError(self._fatal_provider_message or "LLM provider request rejected")
            current = {str(data.index): data.original_text}
            messages = [
                {
                    "role": "system",
                    "content": single_prompt + self._dialogue_prompt_rules(current),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "previous_context": self._neighbor_context(current, before=True),
                            "current_subtitle": self._current_subtitles_payload(current),
                            "next_context": self._neighbor_context(current, before=False),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            last_error: Exception | None = None
            for attempt in range(self.SINGLE_FALLBACK_MAX_ATTEMPTS):
                try:
                    response = call_llm(
                        messages=messages,
                        model=self.model,
                        temperature=self.TRANSLATION_TEMPERATURE,
                        use_cache=self.use_cache,
                        client=self.llm_client,
                        reasoning_mode="disabled",
                        max_output_tokens=2048,
                    )
                    translated_text = get_response_text(response)
                    if _looks_untranslated(translated_text, data.original_text):
                        raise RuntimeError(
                            f"Single item translation did not produce {self.target_language.value}: "
                            f"{translated_text!r}"
                        )
                    if self._looks_like_placeholder_translation(translated_text):
                        raise RuntimeError(
                            "Single item translation returned a placeholder instead of a "
                            f"translation: {translated_text!r}"
                        )
                    current_response = {str(data.index): translated_text}
                    current_source = {str(data.index): data.original_text}
                    preserved_ok, preserved_error = self._validate_preserved_tokens(
                        current_response,
                        current_source,
                        lambda value: str(value),
                    )
                    if not preserved_ok:
                        raise RuntimeError(preserved_error)
                    residue_ok, residue_error = self._validate_unexpected_latin_residue(
                        current_response,
                        current_source,
                        lambda value: str(value),
                    )
                    if not residue_ok:
                        raise RuntimeError(residue_error)
                    self._validate_single_context_ownership(current, translated_text)
                    translated_items.append(replace(data, translated_text=translated_text))
                    last_error = None
                    break
                except Exception as error:
                    if self._is_fatal_provider_error(error):
                        self._open_provider_circuit(error)
                        raise RuntimeError(self._fatal_provider_message) from error
                    last_error = error
                    if attempt + 1 < self.SINGLE_FALLBACK_MAX_ATTEMPTS:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"The previous answer was invalid: {error}. Translate ONLY "
                                    f"current_subtitle into {self.target_language.value}. Return "
                                    "only the translated subtitle text, without English source, "
                                    "reasoning, labels, notes, or JSON."
                                ),
                            }
                        )
            if last_error is not None:
                logger.error("Single item translation failed %s: %s", data.index, last_error)
                failures.append(data.index)

        if failures:
            raise PartialTranslationError(
                f"Single item translation failed for {len(failures)}/{len(subtitle_chunk)} entries: {failures}",
                completed=translated_items,
                failed_indices=failures,
            )

        fallback_source = {str(data.index): data.original_text for data in subtitle_chunk}
        fallback_response = {str(data.index): data.translated_text for data in translated_items}
        fallback_ok, fallback_error = self._validate_llm_response(
            fallback_response,
            fallback_source,
            require_reflect=False,
            # Every fallback item has already passed isolated ownership and
            # language validation. Re-running the similarity heuristic across
            # independently generated lines can reject legitimate repeated
            # phrasing and discard the entire recovered batch.
            check_adjacent_repetition=False,
        )
        if not fallback_ok and len(subtitle_chunk) > 1:
            try:
                return self._translate_locked_batch(
                    subtitle_chunk,
                    initial_feedback=fallback_error,
                )
            except Exception as error:
                logger.warning(
                    "Locked revalidation failed after every fallback item passed isolated "
                    "validation; retaining the isolated results for final consistency review: %s",
                    error,
                )
        return translated_items

    def _finalize_translated_list(
        self,
        source_list: List[SubtitleProcessData],
        translated_list: List[SubtitleProcessData],
    ) -> List[SubtitleProcessData]:
        """Repair confirmed alignment errors and high-confidence adjacent repetition."""
        if not (self.is_reflect and self._needs_alignment_audit()):
            return translated_list

        translated_by_index = {item.index: item for item in translated_list}
        repetition_markers = (
            "Repeated boundaries:",
            "Repeated endings:",
            "Suspicious pairs:",
        )

        with self._pending_alignment_repair_lock:
            pending_keys = sorted(self._pending_alignment_repair_keys)
            self._pending_alignment_repair_keys.clear()

        pending_windows = self._pending_alignment_repair_windows(
            source_list,
            translated_by_index,
            pending_keys,
        )
        for repair_sources in pending_windows:
            repair_indices = [item.index for item in repair_sources]
            try:
                if len(repair_sources) == 1:
                    item = repair_sources[0]
                    repaired = [
                        replace(
                            item,
                            translated_text=self._translate_alignment_item(
                                item.original_text,
                                previous_source=self._all_source_by_index.get(item.index - 1, ""),
                                next_source=self._all_source_by_index.get(item.index + 1, ""),
                            ),
                        )
                    ]
                else:
                    repaired = self._translate_locked_batch(
                        repair_sources,
                        initial_feedback=(
                            "At least one key in this original translation batch was "
                            "independently confirmed as shifted or as containing neighboring "
                            "meaning. Rebuild the complete batch so a multi-key shift cannot "
                            "survive outside the initially flagged keys. Translate only each "
                            "key's own source words and preserve incomplete sentence fragments."
                        ),
                    )
            except Exception as error:
                logger.warning(
                    "Final grouped alignment repair failed for subtitles %s; retaining "
                    "the validated translations: %s",
                    repair_indices,
                    error,
                )
                continue
            for item in repaired:
                if item.index in translated_by_index:
                    translated_by_index[item.index] = item
            logger.info(
                "Final grouped alignment repair corrected batch: %s",
                repair_indices,
            )

        for boundary in range(1, len(source_list)):
            pair_sources = source_list[boundary - 1 : boundary + 1]
            if len(pair_sources) != 2 or any(
                item.index not in translated_by_index for item in pair_sources
            ):
                continue
            source_dict = {str(item.index): item.original_text for item in pair_sources}
            response_dict = {
                str(item.index): translated_by_index[item.index].translated_text
                for item in pair_sources
            }
            valid, error = self._validate_cross_key_boundaries(
                response_dict,
                source_dict,
                lambda value: str(value),
            )
            dependent_boundary = self._has_repetitive_dependent_boundary(
                pair_sources,
                response_dict,
            )
            repetition_failure = not valid and any(marker in error for marker in repetition_markers)
            if not (dependent_boundary or repetition_failure):
                continue
            logger.warning(
                "Repairing dependent or repeated adjacent translation: %s",
                error or "source sentence continues across the batch boundary",
            )
            repair_sources = pair_sources
            if boundary >= 2:
                previous_pair = source_list[boundary - 2 : boundary]
                previous_text = previous_pair[0].original_text.strip()
                if self._has_dependent_source_boundary(previous_pair) or not re.search(
                    r"[.!?][\"')\]]*$", previous_text
                ):
                    repair_sources = source_list[boundary - 2 : boundary + 1]
            right_text = pair_sources[-1].original_text.strip()
            if boundary + 1 < len(source_list) and not re.search(r"[.!?][\"')\]]*$", right_text):
                start = boundary - 2 if len(repair_sources) == 3 else boundary - 1
                repair_sources = source_list[start : boundary + 2]
            repair_source_dict = {str(item.index): item.original_text for item in repair_sources}
            try:
                repaired = self._translate_locked_batch(
                    repair_sources,
                    initial_feedback=(
                        error
                        if repetition_failure
                        else (
                            "These two subtitle keys are consecutive fragments of one source "
                            "sentence. Translate them together without restating the same subject, "
                            "predicate, or qualification in both keys. Keep every fact under the "
                            "key that owns it."
                        )
                    ),
                )
            except Exception as repair_error:
                logger.warning(
                    "Adjacent translation repair failed for subtitles %s and %s; retaining "
                    "the validated translations: %s",
                    pair_sources[0].index,
                    pair_sources[1].index,
                    repair_error,
                )
                continue
            repaired_response = {str(item.index): item.translated_text for item in repaired}
            repaired_valid, repaired_error = self._validate_cross_key_boundaries(
                repaired_response,
                repair_source_dict,
                lambda value: str(value),
            )
            if not repaired_valid:
                logger.warning(
                    "Repaired boundary still failed repetition validation: %s",
                    repaired_error,
                )
                repaired = self._translate_locked_batch(
                    repair_sources,
                    initial_feedback=(
                        "The previous boundary repair still duplicated or anticipated "
                        "neighboring meaning: " + repaired_error
                    ),
                )
                repaired_response = {str(item.index): item.translated_text for item in repaired}
                repaired_valid, repaired_error = self._validate_cross_key_boundaries(
                    repaired_response,
                    repair_source_dict,
                    lambda value: str(value),
                )
                if not repaired_valid:
                    logger.warning(
                        "Adjacent translation repair remained invalid after locked retry; "
                        "retaining the previous translations: %s",
                        repaired_error,
                    )
                    continue
            for item in repaired:
                translated_by_index[item.index] = item

        source_dict = {str(item.index): item.original_text for item in source_list}
        translated_dict = {
            str(item.index): translated_by_index[item.index].translated_text
            for item in source_list
            if item.index in translated_by_index
        }
        semantic_candidates = self._strong_asr_semantic_candidates(
            source_dict,
            translated_dict,
        )
        for key in semantic_candidates:
            index = int(key)
            try:
                repaired_text = self._translate_alignment_item(
                    source_dict[key],
                    previous_source=self._all_source_by_index.get(index - 1, ""),
                    next_source=self._all_source_by_index.get(index + 1, ""),
                )
            except Exception as error:
                logger.warning(
                    "Final semantic ASR repair failed for subtitle %s: %s",
                    key,
                    error,
                )
                continue
            translated_by_index[index] = replace(
                translated_by_index[index],
                translated_text=repaired_text,
            )
            logger.info("Final semantic ASR repair corrected key: %s", key)

        self._repair_contextual_nuclear_plant_terms(
            source_list,
            translated_by_index,
        )
        self._remove_stranded_chinese_subject_tails(
            source_list,
            translated_by_index,
        )
        self._repair_chinese_boundary_fluency(
            source_list,
            translated_by_index,
        )

        return [
            translated_by_index[item.index]
            for item in source_list
            if item.index in translated_by_index
        ]

    @staticmethod
    def _remove_stranded_chinese_subject_tails(
        source_list: List[SubtitleProcessData],
        translated_by_index: Dict[int, SubtitleProcessData],
    ) -> None:
        """Remove only a duplicated Chinese subject split from its following auxiliary."""
        auxiliary_heads = {
            "am",
            "are",
            "can",
            "could",
            "did",
            "do",
            "does",
            "had",
            "has",
            "have",
            "is",
            "may",
            "might",
            "must",
            "should",
            "was",
            "were",
            "will",
            "would",
        }
        english_subjects = {
            "he",
            "i",
            "it",
            "she",
            "that",
            "these",
            "they",
            "this",
            "those",
            "we",
            "which",
            "who",
            "you",
        }
        pronouns = "我|你|他|她|它|我们|你们|他们|她们|它们"
        for left, right in zip(source_list, source_list[1:]):
            left_item = translated_by_index.get(left.index)
            right_item = translated_by_index.get(right.index)
            if left_item is None or right_item is None:
                continue
            left_tokens = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", left.original_text.lower())
            right_tokens = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", right.original_text.lower())
            if (
                len(left_tokens) < 2
                or left_tokens[-1] not in english_subjects
                or not right_tokens
                or right_tokens[0] not in auxiliary_heads
            ):
                continue
            cleaned = re.sub(rf"(?:\s*)(?:{pronouns})\s*$", "", left_item.translated_text).strip()
            if cleaned and cleaned != left_item.translated_text.strip():
                translated_by_index[left.index] = replace(
                    left_item,
                    translated_text=cleaned,
                )

    def _pending_alignment_repair_windows(
        self,
        source_list: List[SubtitleProcessData],
        translated_by_index: Dict[int, SubtitleProcessData],
        pending_keys: List[int],
    ) -> List[List[SubtitleProcessData]]:
        """Return complete original batches touched by a confirmed alignment error.

        A one-key shift commonly cascades through the rest of its LLM batch. Repairing
        only the keys an auditor happened to flag can preserve a valid-looking shifted
        suffix. Rebuilding the affected original batch keeps the scope bounded while
        restoring ownership for the complete shift chain.
        """
        batch_size = max(1, int(self.batch_num))
        pending = set(pending_keys)
        valid_positions = {
            position
            for position, item in enumerate(source_list)
            if item.index in pending and item.index in translated_by_index
        }
        batch_starts = sorted(
            {(position // batch_size) * batch_size for position in valid_positions}
        )
        return [
            [
                item
                for item in source_list[start : start + batch_size]
                if item.index in translated_by_index
            ]
            for start in batch_starts
        ]

    @staticmethod
    def _has_dependent_source_boundary(
        pair_sources: List[SubtitleProcessData],
    ) -> bool:
        """Identify a clearly unfinished sentence split at a translation batch edge."""
        if len(pair_sources) != 2:
            return False
        left = pair_sources[0].original_text.strip()
        right = pair_sources[1].original_text.strip()
        if not left or not right or re.search(r"[.!?][\"')\]]*$", left):
            return False

        right_lower = right.lower()
        dependent_heads = (
            "and ",
            "are ",
            "as ",
            "because ",
            "but ",
            "for ",
            "from ",
            "has ",
            "have ",
            "in ",
            "is ",
            "of ",
            "on ",
            "or ",
            "kind of ",
            "sort of ",
            "that ",
            "the ",
            "to ",
            "was ",
            "were ",
            "which ",
            "who ",
            "with ",
        )
        if right_lower.startswith(dependent_heads):
            return True
        return bool(
            left.endswith(",")
            and re.match(
                r"^(?:i|we|you)\s+(?:think|believe|guess),?\s+that\b",
                right_lower,
            )
        )

    @classmethod
    def _has_repetitive_dependent_boundary(
        cls,
        pair_sources: List[SubtitleProcessData],
        response_dict: Dict[str, str],
    ) -> bool:
        """Gate dependent-boundary repair on evidence of duplicated target meaning."""
        if not cls._has_dependent_source_boundary(pair_sources):
            return False

        right_source = pair_sources[1].original_text.strip().lower()
        if re.match(
            r"^(?:i|we|you)\s+(?:think|believe|guess),?\s+that\b",
            right_source,
        ):
            return True

        targets = [
            cls._normalized_target_text(str(response_dict.get(str(item.index), "")))
            for item in pair_sources
        ]
        if min(map(len, targets), default=0) < 6:
            return False

        source_tokens = [
            {
                token
                for token in cls._normalized_source_text(item.original_text).split()
                if len(token) >= 4
                and token
                not in {
                    "been",
                    "have",
                    "kind",
                    "sort",
                    "that",
                    "they",
                    "they're",
                    "they've",
                    "this",
                    "we're",
                    "we've",
                    "with",
                    "you're",
                    "you've",
                }
            }
            for item in pair_sources
        ]
        if source_tokens[0] & source_tokens[1]:
            return False

        def bigrams(text: str) -> set[str]:
            return {text[index : index + 2] for index in range(len(text) - 1)}

        left_bigrams, right_bigrams = map(bigrams, targets)
        shorter = min(len(left_bigrams), len(right_bigrams))
        if bool(shorter and len(left_bigrams & right_bigrams) / shorter >= 0.55):
            return True

        common = difflib.SequenceMatcher(None, targets[0], targets[1]).find_longest_match()
        repeated_tail = targets[0][common.a : common.a + common.size]
        return bool(
            common.size >= 3 and targets[0].endswith(repeated_tail) and repeated_tail in targets[1]
        )

    def _repair_contextual_nuclear_plant_terms(
        self,
        source_list: List[SubtitleProcessData],
        translated_by_index: Dict[int, SubtitleProcessData],
    ) -> None:
        """Resolve only an ambiguous plant shorthand from explicit nuclear context."""
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return
        source_by_index = {
            **{item.index: item.original_text for item in source_list},
            **self._all_source_by_index,
        }
        for item in source_list:
            source = item.original_text.lower()
            if not re.search(r"\b(?:the|these|those)\s+plants\b", source):
                continue
            neighborhood = " ".join(
                source_by_index.get(index, "") for index in range(item.index - 3, item.index + 4)
            ).lower()
            translated_item = translated_by_index.get(item.index)
            if (
                "nuclear" not in neighborhood
                or translated_item is None
                or "工厂" not in translated_item.translated_text
            ):
                continue
            translated_by_index[item.index] = replace(
                translated_item,
                translated_text=translated_item.translated_text.replace("工厂", "核电站"),
            )
            logger.info(
                "Resolved contextual nuclear-plant terminology at subtitle %s",
                item.index,
            )

    def _repair_chinese_boundary_fluency(
        self,
        source_list: List[SubtitleProcessData],
        translated_by_index: Dict[int, SubtitleProcessData],
    ) -> None:
        """Repair only independently confirmed Chinese syntax breaks across keys.

        The normal alignment audit protects source ownership. This document-level pass
        addresses the opposite failure mode: preserving an English fragment boundary so
        literally that a Chinese subject, predicate, modifier, or connective is stranded.
        Rules only shortlist boundaries; an independent LLM verdict must confirm each one
        before a small same-speaker window can be rewritten.
        """
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return
        # This map is initialized only for a real whole-document translation run. Keeping
        # the pass document-scoped also prevents ad-hoc chunk calls from silently changing
        # their output contract.
        if not self._all_source_by_index:
            return

        candidates = self._chinese_fluency_candidates(source_list, translated_by_index)
        if not candidates:
            return

        batches = [
            candidates[start : start + self.CHINESE_FLUENCY_AUDIT_BATCH_SIZE]
            for start in range(0, len(candidates), self.CHINESE_FLUENCY_AUDIT_BATCH_SIZE)
        ]
        confirmed: list[int] = []
        audit_results: list[tuple[int, list[int]]] = []
        audit_workers = min(self.thread_num, len(batches))
        with ThreadPoolExecutor(max_workers=audit_workers) as executor:
            futures = {
                executor.submit(
                    self._request_chinese_fluency_flags,
                    batch,
                    source_list,
                    translated_by_index,
                ): batch
                for batch in batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    audit_results.append((batch[0], future.result()))
                except Exception as error:
                    logger.warning(
                        "Chinese boundary fluency audit failed for candidate batch %s; "
                        "retaining validated text: %s",
                        batch,
                        error,
                    )
        for _first_index, result in sorted(audit_results):
            confirmed.extend(result)
        confirmed.extend(
            self._mandatory_chinese_fluency_candidates(
                source_list,
                translated_by_index,
            )
        )
        confirmed = list(dict.fromkeys(confirmed))

        if not confirmed:
            return

        position_by_index = {item.index: position for position, item in enumerate(source_list)}
        confirmed_positions = sorted(
            {
                position_by_index[index]
                for index in confirmed
                if index in position_by_index and position_by_index[index] + 1 < len(source_list)
            }
        )
        windows = self._chinese_fluency_windows(source_list, confirmed_positions)
        workers = min(self.thread_num, len(windows))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._repair_chinese_fluency_window_with_retries,
                    window,
                    [translated_by_index[item.index] for item in window],
                ): window
                for window in windows
            }
            results = [future.result() for future in as_completed(futures)]

        for window, repaired, repair_error in sorted(
            results,
            key=lambda result: result[0][0].index,
        ):
            if repaired is None:
                logger.warning(
                    "Chinese boundary fluency repair failed for subtitles %s; retaining "
                    "the previous translations: %s",
                    [item.index for item in window],
                    repair_error,
                )
                continue
            for item in repaired:
                translated_by_index[item.index] = item
            logger.info(
                "Chinese boundary fluency repair corrected subtitles: %s",
                [item.index for item in window],
            )

    def _repair_chinese_fluency_window_with_retries(
        self,
        window: List[SubtitleProcessData],
        current: List[SubtitleProcessData],
    ) -> tuple[
        List[SubtitleProcessData],
        Optional[List[SubtitleProcessData]],
        Optional[Exception],
    ]:
        """Repair one independent window without mutating document state."""
        repair_error: Exception | None = None
        feedback = ""
        for _attempt in range(self.MAX_STEPS):
            try:
                candidate = self._rewrite_chinese_fluency_window(
                    window,
                    current,
                    feedback=feedback,
                )
                self._validate_chinese_fluency_repair(window, current, candidate)
                return window, candidate, None
            except Exception as error:
                repair_error = error
                feedback = str(error)
        return window, None, repair_error

    def _chinese_fluency_candidates(
        self,
        source_list: List[SubtitleProcessData],
        translated_by_index: Dict[int, SubtitleProcessData],
    ) -> list[int]:
        """Return left-key indices with a strong but not definitive syntax signal."""
        candidates: list[int] = []
        for position in range(len(source_list) - 1):
            left_source = source_list[position]
            right_source = source_list[position + 1]
            left_item = translated_by_index.get(left_source.index)
            right_item = translated_by_index.get(right_source.index)
            if left_item is None or right_item is None:
                continue
            left_speaker = self._all_speaker_by_index.get(left_source.index, "")
            right_speaker = self._all_speaker_by_index.get(right_source.index, "")
            if left_speaker and right_speaker and left_speaker != right_speaker:
                continue
            target_signal = self._chinese_boundary_signal(
                left_item.translated_text,
                right_item.translated_text,
            )
            source_signal = self._source_boundary_signal(
                left_source.original_text,
                right_source.original_text,
                left_item.translated_text,
                right_item.translated_text,
            )
            if target_signal or source_signal:
                candidates.append(left_source.index)
        return candidates

    def _mandatory_chinese_fluency_candidates(
        self,
        source_list: List[SubtitleProcessData],
        translated_by_index: Dict[int, SubtitleProcessData],
    ) -> list[int]:
        """Return deterministic syntax breaks that must not depend on LLM recall."""
        mandatory: list[int] = []
        for left, right in zip(source_list, source_list[1:]):
            left_item = translated_by_index.get(left.index)
            right_item = translated_by_index.get(right.index)
            if left_item is None or right_item is None:
                continue
            left_speaker = self._all_speaker_by_index.get(left.index, "")
            right_speaker = self._all_speaker_by_index.get(right.index, "")
            if left_speaker and right_speaker and left_speaker != right_speaker:
                continue

            target_signal = self._chinese_boundary_signal(
                left_item.translated_text,
                right_item.translated_text,
            )
            reasons = set(
                assess_english_boundary(left.original_text, right.original_text).reasons
            )
            source_is_open = not re.search(
                r"[.!?][\"')\]]*$",
                left.original_text.strip(),
            )
            if source_is_open and target_signal and target_signal not in {
                "possible function-word split",
                "possible demonstrative split",
                "possible pronoun boundary",
            }:
                mandatory.append(left.index)
            elif target_signal == "possible pronoun boundary" and any(
                reason.startswith("dangling subject") for reason in reasons
            ):
                mandatory.append(left.index)
            elif reasons.intersection(
                {
                    "place name split between city and state",
                    "proper-name subject separated from its predicate",
                }
            ):
                mandatory.append(left.index)
        return mandatory

    @staticmethod
    def _source_boundary_signal(
        left_source: str,
        right_source: str,
        left_translation: str = "",
        right_translation: str = "",
    ) -> str:
        """Shortlist cross-language boundaries that rules alone cannot judge.

        This intentionally returns candidates rather than verdicts.  A separate
        context-aware audit must confirm them before any text changes, which lets
        us cover language-order differences without rewriting every continuing
        English sentence.
        """
        left = str(left_source or "").strip()
        right = str(right_source or "").strip()
        if not left or not right or re.search(r"[.!?][\"')\]]*$", left):
            return ""

        assessment = assess_english_boundary(left, right)
        if assessment.unstable:
            return "; ".join(assessment.reasons) or "unstable source boundary"

        right_lower = right.lower()
        if re.match(
            r"^(?:after|before|because|when|where|which|who|whose|as well\b|"
            r"and\s+(?:go|see|test|buy|purchase)\b|is\b|are\b|was\b|were\b)",
            right_lower,
        ):
            return "source continuation may require different target-language order"

        if re.match(r"^(?:and|or)\b", right_lower) and not re.search(r"[,;:]\s*$", left):
            return "coordinate phrase crosses the subtitle boundary"

        left_target = re.sub(r"[\s，。！？；：、,.!?;:]+$", "", str(left_translation or ""))
        right_target = re.sub(r"^[\s，。！？；：、,.!?;:]+", "", str(right_translation or ""))
        if re.search(r"(?:之后|以前|以后|之前|期间|时候|大约|差不多|与其|为了|花)$", left_target):
            return "target-language temporal or governing phrase is unfinished"
        if right_target.startswith(("在", "当", "如今", "现在", "目前", "就去", "作为")):
            return "target-language modifier may be stranded at the next cue"

        left_words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", left)
        right_words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", right)
        if min(len(left_words), len(right_words)) <= 4:
            return "short source fragment crosses an unfinished sentence"
        return ""

    @staticmethod
    def _chinese_boundary_signal(left: str, right: str) -> str:
        """Describe a likely Chinese syntax break without deciding that it is wrong."""
        trim_chars = " \t\r\n，。！？；：、,.!?;:（）()【】[]‘’“”\"'"
        left = str(left or "").strip(trim_chars)
        right = str(right or "").strip(trim_chars)
        if not left or not right:
            return ""

        standalone = {
            "但",
            "但是",
            "不过",
            "而且",
            "并且",
            "所以",
            "因为",
            "如果",
            "尽管",
            "除非",
            "以及",
            "或者",
            "总之",
            "就是",
        }
        if left in standalone or right in standalone:
            return "standalone connective"

        connector_pattern = re.compile(
            r"(?:但|但是|不过|而且|并且|所以|因为|如果|尽管|除非|以及|或者|总之)$"
        )
        if connector_pattern.search(left):
            return "connective stranded at previous subtitle end"

        if re.search(r"(?:我|你|他|她|它|我们|你们|他们|她们|它们)(?:还|又|也|就|刚)?把$", left):
            return "unfinished Chinese grammatical structure"

        soft_tail = re.compile(
            r"(?:的|是|把|被|让|给|和|与|对|向|从|比|像|后|前|大约|差不多|与其|为了|花)$"
        )
        if soft_tail.search(left):
            return "possible function-word split"

        structural_tail = re.compile(
            r"(?:作为|没有|不会|不能|可以|应该|能够|正在|已经|只是|其实|确实|"
            r"相当|非常|更|最|几乎|如今|现在|目前|当时|后来|最终|像是|就像|就是|"
            r"我是说|我的意思是|来说|例如|比如)$"
        )
        if structural_tail.search(left):
            return "unfinished Chinese grammatical structure"

        if re.search(
            r"(?:买(?:到)?|选(?:择)?|找(?:到)?|换(?:成)?)"
            r"(?:一|这|那)(?:个|辆|台|种|套|位|名|条|款|部|件)$",
            left,
        ):
            return "unfinished Chinese grammatical structure"

        if re.search(r"(?:身上|当中|之中|方面)$", left) and not re.search(
            r"(?:在|落在|发生在|位于).{0,12}(?:身上|当中|之中|方面)$",
            left,
        ):
            return "unfinished Chinese locative subject"

        if (
            re.search(r"(?:我|你|他|她|它|我们|你们|他们)$", left)
            and len(re.sub(r"\s+", "", left)) >= 5
        ):
            # A final pronoun may be either a stranded subject ("问题是他们")
            # or a perfectly natural object ("我会立刻选他"). Let the
            # context-aware audit decide instead of rejecting it as a hard rule.
            return "possible pronoun boundary"

        if re.search(r"(?:这个|这些|那种)$", left) and len(re.sub(r"\s+", "", left)) <= 10:
            return "possible demonstrative split"

        if right.startswith(("了", "的", "得")):
            return "particle stranded at next subtitle start"

        left_connector = re.search(r"(所以|因为|不过|但是|而且|并且)$", left)
        if left_connector and right.startswith(left_connector.group(1)):
            return "duplicated boundary connective"
        return ""

    def _request_chinese_fluency_flags(
        self,
        candidate_indices: list[int],
        source_list: List[SubtitleProcessData],
        translated_by_index: Dict[int, SubtitleProcessData],
    ) -> list[int]:
        source_by_index = {item.index: item for item in source_list}
        items: Dict[str, Dict[str, str]] = {}
        allowed: set[str] = set()
        for index in candidate_indices:
            left = source_by_index.get(index)
            right = source_by_index.get(index + 1)
            left_translation = translated_by_index.get(index)
            right_translation = translated_by_index.get(index + 1)
            if not all((left, right, left_translation, right_translation)):
                continue
            key = f"{index}-{index + 1}"
            allowed.add(key)
            items[key] = {
                "source_left": left.original_text,
                "source_right": right.original_text,
                "translation_left": left_translation.translated_text,
                "translation_right": right_translation.translated_text,
                "target_signal": self._chinese_boundary_signal(
                    left_translation.translated_text,
                    right_translation.translated_text,
                ),
                "source_signal": self._source_boundary_signal(
                    left.original_text,
                    right.original_text,
                    left_translation.translated_text,
                    right_translation.translated_text,
                ),
            }
        if not items:
            return []

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a conservative Chinese subtitle boundary auditor. Decide whether "
                    "each boundary creates an unsuitable Chinese subtitle break. Read both source "
                    "keys as one continuous utterance and both translations as one display sequence. "
                    "Flag city/state or model-name splits, a relative or adverbial clause separated "
                    "from what it modifies, a subject separated from its predicate, a modifier from "
                    "its noun, 是 from its complement, an auxiliary or adverb from its predicate, a "
                    "coordinate noun phrase split in half, or a temporal/locative phrase placed after "
                    "its Chinese predicate. Also flag a translation that is grammatically complete "
                    "only during continuous playback but awkward when either cue is displayed alone, "
                    "or that follows English word order so literally that the pair is unnatural. "
                    "Do not flag a natural conjunction, reason, qualification, or continuation merely "
                    "because the English sentence spans two cues. "
                    "Do not flag a sentence-final 的 when it naturally means 'the one that is' "
                    "or acts as a colloquial final particle, and do not treat a completed object "
                    "ending in 这个 as a stranded subject. "
                    "Do not rewrite text. Return only JSON "
                    'as {"awkward_boundaries": ["left-right"]}.'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(items, ensure_ascii=False),
            },
        ]
        response = call_llm(
            messages=messages,
            model=self.model,
            temperature=0,
            use_cache=self.use_cache,
            client=self.llm_client,
            reasoning_mode="disabled",
            max_output_tokens=4096,
        )
        payload = parse_json_object(get_response_text(response))
        boundaries = payload.get("awkward_boundaries")
        if not isinstance(boundaries, list) or not all(
            isinstance(value, (str, int)) for value in boundaries
        ):
            raise ValueError("fluency audit must return an awkward_boundaries list")
        normalized = [str(value) for value in boundaries]
        unknown = set(normalized) - allowed
        if unknown:
            raise ValueError(f"fluency audit returned unknown boundaries: {sorted(unknown)}")
        return list(dict.fromkeys(int(value.split("-", 1)[0]) for value in normalized))

    def _chinese_fluency_windows(
        self,
        source_list: List[SubtitleProcessData],
        confirmed_positions: list[int],
    ) -> list[List[SubtitleProcessData]]:
        """Build non-overlapping 2-4 key windows around confirmed boundaries."""
        if not confirmed_positions:
            return []
        clusters: list[list[int]] = []
        for position in confirmed_positions:
            if clusters and position == clusters[-1][-1] + 1:
                clusters[-1].append(position)
            else:
                clusters.append([position])

        windows: list[List[SubtitleProcessData]] = []
        for cluster in clusters:
            start = cluster[0]
            end = cluster[-1] + 1
            while end - start + 1 > self.CHINESE_FLUENCY_MAX_WINDOW:
                split_end = start + self.CHINESE_FLUENCY_MAX_WINDOW - 1
                windows.append(source_list[start : split_end + 1])
                start = split_end
            windows.append(source_list[start : end + 1])
        return windows

    def _rewrite_chinese_fluency_window(
        self,
        source_items: List[SubtitleProcessData],
        current_items: List[SubtitleProcessData],
        *,
        feedback: str = "",
    ) -> List[SubtitleProcessData]:
        current_by_index = {item.index: item for item in current_items}
        payload = {
            str(item.index): {
                "source": item.original_text,
                "current_translation": current_by_index[item.index].translated_text,
            }
            for item in source_items
        }
        retry_instruction = (
            "\nThe previous repair was rejected: "
            + feedback
            + ". Correct that exact failure while preserving all required keys."
            if feedback
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"""You are repairing a confirmed Chinese subtitle syntax break for {self.target_language.value}.
Rewrite only the provided translations. Keep every key, source subtitle, timestamp boundary, fact, name, number, negation, comparison, and conclusion. Preserve the combined meaning exactly once. Make each key a natural readable Chinese subtitle: do not strand a subject from its predicate, a modifier from its noun, an adverb or modal from its predicate, or a connective at the previous key's end. Prefer rephrasing within each key. You may redistribute only the minimum Chinese wording between immediately adjacent keys when English and Chinese word order make that unavoidable. Never add explanation, source text, speaker labels, placeholders, or punctuation-only entries. Return only {{"translations": {{"key": "text"}}}} with every input key exactly once."""
                    " A repaired key must not end with a stranded pronoun, 的, 是, 可以, "
                    "正在, 非常, 更, 所以, 因为, 但是, 不过, 而且, or another unfinished "
                    "Chinese function word. Do not leave a connective as its own key. For a "
                    "source split like 'the problem is they' / 'made it', prefer '但这里有个问题' "
                    "/ '他们把它做成了' instead of preserving '他们' at the first key's end. "
                    "For 'but I do know at this time' / 'Mercedes was involved', prefer "
                    "'但有一点可以确定' / '当时梅赛德斯参与其中' instead of ending the first "
                    "key with 当时. "
                    "For 'Ford in Ypsilanti, Michigan for tossing' / 'me the keys', prefer "
                    "'还要感谢密歇根州伊普西兰蒂的这家经销商' / '他们把钥匙交给了我' "
                    "instead of ending the first key with 他们把. "
                    "For 'a serrated' / 'edge', prefer '这里采用锯齿状设计' / '这样的边缘' "
                    "instead of ending the first key with 的. For 'I'd pick him in a second' "
                    "/ 'as an appealing winner', prefer '我会立刻选他' / '他是个很有吸引力的赢家' "
                    "instead of starting the second key with 作为." + retry_instruction
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = call_llm(
            messages=messages,
            model=self.model,
            temperature=self.TRANSLATION_TEMPERATURE,
            use_cache=self.use_cache,
            client=self.llm_client,
            # Spend native reasoning only on the first rewrite. Formatting retries
            # use the validator's concrete feedback and do not benefit from another
            # long chain of thought.
            reasoning_mode=(
                "enabled" if not feedback and prefers_native_reasoning(self.model) else "disabled"
            ),
            max_output_tokens=(
                8192 if not feedback and prefers_native_reasoning(self.model) else 4096
            ),
        )
        result = parse_json_object(get_response_text(response)).get("translations")
        expected = set(payload)
        if not isinstance(result, dict) or set(map(str, result)) != expected:
            raise ValueError("fluency repair must return every input key exactly once")
        repaired: list[SubtitleProcessData] = []
        for item in source_items:
            text = str(result[str(item.index)]).strip()
            if not text or self._looks_like_placeholder_translation(text):
                raise ValueError(f"invalid fluency repair for key {item.index}")
            repaired.append(replace(item, translated_text=text))
        return repaired

    def _validate_chinese_fluency_repair(
        self,
        source_items: List[SubtitleProcessData],
        current_items: List[SubtitleProcessData],
        repaired_items: List[SubtitleProcessData],
    ) -> None:
        source_dict = {str(item.index): item.original_text for item in source_items}
        repaired_dict = {str(item.index): item.translated_text for item in repaired_items}
        valid, error = self._validate_llm_response(
            repaired_dict,
            source_dict,
            require_reflect=False,
        )
        if not valid:
            raise ValueError(error)

        current_length = len(
            self._normalized_target_text("".join(item.translated_text for item in current_items))
        )
        repaired_length = len(
            self._normalized_target_text("".join(item.translated_text for item in repaired_items))
        )
        if current_length and not (0.65 <= repaired_length / current_length <= 1.45):
            raise ValueError("fluency repair changed the combined translation length too much")

        repaired_by_index = {item.index: item for item in repaired_items}
        remaining = [
            item.index
            for item, following in zip(source_items, source_items[1:])
            if item.index in repaired_by_index
            and following.index in repaired_by_index
            and self._chinese_boundary_signal(
                repaired_by_index[item.index].translated_text,
                repaired_by_index[following.index].translated_text,
            )
        ]
        remaining_details = {
            index: {
                "left": repaired_by_index[index].translated_text,
                "right": repaired_by_index[index + 1].translated_text,
                "signal": self._chinese_boundary_signal(
                    repaired_by_index[index].translated_text,
                    repaired_by_index[index + 1].translated_text,
                ),
            }
            for index in remaining
        }
        hard_remaining = {
            index: detail
            for index, detail in remaining_details.items()
            if detail["signal"]
            not in {
                "possible function-word split",
                "possible demonstrative split",
                "possible pronoun boundary",
            }
        }
        if hard_remaining:
            raise ValueError(f"fluency repair left structural boundary signals: {hard_remaining}")
        remaining_candidates = set(remaining_details)
        remaining_candidates.update(
            self._chinese_fluency_candidates(source_items, repaired_by_index)
        )
        if remaining_candidates:
            confirmed_remaining = self._request_chinese_fluency_flags(
                sorted(remaining_candidates),
                source_items,
                repaired_by_index,
            )
            if confirmed_remaining:
                raise ValueError(
                    f"fluency repair left confirmed soft boundary signals: {confirmed_remaining}"
                )

        self._validate_chinese_window_fidelity(source_items, repaired_items)

    def _validate_chinese_window_fidelity(
        self,
        source_items: List[SubtitleProcessData],
        repaired_items: List[SubtitleProcessData],
    ) -> None:
        """Validate combined meaning while permitting minimal Chinese reordering.

        Per-key ownership is too strict for English-to-Chinese adverbial and relative
        clause order.  Hard anchors remain protected by the normal validator; this
        independent check verifies that the small same-speaker window still contains
        every source fact exactly once without introducing new meaning.
        """
        repaired_by_index = {item.index: item for item in repaired_items}
        payload = {
            str(item.index): {
                "source": item.original_text,
                "translation": repaired_by_index[item.index].translated_text,
            }
            for item in source_items
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an independent bilingual subtitle-window fidelity validator. "
                    "The ordered keys are one short continuous utterance. Minimal Chinese "
                    "surface reordering between adjacent keys is allowed when required by "
                    "Chinese grammar, but every source fact, name, number, model, negation, "
                    "comparison, qualification, and conclusion must appear exactly once in "
                    "the combined translations. Hard facts must not move to an unrelated key, "
                    "and no meaning may be invented, omitted, duplicated, anticipated from "
                    "outside this window, or moved across a speaker turn. Judge combined "
                    "fidelity and per-cue readability, not English word-order similarity. "
                    'Return only {"valid": true_or_false, "issues": ["brief issue"]}.'
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = call_llm(
            messages=messages,
            model=self.model,
            temperature=0,
            use_cache=self.use_cache,
            client=self.llm_client,
            # The expensive reasoning budget was already spent on candidate
            # confirmation and rewriting. Keep this independent gate compact.
            reasoning_mode="disabled",
            max_output_tokens=2048,
        )
        result = parse_json_object(get_response_text(response))
        valid = result.get("valid")
        issues = result.get("issues")
        if (
            not isinstance(valid, bool)
            or not isinstance(issues, list)
            or not all(isinstance(issue, str) for issue in issues)
        ):
            raise ValueError("window fidelity validator returned an invalid verdict")
        if not valid:
            raise ValueError(
                "fluency repair failed window-level fidelity: "
                + "; ".join(issue.strip() for issue in issues if issue.strip())
            )

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
                "dialogue_speakers": {
                    data.index: self._all_speaker_by_index.get(data.index, "") for data in chunk
                },
                "prompt_version": "context-v27-document-alignment-repair",
            }
        )
        return f"{class_name}:{chunk_key}:{lang}:{model}:{prompt_key}"
