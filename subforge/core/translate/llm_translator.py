"""LLM subtitle translator with structured validation and recovery."""

import difflib
import json
import re
import threading
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import openai

from subforge.core.llm import call_llm, get_response_text, parse_json_object
from subforge.core.prompts import get_prompt
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

    def translate_subtitle(self, subtitle_data):
        self._fatal_provider_error.clear()
        self._fatal_provider_message = ""
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
        """Run sparse alignment checks only for reflective MiniMax M3 batches."""
        normalized_model = re.sub(r"[^a-z0-9]+", "", self.model.lower())
        return self.is_reflect and "minimaxm3" in normalized_model

    def _audit_reflective_alignment(
        self,
        subtitle_dict: Dict[str, str],
        translated_dict: Dict[str, str],
        *,
        initial_focus_keys: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Correct only translations that clearly belong to a neighboring key.

        MiniMax M3 can preserve every JSON key while shifting a run of translations
        by one key when the source contains fragments. This independent pass asks
        for sparse corrections, then subjects the combined result to all existing
        structural validators. Audit failure keeps the already validated result.
        """
        try:
            def audit_items(keys, translations=translated_dict):
                items = {}
                for key in keys:
                    item = {
                        "source": subtitle_dict[key],
                        "translation": translations[key],
                    }
                    if str(key).isdigit():
                        speaker = self._all_speaker_by_index.get(int(key), "")
                        if speaker:
                            item["speaker"] = speaker
                    items[key] = item
                return items

            ordered_keys = list(subtitle_dict)
            first_flags = self._request_alignment_flags(audit_items(ordered_keys))
            strong_outliers = self._strong_alignment_length_outliers(
                subtitle_dict,
                translated_dict,
            )
            initial_flags = list(dict.fromkeys([*first_flags, *strong_outliers]))
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
            confirmed = (set(first_flags) & set(confirmed_flags)) | set(strong_outliers)
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
                    "MiniMax M3 alignment flags were not confirmed: %s",
                    sorted(initial_flags),
                )
                return translated_dict

            candidate = dict(translated_dict)
            ordered_keys = list(subtitle_dict)
            for key in dict.fromkeys(misaligned_keys):
                position = ordered_keys.index(key)
                candidate[key] = self._translate_alignment_item(
                    subtitle_dict[key],
                    previous_source=subtitle_dict.get(ordered_keys[position - 1])
                    if position > 0
                    else "",
                    next_source=subtitle_dict.get(ordered_keys[position + 1])
                    if position + 1 < len(ordered_keys)
                    else "",
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
                    initial_feedback=(
                        "Sparse alignment corrections were still invalid: " + error
                    ),
                )
                candidate.update(
                    {str(item.index): item.translated_text for item in recovered}
                )
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
            unresolved_repairs = sorted(set(residual_flags) & set(misaligned_keys))
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
                unresolved_fallbacks = sorted(
                    set(fallback_flags) & set(unresolved_repairs)
                )
                if unresolved_fallbacks:
                    raise ValueError(
                        "alignment corrections did not pass source-only verification: "
                        f"{unresolved_fallbacks}"
                    )
            logger.info(
                "MiniMax M3 alignment audit corrected keys: %s",
                sorted(misaligned_keys, key=lambda key: int(key) if key.isdigit() else key),
            )
            return candidate
        except Exception as error:
            logger.warning("MiniMax M3 alignment audit was ignored: %s", error)
            return translated_dict

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
                token[:1].isupper() and not token.isupper()
                for token in source_words[1:]
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
        system_prompt = f"""You are a conservative bilingual subtitle alignment auditor for {self.target_language.value}.
Compare every source with the translation under the SAME key. Read the ordered items as a continuous transcript so you can detect a run shifted forward or backward by one key. Flag a key only when its translation clearly omits material source meaning, contains a clause owned by another key, or belongs to a neighboring key. A sentence fragment can have a fragmentary translation and is not an error. Different word order, natural compression, pronoun omission, and stylistic quality are not alignment errors. Names, numbers, negation, comparisons, and conclusions are strong ownership anchors. The optional speaker field is anonymous metadata and speaker changes are hard boundaries. Do not write translations or judge style.{focus_instruction} You MUST evaluate every input key and return ONLY {{\"alignment\": {{\"key\": true_or_false}}, \"misaligned_keys\": [\"key\"]}}. The alignment object must contain every input key exactly once; true means ownership is correct. misaligned_keys must contain exactly the keys marked false."""
        response = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps({"items": items}, ensure_ascii=False)},
            ],
            model=self.model,
            temperature=0.1,
            use_cache=self.use_cache,
            client=self.llm_client,
        )
        audit = parse_json_object(get_response_text(response))
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
        system_prompt = f"""Translate the exact source text into {self.target_language.value}.
Translate ONLY current_source. Use previous_source and next_source solely to resolve references, word sense, and terminology. They are read-only: never include one of their clauses unless it is also present in current_source. If current_source is a sentence fragment, return a natural fragment without completing it. Preserve names, model identifiers, numbers, and technical terms. Do not infer or add any clause that is absent from current_source. Return only the translation with no JSON, labels, reasoning, markdown, or notes."""
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
        payload = {
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
            )
            translated = get_response_text(response).strip()
            if not translated or self._looks_like_placeholder_translation(translated):
                last_error = "alignment item translation was empty or a placeholder"
            else:
                translated = self._apply_alignment_role_hint(translated, role_hint)
                valid, error = self._validate_llm_response(
                    {"1": translated},
                    {"1": source},
                    require_reflect=False,
                )
                if valid:
                    return translated
                last_error = error
            if attempt == 0:
                messages.extend(
                    [
                        {"role": "assistant", "content": translated},
                        {
                            "role": "user",
                            "content": (
                                f"Validation failed: {last_error}. Correct only that error. "
                                "Preserve the exact current_source boundary and return only "
                                "the complete corrected translation."
                            ),
                        },
                    ]
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
            self._all_speaker_by_index.get(int(key))
            for key in subtitle_dict
            if str(key).isdigit()
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
            self._all_speaker_by_index.get(int(key))
            for key in subtitle_dict
            if str(key).isdigit()
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
                    token_pattern = rf"(?<!\d){re.escape(token_compact)}(?!\d)"
                else:
                    token_pattern = (
                        rf"(?<![a-z0-9]){re.escape(token_compact)}(?![a-z0-9])"
                    )
                if re.search(token_pattern, compact, flags=re.IGNORECASE):
                    translated_owners.setdefault(token, set()).add(str(key))

        leaks = []
        for token, owners in source_owners.items():
            output_keys = translated_owners.get(token, set())
            leaked_keys = output_keys - owners
            leaks.extend(f"{key}:{token}" for key in sorted(leaked_keys))
        if leaks:
            return (
                False,
                "A number or model fact was duplicated into a different subtitle key. "
                "Keep each fact in the key that contains it in current_subtitles. "
                f"Cross-key duplicates: {leaks[:20]}",
            )

        if not check_adjacent_repetition:
            return True, ""

        ordered_keys = sorted(
            subtitle_dict,
            key=lambda key: int(key) if str(key).isdigit() else str(key),
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
                "of",
                "on",
                "or",
                "that",
                "the",
                "this",
                "to",
                "with",
            }
            left_tokens = set(
                self._normalized_source_text(subtitle_dict[left_key]).split()
            ) - stopwords
            right_tokens = set(
                self._normalized_source_text(subtitle_dict[right_key]).split()
            ) - stopwords
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
            if not left_speaker or left_speaker != right_speaker:
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
            left_source_tokens = self._normalized_source_text(
                subtitle_dict[left_key]
            ).split()
            right_source_tokens = self._normalized_source_text(
                subtitle_dict[right_key]
            ).split()
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
            if not left_speaker or left_speaker != right_speaker:
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
            if source_repeats_meaning(left_key, right_key):
                continue
            left_source_tokens = self._normalized_source_text(
                subtitle_dict[left_key]
            ).split()
            right_source_tokens = self._normalized_source_text(
                subtitle_dict[right_key]
            ).split()
            if (
                left_source_tokens
                and right_source_tokens
                and left_source_tokens[-1] == right_source_tokens[-1]
            ):
                continue
            duplicated_endings.append(
                f"{left_key}-{right_key}:{repeated_ending}"
            )
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
            left_speaker = (
                self._all_speaker_by_index.get(int(left_key), "")
                if left_key.isdigit()
                else ""
            )
            right_speaker = (
                self._all_speaker_by_index.get(int(right_key), "")
                if right_key.isdigit()
                else ""
            )
            if (
                left_speaker
                and left_speaker == right_speaker
                and source_repeats_meaning(left_key, right_key)
            ):
                continue
            target_ratio = difflib.SequenceMatcher(None, left_target, right_target).ratio()
            left_source = self._normalized_source_text(subtitle_dict[left_key])
            right_source = self._normalized_source_text(subtitle_dict[right_key])
            source_ratio = difflib.SequenceMatcher(None, left_source, right_source).ratio()
            target_common = difflib.SequenceMatcher(
                None, left_target, right_target
            ).find_longest_match().size
            common_share = target_common / min(len(left_target), len(right_target))
            repeated_phrase = (
                target_common >= 7
                and common_share >= 0.45
                and source_ratio < 0.45
                and common_share - source_ratio >= 0.15
            )
            contained_short_phrase = (
                shorter_target >= 6
                and common_share == 1.0
                and source_ratio < 0.45
                and common_share - source_ratio >= 0.35
            )
            if (
                target_ratio >= 0.68 and target_ratio - source_ratio >= 0.25
            ) or repeated_phrase or contained_short_phrase:
                duplicate_pairs.append(
                    f"{left_key}-{right_key} (target={target_ratio:.0%}, "
                    f"shared={common_share:.0%}, source={source_ratio:.0%})"
                )
        if duplicate_pairs:
            return (
                False,
                "Adjacent translations are substantially more repetitive than their source "
                "subtitles. Do not complete, repeat, or anticipate a neighboring key. "
                f"Suspicious pairs: {duplicate_pairs[:10]}",
            )
        return True, ""

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
                r"|\b\d{2,3}\b"
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
            equivalents = (
                {"一战", "第一次世界大战"}
                if roman == "I"
                else {"二战", "第二次世界大战"}
            )
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
                if _world_war_roman_preserved(original, token, translated_norm):
                    continue
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
        fallback_response = {
            str(data.index): data.translated_text for data in translated_items
        }
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
                raise PartialTranslationError(
                    f"Fallback translations failed cross-key validation: {error}",
                    completed=[],
                    failed_indices=[data.index for data in subtitle_chunk],
                ) from error
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
                "dialogue_speakers": {
                    data.index: self._all_speaker_by_index.get(data.index, "") for data in chunk
                },
                "prompt_version": "context-v19-title-fragment",
            }
        )
        return f"{class_name}:{chunk_key}:{lang}:{model}:{prompt_key}"
