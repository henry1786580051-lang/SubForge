"""LLM subtitle translator with structured validation and recovery."""

import difflib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import openai

from subforge.core.llm import (
    call_llm,
    get_response_history_message,
    get_response_text,
    is_glm_53_model,
    is_kimi_k3_model,
    is_lmstudio_qwen_38_request,
    is_nemotron_3_ultra_model,
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
from subforge.core.translate.context import TranslationContext, is_unsafe_global_term_source
from subforge.core.translate.guidance import (
    repair_mode_guidance,
    target_language_style_rules,
)
from subforge.core.translate.quality import (
    ORDERED_CHINESE_BOUNDARY_DETECTORS,
    CanonicalEvidenceSummary,
    ChineseBoundaryFeatures,
    PlannedReasoningMode,
    ProviderCapabilities,
    QualityDiagnostic,
    RepairPlan,
    RepairStrategy,
    ShadowRepairComparison,
    ShadowRepairObservation,
    ShadowRepairRecorder,
    ShadowRepairSummary,
    TranslationModePolicy,
    TranslationSession,
    boundary_diagnostic_from_legacy_message,
    build_translation_session,
    collect_canonical_evidence,
    detect_document_shortened_place,
    detect_semantic_action_mismatch,
    detect_visible_pause_boundary,
    inspect_placeholders,
    inspect_reasoning_leaks,
    inspect_reflect_value_schema,
    inspect_response_schema,
    inspect_target_script,
    is_untranslated_output,
    legacy_diagnostic_message,
    legacy_preserved_token_message,
    select_translation_mode_policy,
    translation_metadata_guidance,
)
from subforge.core.translate.quality.numbers import normalize_grouped_numbers
from subforge.core.translate.quality.preservation import (
    _CHINESE_ENTITY_ALIASES,
    _CHINESE_TOKEN_EQUIVALENTS,
    _SOURCE_TOKEN_EQUIVALENTS,
    inspect_preserved_tokens,
)
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
    CHINESE_FLUENCY_MAX_WINDOW = 6
    CHINESE_FLUENCY_ANCHORED_MAX_ATTEMPTS = 1
    CHINESE_FLUENCY_FRESH_MAX_ATTEMPTS = 1
    REASONING_REWRITE_MAX_OUTPUT_TOKENS = 6144
    ALIGNMENT_REPAIR_CONTEXT = 1
    MULTISPEAKER_HANDOFF_MAX_GAP_MS = 450
    CONTINUOUS_DISPLAY_GAP_MS = 120
    SEPARATED_DISPLAY_GAP_MS = 300
    REASONING_METRIC_KEYS = (
        "audit_requests",
        "rewrite_requests",
        "final_answers",
        "no_final_answers",
        "accepted_repairs",
        "rejected_repairs",
        "fallback_requests",
    )

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
        cache_namespace: str = "",
        collect_canonical_evidence_telemetry: bool = False,
    ):
        super().__init__(
            thread_num=thread_num,
            batch_num=batch_num,
            target_language=target_language,
            update_callback=update_callback,
            use_cache=use_cache,
            cache_namespace=cache_namespace,
        )

        self.model = model
        self.custom_prompt = custom_prompt
        self.is_reflect = is_reflect
        self.translation_context = translation_context or TranslationContext(
            custom_prompt=custom_prompt
        )
        self._all_source_by_index: Dict[int, str] = {}
        self._all_speaker_by_index: Dict[int, str] = {}
        self._all_language_by_index: Dict[int, str] = {}
        self._gap_after_index: Dict[int, int] = {}
        self._translation_session: Optional[TranslationSession] = None
        self._active_translation_policy: Optional[TranslationModePolicy] = None
        self.llm_client = llm_client
        self._collect_canonical_evidence_telemetry = bool(collect_canonical_evidence_telemetry)
        self._canonical_evidence_summary = CanonicalEvidenceSummary()
        self._fatal_provider_error = threading.Event()
        self._fatal_provider_message = ""
        self._pending_alignment_repair_keys: set[int] = set()
        self._pending_alignment_repair_lock = threading.Lock()
        self._reasoning_metrics_lock = threading.Lock()
        self._reasoning_metrics = self._empty_reasoning_metrics()
        self._shadow_repair_recorder = ShadowRepairRecorder()
        self._shadow_repair_context = threading.local()

    @classmethod
    def _empty_reasoning_metrics(cls) -> Dict[str, int]:
        return {key: 0 for key in cls.REASONING_METRIC_KEYS}

    def _reset_reasoning_metrics(self) -> None:
        with self._reasoning_metrics_lock:
            self._reasoning_metrics = self._empty_reasoning_metrics()

    def _record_reasoning_metric(self, key: str) -> None:
        if key not in self.REASONING_METRIC_KEYS:
            raise ValueError(f"Unsupported reasoning metric: {key}")
        with self._reasoning_metrics_lock:
            self._reasoning_metrics[key] += 1

    def reasoning_metrics(self) -> Dict[str, int]:
        """Return one task's selective-reasoning counters."""
        with self._reasoning_metrics_lock:
            return dict(self._reasoning_metrics)

    def _record_shadow_repair_plan(
        self,
        diagnostics: tuple[QualityDiagnostic, ...],
    ) -> RepairPlan:
        """Record a reproducible plan without changing legacy control flow."""
        plan = self._shadow_repair_recorder.plan_and_record(
            diagnostics,
            capabilities=ProviderCapabilities(supports_reasoning=self._prefers_native_reasoning()),
            session=self._translation_session,
            cancelled=not self.is_running,
        )
        self._shadow_repair_context.last_plan = plan
        return plan

    def _last_shadow_repair_plan(self) -> RepairPlan | None:
        plan = getattr(self._shadow_repair_context, "last_plan", None)
        return plan if isinstance(plan, RepairPlan) else None

    def _record_shadow_action_for_plan(
        self,
        plan: RepairPlan,
        *,
        strategy: RepairStrategy,
        reasoning_mode: PlannedReasoningMode,
    ) -> None:
        """Record an explicitly executed candidate action for one captured plan."""
        if self._last_shadow_repair_plan() == plan:
            self._shadow_repair_context.last_plan = None
        self._shadow_repair_recorder.record_legacy_action(
            plan,
            strategy=strategy,
            reasoning_mode=reasoning_mode,
        )

    def _record_shadow_legacy_action(
        self,
        *,
        strategy: RepairStrategy,
        reasoning_mode: PlannedReasoningMode,
    ) -> None:
        plan = getattr(self._shadow_repair_context, "last_plan", None)
        if not isinstance(plan, RepairPlan):
            return
        self._shadow_repair_context.last_plan = None
        self._shadow_repair_recorder.record_legacy_action(
            plan,
            strategy=strategy,
            reasoning_mode=reasoning_mode,
        )

    def shadow_repair_observations(self) -> tuple[ShadowRepairObservation, ...]:
        """Return aggregated, text-free Phase 5 shadow evidence."""
        return self._shadow_repair_recorder.snapshot()

    def shadow_repair_comparisons(self) -> tuple[ShadowRepairComparison, ...]:
        """Return aggregated planner-versus-legacy decisions without subtitle text."""
        return self._shadow_repair_recorder.comparison_snapshot()

    def shadow_repair_summary(self) -> ShadowRepairSummary:
        """Return bounded, text-free evidence for candidate admission checks."""
        return self._shadow_repair_recorder.summary()

    def canonical_evidence_summary(self) -> CanonicalEvidenceSummary:
        """Return text-free evidence about context canonicalization ownership."""
        return self._canonical_evidence_summary

    def _log_reasoning_metrics(self) -> None:
        metrics = self.reasoning_metrics()
        if not any(metrics.values()):
            return
        logger.info(
            "Selective reasoning summary: audits=%d, rewrites=%d, final_answers=%d, "
            "no_final_answers=%d, accepted=%d, rejected=%d, fallbacks=%d",
            metrics["audit_requests"],
            metrics["rewrite_requests"],
            metrics["final_answers"],
            metrics["no_final_answers"],
            metrics["accepted_repairs"],
            metrics["rejected_repairs"],
            metrics["fallback_requests"],
        )

    def translate_subtitle(self, subtitle_data):
        self._fatal_provider_error.clear()
        self._fatal_provider_message = ""
        self._reset_reasoning_metrics()
        self._shadow_repair_recorder = ShadowRepairRecorder()
        self._shadow_repair_context = threading.local()
        self._canonical_evidence_summary = CanonicalEvidenceSummary()
        with self._pending_alignment_repair_lock:
            self._pending_alignment_repair_keys.clear()
        self._all_source_by_index = {i: seg.text for i, seg in enumerate(subtitle_data.segments, 1)}
        if self._collect_canonical_evidence_telemetry:
            self._canonical_evidence_summary = collect_canonical_evidence(
                self.translation_context.terminology,
                self._all_source_by_index,
                parse_mapping=self._parse_context_asr_mapping,
                has_document_support=self._context_mapping_has_document_support,
            )
        self._translation_session = build_translation_session(
            subtitle_data,
            target_language=self.target_language,
            model=self.model,
        )
        self._active_translation_policy = select_translation_mode_policy(self._translation_session)
        self._all_language_by_index = {
            i: seg.language_code
            for i, seg in enumerate(subtitle_data.segments, 1)
            if seg.language_code
        }
        self._gap_after_index = {
            index: max(0, following.start_time - current.end_time)
            for index, (current, following) in enumerate(
                zip(subtitle_data.segments, subtitle_data.segments[1:]),
                1,
            )
        }
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
            self._log_reasoning_metrics()
            self._all_source_by_index = {}
            self._all_speaker_by_index = {}
            self._all_language_by_index = {}
            self._gap_after_index = {}
            self._translation_session = None
            self._active_translation_policy = None

    def _translation_mode_policy(self) -> TranslationModePolicy:
        if self._active_translation_policy is not None:
            return self._active_translation_policy
        return select_translation_mode_policy(
            speakers=self._all_speaker_by_index.values(),
            languages=self._all_language_by_index.values(),
        )

    def _is_multispeaker_document(self) -> bool:
        return self._translation_mode_policy().is_dialogue

    def _batch_translation_prompt_name(self, *, reflect: bool) -> str:
        """Keep the proven monologue prompt separate from dialogue tuning."""
        if is_lmstudio_qwen_38_request(self.model, self.llm_client):
            return "translate/qwen_38_local"
        if reflect and is_kimi_k3_model(self.model):
            return "translate/reflect_kimi_k3"
        base = "reflect" if reflect else "standard"
        return f"translate/{base}{self._translation_mode_policy().prompt_suffix}"

    def _prefers_native_reasoning(self) -> bool:
        """Select sparse reasoning with both model and endpoint capabilities."""
        return prefers_native_reasoning(self.model) or is_lmstudio_qwen_38_request(
            self.model,
            self.llm_client,
        )

    def _assistant_followup_message(
        self,
        response: Any,
        fallback_content: str,
    ) -> Dict[str, Any]:
        """Preserve K3 thinking history without changing other model flows."""
        if is_kimi_k3_model(self.model):
            try:
                return get_response_history_message(response)
            except ValueError:
                pass
        return {"role": "assistant", "content": fallback_content}

    def _is_edited_speaker_handoff(
        self,
        left: SubtitleProcessData,
        right: SubtitleProcessData,
    ) -> bool:
        """Identify a tightly edited speaker change inside one grammatical phrase.

        The speaker labels remain untouched. This only allows the translation
        quality pass to read the two turns together when a real edit cuts a
        dependent phrase between voices, as can happen in host-read adverts.
        """
        if not self._is_multispeaker_document():
            return False
        left_speaker = self._all_speaker_by_index.get(left.index, "")
        right_speaker = self._all_speaker_by_index.get(right.index, "")
        if not left_speaker or not right_speaker or left_speaker == right_speaker:
            return False
        if self._gap_after_index.get(left.index, self.MULTISPEAKER_HANDOFF_MAX_GAP_MS + 1) > (
            self.MULTISPEAKER_HANDOFF_MAX_GAP_MS
        ):
            return False
        if re.search(r"[.!?][\"')\]]*$", left.original_text.strip()):
            return False
        return (
            assess_english_boundary(
                left.original_text,
                right.original_text,
            ).risk
            >= 24
        )

    def _display_boundary_metadata(self, index: int) -> Dict[str, Any]:
        """Describe whether adjacent cues are perceived as one display sequence."""
        if index not in self._gap_after_index:
            return {}
        gap_ms = self._gap_after_index[index]
        if gap_ms <= self.CONTINUOUS_DISPLAY_GAP_MS:
            continuity = "continuous"
        elif gap_ms >= self.SEPARATED_DISPLAY_GAP_MS:
            continuity = "separated"
        else:
            continuity = "brief_pause"
        return {
            "gap_after_ms": gap_ms,
            "display_continuity": continuity,
        }

    @staticmethod
    def _dependent_boundary_repeats_meaning(left: str, right: str) -> bool:
        """Detect repeated Chinese content inside an already proven source dependency."""

        def bigrams(text: str) -> set[str]:
            compact = re.sub(r"[^\u3400-\u9fff]", "", str(text or ""))
            return {
                compact[position : position + 2]
                for position in range(len(compact) - 1)
                if compact[position : position + 2]
                not in {
                    "一个",
                    "一种",
                    "这个",
                    "那个",
                    "这是",
                    "就是",
                    "亿美",
                    "美元",
                    "万元",
                    "公里",
                    "平方",
                }
            }

        shared = bigrams(left) & bigrams(right)
        return len(shared) >= 2

    def _long_gap_chinese_boundary_signal(
        self,
        index: int,
        left: str,
        right: str,
    ) -> str:
        """Return only target fragments that become misleading across a real pause."""
        features = ChineseBoundaryFeatures.from_text(
            left,
            right,
            gap_ms=self._gap_after_index.get(index, 0),
        )
        match = detect_visible_pause_boundary(
            features,
            separated_gap_ms=self.SEPARATED_DISPLAY_GAP_MS,
        )
        return match.message if match is not None else ""

    def _target_boundary_signal(self, index: int, left: str, right: str) -> str:
        diagnostic = self._target_boundary_diagnostic(index, left, right)
        return diagnostic.message if diagnostic is not None else ""

    @staticmethod
    def _source_ends_with_parallel_plural_attributes(source: str) -> bool:
        return bool(
            re.search(
                r"\b(?:all(?:\s+different)?|different|various)\s+"
                r"[a-z][a-z'’-]*s\s+(?:and|or)\s+[a-z][a-z'’-]*s[,;:]?$",
                str(source or "").strip(),
                re.IGNORECASE,
            )
        )

    def _target_boundary_diagnostic(
        self,
        index: int,
        left: str,
        right: str,
    ) -> QualityDiagnostic | None:
        message = self._long_gap_chinese_boundary_signal(
            index,
            left,
            right,
        ) or self._chinese_boundary_signal(left, right)
        source = getattr(self, "_all_source_by_index", {}).get(index, "").strip()
        if (
            message == "possible function-word split"
            and re.search(r"的[\s，。！？；：、,.!?;:…]*$", left)
            and self._source_ends_with_parallel_plural_attributes(source)
        ):
            # English plural lists are complete nominal objects. A Chinese
            # rendering ending in 的 has instead turned that object into an
            # attributive phrase without a head noun.
            message = "nominal modifier is stranded"
        return boundary_diagnostic_from_legacy_message(
            message,
            cue_keys=(index, index + 1),
            evidence=(("left", left), ("right", right)),
        )

    def _translate_chunk(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """翻译字幕块"""
        if self._fatal_provider_error.is_set():
            raise RuntimeError(self._fatal_provider_message or "LLM provider request rejected")
        logger.debug(f"[+]正在翻译字幕: {subtitle_chunk[0].index} - {subtitle_chunk[-1].index}")

        # 转换为字典格式用于API调用
        subtitle_dict = {
            str(data.index): self._source_for_translation(data.original_text)
            for data in subtitle_chunk
        }

        # 获取提示词
        prompt = get_prompt(
            self._batch_translation_prompt_name(reflect=self.is_reflect),
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
        return self.is_reflect and not is_lmstudio_qwen_38_request(
            self.model,
            self.llm_client,
        )

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
                    item: Dict[str, Any] = {
                        "source": subtitle_dict[key],
                        "translation": translations[key],
                    }
                    if str(key).isdigit():
                        numeric_key = int(key)
                        speaker = self._all_speaker_by_index.get(numeric_key, "")
                        if speaker:
                            item["speaker"] = speaker
                        item.update(self._display_boundary_metadata(numeric_key))
                        previous_source = self._all_source_by_index.get(numeric_key - 1, "")
                        next_source = self._all_source_by_index.get(numeric_key + 1, "")
                        if previous_source:
                            item["previous_source"] = previous_source
                        if next_source:
                            item["next_source"] = next_source
                            source_signal = self._source_boundary_signal(
                                subtitle_dict[key],
                                next_source,
                                translations[key],
                                translations.get(str(numeric_key + 1), ""),
                            )
                            if source_signal:
                                item["source_boundary_signal"] = source_signal
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
            prose_candidates = self._strong_chinese_prose_candidates(
                subtitle_dict,
                translated_dict,
            )
            initial_flags = list(
                dict.fromkeys(
                    [
                        *first_flags,
                        *strong_outliers,
                        *semantic_candidates,
                        *prose_candidates,
                    ]
                )
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
            audit_flags = (
                set(first_flags) | set(confirmed_flags)
                if is_nemotron_3_ultra_model(self.model)
                else set(first_flags) & set(confirmed_flags)
            )
            confirmed = (
                audit_flags
                | set(strong_outliers)
                | set(semantic_candidates)
                | set(prose_candidates)
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
            repaired_keys: List[str] = []
            failed_repair_keys: List[str] = []
            for key in dict.fromkeys(misaligned_keys):
                position = ordered_keys.index(key)
                numeric_key = int(key) if key.isdigit() else None
                previous_source = (
                    subtitle_dict.get(ordered_keys[position - 1], "")
                    if position > 0
                    else self._all_source_by_index.get((numeric_key or 1) - 1, "")
                )
                next_source = (
                    subtitle_dict.get(ordered_keys[position + 1], "")
                    if position + 1 < len(ordered_keys)
                    else self._all_source_by_index.get((numeric_key or -1) + 1, "")
                )
                try:
                    candidate[key] = self._translate_alignment_item(
                        subtitle_dict[key],
                        source_key=key,
                        previous_source=previous_source,
                        next_source=next_source,
                        repair_hint=self._selective_semantic_repair_hint(
                            subtitle_dict[key],
                            translated_dict.get(key, ""),
                            previous_source=previous_source,
                            next_source=next_source,
                        ),
                    )
                    repaired_keys.append(key)
                except Exception as repair_error:
                    failed_repair_keys.append(key)
                    logger.warning(
                        "Alignment repair failed for key %s; other confirmed repairs will "
                        "continue: %s",
                        key,
                        repair_error,
                    )
            self._queue_alignment_repairs(failed_repair_keys)
            if not repaired_keys:
                return translated_dict
            valid, error = self._validate_llm_response(
                candidate,
                subtitle_dict,
                require_reflect=False,
            )
            if not valid:
                self._record_shadow_legacy_action(
                    strategy=RepairStrategy.BATCH_REBUILD,
                    reasoning_mode=PlannedReasoningMode.DISABLED,
                )
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
                    self._record_shadow_legacy_action(
                        strategy=RepairStrategy.NONE,
                        reasoning_mode=PlannedReasoningMode.DISABLED,
                    )
                    raise ValueError(error)

            residual_flags = self._request_alignment_flags(
                audit_items(
                    repaired_keys,
                    candidate,
                ),
                focused=True,
            )
            unresolved_repairs = sorted(
                (set(residual_flags) & set(repaired_keys))
                - set(semantic_candidates)
                - set(prose_candidates)
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
                    self._record_shadow_legacy_action(
                        strategy=RepairStrategy.NONE,
                        reasoning_mode=PlannedReasoningMode.DISABLED,
                    )
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
                sorted(repaired_keys, key=lambda key: int(key) if key.isdigit() else key),
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
        """Keep unresolved alignment shifts out of reusable translation caches."""
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
            if self._alignment_asr_hint(
                source,
                self._all_source_by_index.get(int(key) - 1, "") if key.isdigit() else "",
                self._all_source_by_index.get(int(key) + 1, "") if key.isdigit() else "",
            ):
                candidates.append(key)
                continue
            semantic_action = detect_semantic_action_mismatch(
                source,
                translated,
                previous_source=(
                    self._all_source_by_index.get(int(key) - 1, "") if key.isdigit() else ""
                ),
                next_source=(
                    self._all_source_by_index.get(int(key) + 1, "") if key.isdigit() else ""
                ),
            )
            if semantic_action:
                candidates.append(key)
                continue
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

            malformed_numeric_magnitude = bool(
                re.search(
                    r"\b\d{1,3}(?:,\d{3})+\s+"
                    r"(?:hundred|thousand|million|billion)\b",
                    source,
                    flags=re.IGNORECASE,
                )
            )
            if malformed_numeric_magnitude:
                # A grouped number already encodes its magnitude. A following
                # scale word is therefore a high-confidence ASR contradiction,
                # independent of topic (for example, "4,700 hundred"). The
                # alignment model still makes the final contextual decision.
                candidates.append(key)
                continue

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

    def _strong_chinese_prose_candidates(
        self,
        subtitle_dict: Dict[str, str],
        translated_dict: Dict[str, str],
    ) -> List[str]:
        """Select sparse, high-confidence calques for the existing reasoning repair path."""
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return []
        return [
            str(key)
            for key, source in subtitle_dict.items()
            if self._chinese_prose_repair_hint(
                str(source or ""),
                str(translated_dict.get(key, "") or ""),
            )
        ]

    def _chinese_prose_repair_hint(self, source: str, translated: str) -> str:
        """Explain a confirmed Chinese calque without prescribing one fixed translation."""
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return ""

        source_text = str(source or "")
        compact_target = re.sub(r"[\s，。！？；：、,.!?;:]+", "", str(translated or ""))
        if re.search(r"\bon\s+paper\b", source_text, re.IGNORECASE) and (
            "纸面" in compact_target
            or (
                re.search(r"\bon\s+social\s+media\b", source_text, re.IGNORECASE)
                and "理论" in compact_target
            )
        ):
            return (
                "The current Chinese is a literal 'on paper' calque. Infer whether the source "
                "means accounts, apparent evidence, or theoretical feasibility, then use the "
                "natural Chinese expression for that meaning. When it is coordinated with social "
                "media as evidence of apparent success, use a shared evidence/perspective frame "
                "and do not interpret it as 理论上. Do not retain 纸面 or 纸面上."
            )
        if re.search(
            r"\b(?:economic|financial|commercial)\s+sense\b",
            source_text,
            re.IGNORECASE,
        ) and re.search(r"(?:经济|财务|商业)(?:意义|含义)", compact_target):
            return (
                "Express practical economic or commercial viability as a natural Chinese "
                "predicate, not the nominal calque 经济/财务/商业意义 or 含义."
            )
        if re.search(
            r"\b(?:spark|ignite|inspire|awaken)\b.{0,100}"
            r"\b(?:family|children|kids|audience|community|team)\b",
            source_text,
            re.IGNORECASE,
        ) and re.search(
            r"用.{0,10}(?:家人|孩子|观众|社区|团队).{0,10}(?:点燃|激发|唤起)",
            compact_target,
        ):
            return (
                "The animate participant is the experiencer or beneficiary of the emotion, not "
                "an instrument used to create it. Preserve that semantic role in natural Chinese."
            )
        if re.search(
            r"\b(?:approach|architecture|design|concept|strategy|technology)\b.{0,100}"
            r"\b(?:product|delivered|delivery|implementation|implemented)\b",
            source_text,
            re.IGNORECASE,
        ) and re.search(r"(?:交付方式|进行交付|作为一种(?:产品|商品)|建筑作为)", compact_target):
            return (
                "Rewrite the abstract English noun chain as a direct Chinese verb or state "
                "describing how the idea is realized, deployed, or productized. Do not retain "
                "交付方式, 进行交付, or 作为一种产品."
            )
        if (
            re.search(r"\bhome\s+run\b", source_text, re.IGNORECASE)
            and re.search(r"\bball\s+game\b", source_text, re.IGNORECASE)
            and not (re.search(r"全垒打", translated) and re.search(r"(?:赛场|球场)", translated))
        ):
            return (
                "The source deliberately links 'home run' and 'ball game'. Preserve both halves "
                "as one natural Chinese sports image; keeping only the venue/field image or "
                "reducing the home run to generic success loses the rhetoric."
            )
        if re.search(r"\broaring\s+success\b", source_text, re.IGNORECASE) and (
            re.search(r"(?:轰动性成功|巨大成功|取得成功)", translated)
            or not re.search(r"(?:大获成功|引起轰动|风靡|一炮而红)", translated)
        ):
            return (
                "Preserve the emphatic force of 'roaring success' with concise vivid Chinese; "
                "a flat generic 巨大成功/取得成功 or unnatural 轰动性成功 rendering is insufficient."
            )
        if re.search(r"\bwith\s+a\s+vengeance\b", source_text, re.IGNORECASE) and not re.search(
            r"(?:来势|汹汹|猛烈|卷土重来|变本加厉)", translated
        ):
            return (
                "Preserve the forceful return or escalation conveyed by 'with a vengeance'; "
                "do not flatten it into a neutral return."
            )
        if re.search(r"\bDM\s+me\b", source_text, re.IGNORECASE) and re.search(
            r"(?:DM私信|私信DM)",
            compact_target,
            re.IGNORECASE,
        ):
            return (
                "DM is already expressed by the Chinese verb 私信. Use 私信我 or 给我发私信 "
                "once; do not duplicate the same action as DM私信 or 私信DM."
            )
        return ""

    def _deterministic_chinese_prose_fallback(self, source: str, translated: str) -> str:
        """Apply semantic-safe idiom cleanup only after sparse LLM repair is exhausted."""
        if self.target_language != TargetLanguage.SIMPLIFIED_CHINESE:
            return translated

        source_text = str(source or "")
        repaired = str(translated or "")
        if re.search(
            r"\bon\s+paper\s+and\s+on\s+social\s+media\b",
            source_text,
            re.IGNORECASE,
        ):
            repaired = re.sub(
                r"(?:从)?纸面(?:上)?和社交媒体(?:上)?(?:来)?看",
                "从账面和社交媒体来看",
                repaired,
                count=1,
            )
        if re.search(r"\broaring\s+success\b", source_text, re.IGNORECASE):
            repaired = re.sub(
                r"(?:取得|获得)(?:了)?巨大成功",
                "大获成功",
                repaired,
                count=1,
            )
            repaired = re.sub(
                r"是(?:一场|一次)?轰动性成功",
                "引起轰动",
                repaired,
                count=1,
            )
        if (
            re.search(r"\bhome\s+run\b", source_text, re.IGNORECASE)
            and re.search(r"\bball\s+game\b", source_text, re.IGNORECASE)
            and re.search(r"(?:赛场|球场)", repaired)
        ):
            repaired = re.sub(
                r"再(?:次)?创佳绩",
                "再打出一记全垒打",
                repaired,
                count=1,
            )
        return repaired

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
            "\nIf one source key ends inside a modifier, list, or other open phrase, distinguish "
            "legitimate minimal Chinese reordering within that adjacent pair from a real shift. "
            "Do not flag a pair when its combined meaning is complete exactly once and both cues "
            "remain readable; do flag duplicated, omitted, unrelated, or prematurely concluded "
            "material. Also flag an objectively wrong semantic-role assignment, such as turning "
            "the person who experiences an emotion into the instrument that causes it."
            " A source_boundary_signal is a deterministic dependency shortlist, not a verdict; "
            "when it identifies a modifier-head or other indivisible phrase, verify that material "
            "meaning from the following source was not moved into the current key."
            "\nIn addition to boundary alignment, mark a key false when its translation "
            "blindly follows an ASR rendering whose literal meaning is impossible in the "
            "explicit local topic. High-confidence cases include incompatible currency or "
            "number formatting, an impossible unit, an obvious homophone, or abbreviated "
            "model-year wording with one unambiguous contextual interpretation. Do not flag "
            "uncertain wording, unsupported proper-noun corrections, normal colloquial "
            "compression, or style preferences. Also flag a phonetic ASR variant of a name, "
            "model, or trim only when global terminology or repeated document evidence gives "
            "one unambiguous canonical form. A grammatically valid everyday noun may still be "
            "an obvious recognition error when its literal meaning conflicts with the surrounding "
            "parallel list, physical action, and recurring document subject, and one close spoken "
            "form uniquely fits all three. Flag that narrow case, but do not turn this into general "
            "copy-editing or replace merely unusual wording. The previous_source and next_source fields "
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

        self._record_reasoning_metric("audit_requests")
        response = call_llm(
            messages=messages,
            model=self.model,
            temperature=0.1,
            use_cache=self.use_cache,
            cache_namespace=self.cache_namespace,
            client=self.llm_client,
            # Classification needs a short exhaustive JSON verdict. Native
            # reasoning is reserved for the sparse rewrites selected below.
            reasoning_mode="disabled",
            max_output_tokens=4096,
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
        repair_hint: str = "",
        source_key: str = "1",
        allow_reasoning: bool = True,
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
        system_prompt += (
            "\nWhen confirmed_canonical_name is present, reproduce that exact Latin string in "
            "the translation. Do not translate, transliterate, abbreviate, or respell it."
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
        system_prompt += (
            "\nWhen native reasoning is available, use it internally to compare a faithful "
            "literal reading with a concise idiomatic subtitle. Choose the most natural wording "
            "that preserves the exact facts, tone, register, and current_source ownership. "
            "The priority is fidelity first, clear Chinese second, and elegance third; never "
            "trade accuracy for polish. Keep the internal analysis concise and reserve enough "
            "output budget for the final answer. Return only the final translation."
        )
        system_prompt += self._target_language_style_rules([source])
        role_hint = self._alignment_role_hint(source, previous_source, next_source)
        payload: Dict[str, Any] = {
            "previous_source": previous_source,
            "current_source": source,
            "next_source": next_source,
        }
        if repair_hint:
            payload["confirmed_repair_requirement"] = repair_hint
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
            canonical = self._confirmed_context_canonical(source)
            if asr_hint.get("kind") == "document_repeated_place_variant":
                canonical = str(asr_hint.get("canonical") or "").strip()
            if canonical:
                payload["confirmed_canonical_name"] = canonical
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error = "alignment item translation was invalid"
        for attempt in range(2):
            use_reasoning = allow_reasoning and attempt == 0 and self._prefers_native_reasoning()
            if use_reasoning:
                self._record_reasoning_metric("rewrite_requests")
            try:
                response = call_llm(
                    messages=messages,
                    model=self.model,
                    temperature=0.1,
                    use_cache=self.use_cache,
                    cache_namespace=self.cache_namespace,
                    client=self.llm_client,
                    reasoning_mode="enabled" if use_reasoning else "disabled",
                    max_output_tokens=(
                        self.REASONING_REWRITE_MAX_OUTPUT_TOKENS if use_reasoning else 4096
                    ),
                    **({"reasoning_effort": "low"} if use_reasoning else {}),
                )
            except Exception as error:
                if self._is_fatal_provider_error(error):
                    raise
                last_error = str(error).strip() or error.__class__.__name__
                if attempt == 0:
                    if use_reasoning:
                        self._record_reasoning_metric("no_final_answers")
                        self._record_reasoning_metric("rejected_repairs")
                        self._record_reasoning_metric("fallback_requests")
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous selective repair request failed temporarily: "
                                f"{last_error}. Retry the same exact task without reasoning and "
                                "return only the complete final translation."
                            ),
                        }
                    )
                    continue
                raise ValueError(last_error) from error
            try:
                translated = get_response_text(response).strip()
                if use_reasoning:
                    self._record_reasoning_metric("final_answers")
            except ValueError as error:
                translated = ""
                last_error = str(error)
                if use_reasoning:
                    self._record_reasoning_metric("no_final_answers")
            if not translated or self._looks_like_placeholder_translation(translated):
                if translated:
                    last_error = "alignment item translation was empty or a placeholder"
            else:
                translated = self._apply_alignment_role_hint(translated, role_hint)
                asr_error = self._validate_alignment_asr_hint(translated, asr_hint)
                if asr_error:
                    last_error = asr_error
                else:
                    validation_source = (
                        str(asr_hint.get("normalized_source") or source) if asr_hint else source
                    )
                    valid, error = self._validate_llm_response(
                        {source_key: translated},
                        {source_key: validation_source},
                        require_reflect=False,
                    )
                    if valid:
                        remaining_issue = self._selective_semantic_repair_hint(
                            source,
                            translated,
                            previous_source=previous_source,
                            next_source=next_source,
                        )
                        if repair_hint and remaining_issue:
                            last_error = remaining_issue
                            valid = False
                    if valid:
                        if use_reasoning:
                            self._record_reasoning_metric("accepted_repairs")
                        return translated
                    self._record_shadow_legacy_action(
                        strategy=(RepairStrategy.RETRY if attempt == 0 else RepairStrategy.NONE),
                        reasoning_mode=PlannedReasoningMode.DISABLED,
                    )
                    if error:
                        last_error = error
            if attempt == 0:
                if use_reasoning:
                    self._record_reasoning_metric("rejected_repairs")
                    self._record_reasoning_metric("fallback_requests")
                if translated:
                    messages.append(self._assistant_followup_message(response, translated))
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
        if re.match(
            r"^\s*that\s+is\s+(?:(?:quite|really|pretty|so|very)\s+)?"
            r"(?:good|nice|great|cool|beautiful)[,;]",
            source,
            flags=re.IGNORECASE,
        ):
            return (
                "The opening demonstrative evaluation refers to the object in previous_source. "
                "A new noun phrase after the comma starts a separate observation; do not attach "
                "the evaluation to that newly introduced object."
            )
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

    def _context_asr_variant(self, source: str) -> Dict[str, str]:
        """Apply only context terms explicitly labelled as probable ASR corrections."""
        for raw_line in self.translation_context.terminology.splitlines():
            mapping = self._parse_context_asr_mapping(raw_line)
            if not mapping:
                continue
            heard, canonical = mapping
            if not heard or not canonical or heard.casefold() == canonical.casefold():
                continue
            if not re.search(rf"(?<!\w){re.escape(heard)}(?!\w)", source, re.IGNORECASE):
                continue
            if self._all_source_by_index and not self._context_mapping_has_document_support(
                heard,
                canonical,
                source,
            ):
                continue
            normalized = re.sub(
                rf"(?<!\w){re.escape(heard)}(?!\w)",
                canonical,
                source,
                count=1,
                flags=re.IGNORECASE,
            )
            return {
                "kind": "context_confirmed_asr_variant",
                "heard": heard,
                "canonical": canonical,
                "instruction": (
                    f"Global terminology independently confirmed '{heard}' as an ASR "
                    f"variant of '{canonical}'. Use the canonical form."
                ),
                "normalized_source": normalized,
            }
        return {}

    def _source_for_translation(self, source: str) -> str:
        """Use a document-confirmed correction without mutating the source subtitle."""
        variant = self._context_asr_variant(source)
        normalized = str(variant.get("normalized_source") or "").strip()
        return normalized or source

    def _confirmed_context_canonical(self, source: str) -> str:
        """Return an exact document-confirmed name required by a corrected source."""
        variant = self._context_asr_variant(source)
        if variant.get("kind") != "context_confirmed_asr_variant":
            return ""
        canonical = str(variant.get("canonical") or "").strip()
        if not canonical or not re.search(r"[A-Z]", canonical):
            return ""
        return canonical

    def _confirmed_context_epithet_target(self, source: str) -> str:
        """Return the reviewed target rendering for an explicit contextual epithet."""
        for raw_line in self.translation_context.terminology.splitlines():
            line = str(raw_line or "").lstrip("- ").strip()
            if " -> " not in line or not re.search(
                r"(?:epithet|nickname|non-?literal|别称|別稱|绰号|綽號|非字面)",
                line,
                flags=re.IGNORECASE,
            ):
                continue
            heard, target_with_note = line.split(" -> ", 1)
            heard = heard.strip()
            target = re.sub(r"\s+\(.*$", "", target_with_note).strip()
            if (
                heard
                and target
                and re.search(rf"(?<!\w){re.escape(heard)}(?!\w)", source, re.IGNORECASE)
            ):
                return target
        return ""

    def _confirmed_context_official_identifier_target(self, source: str) -> str:
        """Return a manufacturer label deterministically confirmed in document context."""
        for raw_line in self.translation_context.terminology.splitlines():
            line = str(raw_line or "").lstrip("- ").strip()
            if " -> " not in line or not re.search(
                r"official\s+manufacturer\s+identifier",
                line,
                flags=re.IGNORECASE,
            ):
                continue
            heard, target_with_note = line.split(" -> ", 1)
            heard = heard.strip()
            target = re.sub(r"\s+\(.*$", "", target_with_note).strip()
            if (
                heard
                and target
                and re.search(
                    rf"(?<!\w){re.escape(heard)}(?!\w)",
                    source,
                    flags=re.IGNORECASE,
                )
            ):
                return target
        return ""

    def _confirmed_context_acronym_target(self, source: str) -> tuple[str, str] | None:
        """Return a CJK expansion for an acronym explicitly mapped in context."""
        for raw_line in self.translation_context.terminology.splitlines():
            line = str(raw_line or "").lstrip("- ").strip()
            if " -> " not in line:
                continue
            acronym, target_with_note = line.split(" -> ", 1)
            acronym = acronym.strip()
            target = re.sub(r"\s+\(.*$", "", target_with_note).strip()
            if (
                re.fullmatch(r"[A-Z][A-Z0-9&-]{1,9}", acronym)
                and re.search(r"[㐀-鿿]", target)
                and re.search(rf"(?<!\w){re.escape(acronym)}(?!\w)", source)
            ):
                return acronym, target
        return None

    @staticmethod
    def _parse_context_asr_mapping(raw_line: str) -> tuple[str, str] | None:
        """Read one ASR terminology line without treating its note as a name.

        Context providers sometimes return a translated target and put the actual
        corrected spelling in a nested note, for example ``Infinity -> 英菲尼迪
        (Probable ASR correction: 'Infinity' should be 'Infiniti' (brand name).)``.
        The explicit correction in that note is authoritative for source
        normalization; the translated target is only a display translation.
        """
        line = str(raw_line or "").lstrip("- ").strip()
        if " -> " not in line or not re.search(
            r"(?:asr|phonetic|mishear|recognition|spoken\s+self-correction|"
            r"self-correction|转录|听写|同音|口误|自我修正)",
            line,
            flags=re.IGNORECASE,
        ):
            return None
        heard, target_with_note = line.split(" -> ", 1)
        heard = heard.strip()
        if is_unsafe_global_term_source(heard):
            return None
        explicit = re.search(
            r"(?:for|intended(?:\s+as)?|should\s+be|"
            r"correct(?:ed)?(?:\s+as|\s+to)?)\s+['\"]([^'\"]{2,80})['\"]",
            target_with_note,
            flags=re.IGNORECASE,
        )
        note_canonical = re.search(
            r"(?:(?i:canonical\s+(?:form|name|spelling))(?:\s+(?i:is)|\s*:)|"
            r"(?i:variant\s+of|phonetic\s+candidate\s+for))\s+"
            r"['\"]?([A-Z][A-Za-z'’.-]*(?:\s+[A-Z][A-Za-z'’.-]*){0,5})",
            target_with_note,
        )
        canonical = (
            explicit.group(1).strip()
            if explicit
            else (
                note_canonical.group(1).strip(" ,.;:!?-'’\"")
                if note_canonical
                else re.sub(r"\s+\(.*$", "", target_with_note).strip()
            )
        )
        if (
            not (explicit or note_canonical)
            and re.search(r"[A-Za-z]", heard)
            and re.search(r"[一-鿿぀-ヿ가-힯]", canonical)
        ):
            return None
        if not heard or not canonical or heard.casefold() == canonical.casefold():
            return None
        return heard, canonical

    def _context_variant_has_document_support(self, canonical: str, source: str) -> bool:
        """Require recurring evidence before trusting an LLM's proper-name correction."""
        raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", canonical)
        entity_like = any(
            re.fullmatch(r"[A-Z]{2,}[A-Z0-9-]*", token)
            or (re.search(r"[A-Za-z]", token) and re.search(r"\d", token))
            for token in raw_tokens
        ) or any(token[:1].isupper() for token in raw_tokens[1:])
        if len(raw_tokens) == 1 and raw_tokens[0][:1].isupper():
            entity_like = True
        if not entity_like:
            return True
        mapped_heard_forms: set[str] = set()
        for raw_line in self.translation_context.terminology.splitlines():
            mapping = self._parse_context_asr_mapping(raw_line)
            if not mapping:
                continue
            heard, mapped_canonical = mapping
            if mapped_canonical.casefold() == canonical.casefold():
                mapped_heard_forms.add(heard.casefold())
        if len(mapped_heard_forms) >= 2:
            return True
        evidence = "\n".join(
            [
                self.custom_prompt,
                *(
                    text
                    for text in self._all_source_by_index.values()
                    if text.strip() != source.strip()
                ),
            ]
        ).casefold()
        cleaned = re.sub(
            r"\b(?:or\s+something|last\s+year|this\s+year|next\s+year)\b.*$",
            "",
            canonical,
            flags=re.IGNORECASE,
        ).strip()
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", cleaned)
        for width in range(len(tokens), 1, -1):
            for start in range(0, len(tokens) - width + 1):
                phrase = " ".join(tokens[start : start + width]).casefold()
                if phrase in evidence:
                    return True
        return len(tokens) <= 1 and cleaned.casefold() in evidence

    def _context_mapping_has_document_support(
        self,
        heard: str,
        canonical: str,
        source: str,
    ) -> bool:
        return self._context_variant_has_document_support(canonical, source)

    def _document_model_asr_variant(self, source: str) -> Dict[str, str]:
        """Reconcile a phonetic model mention with a repeated document subject."""
        model_pattern = re.compile(
            r"\b(?:[A-Z]{2,}\s+[A-Z][A-Za-z]{3,}|"
            r"[A-Z][A-Za-z]{3,}\s+[A-Z0-9]{1,4})\b"
        )
        counts: dict[str, tuple[str, int]] = {}
        for text in [self.custom_prompt, *self._all_source_by_index.values()]:
            for match in model_pattern.finditer(str(text or "")):
                value = match.group()
                key = value.casefold()
                previous = counts.get(key, (value, 0))
                counts[key] = (previous[0], previous[1] + 1)

        # In an ASR correction line, the left side is explicitly the malformed
        # heard form. Only the right side is eligible as canonical evidence.
        context_models: set[str] = set()
        for raw_line in self.translation_context.terminology.splitlines():
            mapping = self._parse_context_asr_mapping(raw_line)
            if mapping:
                _heard, evidence = mapping
            else:
                evidence = re.sub(
                    r"\s+\(.*$",
                    "",
                    str(raw_line or "").lstrip("- ").strip(),
                ).strip()
            for match in model_pattern.finditer(evidence):
                value = match.group()
                key = value.casefold()
                context_models.add(key)
                previous = counts.get(key, (value, 0))
                counts[key] = (previous[0], previous[1])
        canonical_models = [
            value
            for key, (value, count) in counts.items()
            if count >= 2
            or value.casefold() in self.custom_prompt.casefold()
            or (key in context_models and count >= 1)
        ]
        if not canonical_models:
            return {}

        suspicious_patterns = (
            r"\byour\s+([A-Z][A-Za-z]+\s+[A-Za-z]{3,})\b",
            r"\b(?:this|the|new)\s+(?:\d{2,4}\s+)?([A-Z]{2,}\s+[A-Z][A-Za-z]{3,})\b",
            r"\b([A-Z][A-Za-z]{3,}\s+[A-Z])\b",
        )
        for pattern in suspicious_patterns:
            match = re.search(pattern, source)
            if not match:
                continue
            heard = match.group(1)
            heard_compact = re.sub(r"[^a-z0-9]", "", heard.casefold())
            canonical_compacts = {
                re.sub(r"[^a-z0-9]", "", value.casefold()) for value in canonical_models
            }
            if heard_compact in canonical_compacts:
                return {}
            for canonical in canonical_models:
                canonical_compact = re.sub(r"[^a-z0-9]", "", canonical.casefold())
                if heard_compact == canonical_compact:
                    continue
                similarity = difflib.SequenceMatcher(
                    None,
                    heard_compact,
                    canonical_compact,
                ).ratio()
                same_model_code = heard.split()[0].casefold() == canonical.split()[0].casefold()
                threshold = 0.60 if same_model_code else 0.69
                if similarity < threshold:
                    continue
                return {
                    "kind": "document_repeated_model_variant",
                    "heard": heard,
                    "canonical": canonical,
                    "instruction": (
                        f"The document repeatedly identifies the reviewed model as '{canonical}'. "
                        f"The referential phrase '{heard}' is a close phonetic ASR variant."
                    ),
                    "normalized_source": source.replace(heard, canonical, 1),
                }
        return {}

    def _alignment_asr_hint(
        self,
        source: str,
        previous_source: str,
        next_source: str,
    ) -> Dict[str, str]:
        """Build a machine-verifiable hint only for an explicit local contradiction."""
        if re.search(r"\bGrimina\s+GR\s+Corolla\b", source, re.IGNORECASE):
            normalized = re.sub(
                r"\bGrimina\s+GR\s+Corolla\b",
                "GRMN GR Corolla",
                source,
                count=1,
                flags=re.IGNORECASE,
            )
            return {
                "kind": "known_model_acronym_asr_variant",
                "heard": "Grimina GR Corolla",
                "canonical": "GRMN GR Corolla",
                "instruction": (
                    "The Toyota track-focused model is GRMN GR Corolla; 'Grimina' is the "
                    "spoken acronym rendered phonetically by ASR."
                ),
                "normalized_source": normalized,
            }
        known_automotive_variants = (
            (
                r"\bLexus\s+LMXX\s+Grimina\b",
                "Lexus LBX Morizo RR",
                "The same-engine Lexus model is LBX Morizo RR; the heard phrase is a "
                "multi-token phonetic ASR error.",
            ),
            (
                r"\bMarizzo(?=-style\b)",
                "Morizo",
                "The Toyota performance trim/style name is Morizo; 'Marizzo' is a "
                "phonetic ASR variant.",
            ),
        )
        for pattern, replacement, instruction in known_automotive_variants:
            if re.search(pattern, source, re.IGNORECASE):
                return {
                    "kind": "known_automotive_name_asr_variant",
                    "canonical": replacement,
                    "instruction": instruction,
                    "normalized_source": re.sub(
                        pattern,
                        replacement,
                        source,
                        count=1,
                        flags=re.IGNORECASE,
                    ),
                }
        document_variant = self._document_model_asr_variant(source)
        if document_variant:
            return document_variant
        shortened_place = detect_document_shortened_place(
            source,
            self._all_source_by_index.values(),
        )
        if shortened_place:
            canonical, normalized = shortened_place
            return {
                "kind": "document_repeated_place_variant",
                "canonical": canonical,
                "instruction": (
                    "The document explicitly names this numbered road elsewhere. The shortened "
                    "plural form in a return-to navigation phrase refers to that same unique "
                    "place, not a remaining distance."
                ),
                "normalized_source": normalized,
            }
        if re.search(r"\bElantra\s+M\b", source, re.IGNORECASE) and any(
            re.search(r"\bElantra\s+N\b", value, re.IGNORECASE)
            for value in self._all_source_by_index.values()
        ):
            normalized = re.sub(
                r"\bElantra\s+M\b",
                "Elantra N",
                source,
                count=1,
                flags=re.IGNORECASE,
            )
            normalized = re.sub(
                r"\bthe\s+Bose\s+and\s+the\s+Elantra\s+N\b",
                "the Bose system in the Elantra N",
                normalized,
                count=1,
                flags=re.IGNORECASE,
            )
            return {
                "kind": "document_repeated_model_variant",
                "heard": "Elantra M",
                "canonical": "Elantra N",
                "instruction": (
                    "The same automotive document names the performance model Elantra N; "
                    "the isolated Elantra M is a one-letter phonetic ASR variant. In this "
                    "audio-system comparison, 'the Bose and the Elantra M' means the Bose "
                    "system in the Elantra N."
                ),
                "normalized_source": normalized,
            }
        context_variant = self._context_asr_variant(source)
        if context_variant:
            return context_variant

        construction_products = re.search(
            r"\b(?P<count>\d{1,3})\s+"
            r"(?P<modifier>(?:[a-z][a-z-]*\s+){0,3})"
            r"construction\s+products\b",
            source,
            flags=re.IGNORECASE,
        )
        if (
            construction_products
            and re.search(
                r"\b(?:called|named|titled)\b",
                previous_source,
                flags=re.IGNORECASE,
            )
            and re.search(
                rf"\b{re.escape(construction_products.group('count'))}\s+projects\b",
                next_source,
                flags=re.IGNORECASE,
            )
        ):
            return {
                "kind": "locally_confirmed_common_noun_variant",
                "instruction": (
                    "The previous cue introduces a title and the next cue independently "
                    "repeats the same count as projects. In this title, ASR rendered "
                    "'construction projects' as 'construction products'."
                ),
                "normalized_source": re.sub(
                    r"\bconstruction\s+products\b",
                    "construction projects",
                    source,
                    count=1,
                    flags=re.IGNORECASE,
                ),
            }

        lexical_repairs = (
            (
                r"\bhot\s+hat\b",
                "hot hatch",
                "A performance-car description requires 'hot hatch', not headwear.",
            ),
            (
                r"\b(?:bump|switch|put)\b([^.!?]{0,40})\binto\s+support\b",
                None,
                "A vehicle drive-mode control selects Sport mode; 'support' is an ASR homophone.",
            ),
            (
                r"\bwant\s+you\s+to\s+love\s+here\b",
                "want you to look here",
                "The deictic instruction asks the viewer to look here; 'love' is an ASR homophone.",
            ),
            (
                r"\bdealer\s+accessory\s+matte\b",
                "dealer accessory mat",
                "The opened cargo-area accessory is a mat; 'matte' is an ASR homophone.",
            ),
            (
                r"\bstick\s+the\s+car\s+in\s+the\s+car\b",
                "stick the key in the car",
                "The physical action is inserting the key into the car; repeating 'car' as "
                "both the inserted object and container is an impossible ASR rendering.",
            ),
            (
                r"\bnot\s+an\s+up\s+badge\b",
                "not an M badge",
                "The comparison is with a recognizable BMW M badge; 'up badge' is a "
                "letter-name ASR error in this automotive context.",
            ),
            (
                r"\bdine[ -]in\s+bits\b",
                "Dinan bits",
                "Dealer-installed performance parts and their D identifiers refer to Dinan; "
                "the restaurant phrase 'dine-in' is a phonetic ASR error.",
            ),
        )
        for pattern, replacement, instruction in lexical_repairs:
            match = re.search(pattern, source, flags=re.IGNORECASE)
            if not match:
                continue
            if "into\\s+support" in pattern:
                normalized = re.sub(
                    r"\binto\s+support\b",
                    "into Sport mode",
                    source,
                    count=1,
                    flags=re.IGNORECASE,
                )
            else:
                assert replacement is not None
                normalized = re.sub(
                    pattern,
                    replacement,
                    source,
                    count=1,
                    flags=re.IGNORECASE,
                )
            return {
                "kind": "local_semantic_homophone",
                "instruction": instruction,
                "normalized_source": normalized,
            }

        if re.search(
            r"\bget\s+a\s+move\s+on\s+and\s+exit\s+the\s+city\s+into\s+reverse\b",
            source,
            flags=re.IGNORECASE,
        ):
            return {
                "kind": "reverse_control_punctuation",
                "instruction": (
                    "ASR lost a sentence boundary. The speaker will leave the city, then "
                    "selects reverse gear for the immediate maneuver."
                ),
                "normalized_source": re.sub(
                    r"\bexit\s+the\s+city\s+into\s+reverse\b",
                    "exit the city. First, shift into reverse",
                    source,
                    count=1,
                    flags=re.IGNORECASE,
                ),
            }

        if re.search(r"\bgoop\b", source, re.IGNORECASE) and re.search(
            r"\bbody\s+panels?\b",
            next_source,
            re.IGNORECASE,
        ):
            return {
                "kind": "body_adhesive_colloquialism",
                "instruction": (
                    "The following source places this material within body panels to add "
                    "rigidity, so colloquial 'goop' denotes structural body adhesive."
                ),
                "normalized_source": re.sub(
                    r"\b(?:sort\s+of\s+)?goop\b",
                    "structural body adhesive",
                    source,
                    count=1,
                    flags=re.IGNORECASE,
                ),
            }

        if re.fullmatch(r"\s*not\s+actually[.!?]?\s*", next_source, re.IGNORECASE) and re.search(
            r"\b(?:favorite|great|wonderful|better|best|love)\b",
            source,
            re.IGNORECASE,
        ):
            return {
                "kind": "explicitly_retracted_sarcasm",
                "instruction": (
                    "The next cue explicitly retracts this compliment, confirming ironic "
                    "delivery. Preserve the sarcasm without importing next_source wording."
                ),
                "normalized_source": f"[sarcastic] {source}",
            }

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
    ) -> str | None:
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
        elif hint.get("kind") in {
            "context_confirmed_asr_variant",
            "document_repeated_model_variant",
            "document_repeated_place_variant",
            "known_automotive_name_asr_variant",
        }:
            canonical = str(hint.get("canonical") or "").strip()
            if hint.get("kind") == "context_confirmed_asr_variant":
                canonical_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", canonical)
                entity_like = any(
                    token[:1].isupper()
                    or re.search(r"\d", token)
                    or (token.isupper() and len(token) >= 2)
                    for token in canonical_tokens
                )
                if not entity_like:
                    return None
            translated_compact = re.sub(r"[^a-z0-9]", "", translated.casefold())
            canonical = re.sub(
                r"\s+(?:last|this|next)\s+(?:model\s+)?year\b.*$",
                "",
                canonical,
                flags=re.IGNORECASE,
            ).strip()
            missing_tokens: list[str] = []
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", canonical):
                token_compact = re.sub(r"[^a-z0-9]", "", token.casefold())
                if token_compact in {"a", "an", "the"}:
                    continue
                if token_compact and token_compact in translated_compact:
                    continue
                if any(
                    alias in translated for alias in _CHINESE_ENTITY_ALIASES.get(token_compact, ())
                ):
                    continue
                missing_tokens.append(token)
            if missing_tokens:
                return f"The confirmed canonical model/name '{canonical}' must be preserved."
        elif hint.get("kind") == "local_semantic_homophone":
            normalized = str(hint.get("normalized_source") or "").casefold()
            if "sport mode" in normalized and not re.search(
                r"(?:运动|sport)模式", translated, re.I
            ):
                return "The confirmed vehicle control selects Sport mode, not support."
            if "look here" in normalized and not re.search(
                r"(?:看|注意)(?:看)?这|看看", translated
            ):
                return "The confirmed deictic instruction means 'look here', not love or like."
            if "hot hatch" in normalized and not re.search(r"(?:小钢炮|性能掀背)", translated):
                return "The confirmed vehicle category is hot hatch."
            if "accessory mat" in normalized and (
                not re.search(r"(?:脚垫|地垫|垫)", translated) or "哑光" in translated
            ):
                return "The confirmed cargo accessory is a mat, not a matte finish."
            if "stick the key in the car" in normalized and "钥匙" not in translated:
                return "The confirmed inserted object is the car key, not the car itself."
            if "not an m badge" in normalized and not re.search(
                r"(?:M|m).{0,4}(?:徽章|标识|标志)", translated
            ):
                return "The confirmed comparison is with an M badge."
            if "dinan bits" in normalized and (
                not re.search(r"(?:Dinan|迪南)", translated, re.IGNORECASE) or "堂食" in translated
            ):
                return "The confirmed dealer-installed parts are Dinan parts, not dine-in items."
        elif hint.get("kind") == "locally_confirmed_common_noun_variant":
            if "产品" in translated or not re.search(r"(?:工程|项目)", translated):
                return "The context-confirmed title names construction projects, not products."
        elif hint.get("kind") == "reverse_control_punctuation":
            if not re.search(r"(?:挂|切换|切)入?倒挡|倒挡", translated) or re.search(
                r"倒车(?:.{0,6})(?:离开|驶出)(?:市区|城市)|"
                r"倒车出城|倒着离开|挂入?倒挡.{0,6}离开市区",
                translated,
            ):
                return "The confirmed immediate action is selecting reverse gear, not reversing out of the city."
        elif hint.get("kind") == "body_adhesive_colloquialism":
            if not re.search(r"(?:车身|结构).{0,4}(?:胶|粘合剂)|车身粘合剂", translated):
                return "The confirmed material is structural body adhesive, not generic goop."
        return ""

    def _semantic_action_repair_hint(
        self,
        source: str,
        translated: str,
        *,
        previous_source: str = "",
        next_source: str = "",
    ) -> str:
        signal = detect_semantic_action_mismatch(
            source,
            translated,
            previous_source=previous_source,
            next_source=next_source,
        )
        return signal.hint if signal else ""

    def _selective_semantic_repair_hint(
        self,
        source: str,
        translated: str,
        *,
        previous_source: str = "",
        next_source: str = "",
    ) -> str:
        """Combine independent high-confidence hints for one sparse repair request."""
        return self._semantic_action_repair_hint(
            source,
            translated,
            previous_source=previous_source,
            next_source=next_source,
        ) or self._chinese_prose_repair_hint(source, translated)

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
            cache_namespace=self.cache_namespace,
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
            self._record_shadow_legacy_action(
                strategy=RepairStrategy.NONE,
                reasoning_mode=PlannedReasoningMode.DISABLED,
            )
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
        # Review mode is implemented by selective alignment/fluency audits below.
        # The broad first pass should return only final text for every provider;
        # exposing a draft and reflection for every cue adds output cost without
        # making the later independent review more reliable.
        compact_reflect_output = self.is_reflect
        if compact_reflect_output:
            compact_output = """<output_format>
Perform the draft and audit internally; do not expose or duplicate them.
Return exactly one JSON object with all and only the current_subtitles keys:
{
  "1": {"native_translation": "Final translation owned only by key 1"}
}
</output_format>"""
            compacted_prompt = re.sub(
                r"<output_format>.*?</output_format>",
                compact_output,
                system_prompt,
                count=1,
                flags=re.DOTALL,
            )
            system_prompt = (
                compacted_prompt
                if compacted_prompt != system_prompt
                else f"{system_prompt}\n\n{compact_output}"
            )
        system_prompt += self._dialogue_prompt_rules(subtitle_dict)
        style_rules = self._target_language_style_rules(subtitle_dict.values())
        if not is_glm_53_model(self.model):
            system_prompt += style_rules
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
        if is_glm_53_model(self.model):
            # GLM caches the stable task-wide prefix. Append batch-specific
            # source hints only after the shared global context.
            system_prompt += style_rules

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
                cache_namespace=self.cache_namespace,
                client=self.llm_client,
                # The main batch is broad generation, not a confirmed defect.
                # Reserve native thinking for sparse semantic/fluency rewrites.
                reasoning_mode="disabled",
                max_output_tokens=4096,
            )
            try:
                content = get_response_text(response)
                response_dict = parse_json_object(content)
                self._normalize_chinese_response_connectives(response_dict)
            except ValueError as exc:
                logger.warning(
                    "LLM returned an invalid final answer, step %s/%s: %s",
                    step + 1,
                    self.MAX_STEPS,
                    exc,
                )
                if is_kimi_k3_model(self.model):
                    messages.append(self._assistant_followup_message(response, content or ""))
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
                self._record_shadow_legacy_action(
                    strategy=(
                        RepairStrategy.RETRY if step + 1 < self.MAX_STEPS else RepairStrategy.NONE
                    ),
                    reasoning_mode=PlannedReasoningMode.DISABLED,
                )
                messages.append(
                    self._assistant_followup_message(
                        response,
                        json.dumps(response_dict, ensure_ascii=False),
                    )
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
            self._record_shadow_legacy_action(
                strategy=RepairStrategy.NONE,
                reasoning_mode=PlannedReasoningMode.DISABLED,
            )
            logger.warning(
                f"LLM translation failed validation after {self.MAX_STEPS} retries: {error_msg}"
            )
            raise RuntimeError(f"LLM translation failed validation: {error_msg}")
        return cast(Dict[str, Any], last_response_dict)

    @staticmethod
    def _normalize_stacked_chinese_connectives(text: str) -> str:
        """Resolve only unambiguous sentence-initial connector collisions.

        Spoken English often stacks fillers such as ``and so, but yeah``.
        Chinese models occasionally mirror that as ``所以但`` and then repeat
        the same invalid form after validator feedback.  This small rewrite
        preserves the dominant relation while avoiding an otherwise fatal
        single-item recovery loop.  It deliberately does not rewrite normal
        connectives elsewhere in the subtitle.
        """
        result = str(text or "").strip()
        rewrites = (
            (r"^(?:所以|因此)\s*(?:但|但是|不过)", "不过"),
            (r"^(?:而且|并且)\s*(?:但|但是|不过)", "不过"),
            (r"^(?:但|但是|不过)\s*(?:所以|因此)", "所以"),
            (r"^(?:但|但是|不过)\s*(?:是的|对的|没错)", "是的 不过"),
            (r"^(?:所以|因此)\s*(?:是的|对的|没错)", "是的 所以"),
        )
        for pattern, replacement in rewrites:
            updated = re.sub(pattern, replacement, result, count=1)
            if updated != result:
                return updated.strip()
        result = re.sub(
            r"^(?:所以\s*)?我(?:并)?不(?:[—–-]+|[，,\s]+)+"
            r"我(?=(?:在|想|觉得|认为|写|说))",
            "我",
            result,
            count=1,
        )
        return result

    def _normalize_chinese_response_connectives(
        self,
        response_dict: Dict[str, Any],
    ) -> None:
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return

        def normalize(key: str, translated: str) -> str:
            result = self._normalize_stacked_chinese_connectives(translated)
            source = self._all_source_by_index.get(int(key), "") if str(key).isdigit() else ""
            mapping = self._confirmed_context_acronym_target(source)
            if not mapping:
                return result
            acronym, target = mapping
            result = re.sub(
                rf"{re.escape(acronym)}\s*[（(]?\s*{re.escape(target)}\s*[）)]?",
                target,
                result,
                flags=re.IGNORECASE,
            )
            return re.sub(
                rf"{re.escape(target)}\s*[（(]?\s*{re.escape(acronym)}\s*[）)]?",
                target,
                result,
                flags=re.IGNORECASE,
            )

        for key, value in list(response_dict.items()):
            if isinstance(value, str):
                response_dict[key] = normalize(key, value)
                continue
            if not isinstance(value, dict):
                continue
            for field in ("native_translation", "initial_translation", "translation"):
                translated = value.get(field)
                if isinstance(translated, str):
                    value[field] = normalize(key, translated)

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
            language = self._all_language_by_index.get(index)
            if language:
                item["source_language"] = language
            result.append(item)
        return result

    def _current_subtitles_payload(self, subtitle_dict: Dict[str, str]) -> Dict[str, Any]:
        """Attach anonymous dialogue turns without mixing labels into source text."""
        has_speakers = any(
            self._all_speaker_by_index.get(int(key)) for key in subtitle_dict if str(key).isdigit()
        )
        has_languages = any(
            self._all_language_by_index.get(int(key)) for key in subtitle_dict if str(key).isdigit()
        )
        if not has_speakers and not has_languages:
            return dict(subtitle_dict)
        payload: Dict[str, Any] = {}
        for key, source in subtitle_dict.items():
            item = {"source": source}
            if has_speakers:
                item["speaker"] = self._all_speaker_by_index.get(int(key), "")
            if has_languages:
                item["source_language"] = self._all_language_by_index.get(int(key), "")
            payload[key] = item
        return payload

    def _dialogue_prompt_rules(self, subtitle_dict: Dict[str, str]) -> str:
        has_speakers = any(
            self._all_speaker_by_index.get(int(key)) for key in subtitle_dict if str(key).isdigit()
        )
        has_languages = any(
            self._all_language_by_index.get(int(key)) for key in subtitle_dict if str(key).isdigit()
        )
        return translation_metadata_guidance(
            include_speakers=has_speakers,
            include_languages=has_languages,
        )

    def _target_language_style_rules(self, source_texts=()) -> str:
        return target_language_style_rules(
            self.target_language.value,
            source_texts,
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
        self._shadow_repair_context.last_plan = None
        expected_keys = set(subtitle_dict.keys())
        schema_diagnostics = inspect_response_schema(response_dict, expected_keys)
        if schema_diagnostics:
            self._record_shadow_repair_plan(schema_diagnostics)
            return False, legacy_diagnostic_message(schema_diagnostics)
        response_dict = cast(Dict[str, Any], response_dict)
        actual_keys = set(response_dict.keys())

        def sort_keys(keys):
            return sorted(keys, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))

        # Helper: extract translated text from a response value
        def _extract_text(val):
            if isinstance(val, dict):
                return val.get("native_translation", val.get("initial_translation", ""))
            return str(val)

        # Check if translated text is actually in the target language
        _cjk_langs = {"简体中文", "繁体中文", "日本語", "韩语", "粤语"}
        _is_cjk_target = self.target_language.value in _cjk_langs
        if _is_cjk_target:
            source_languages = {
                key: self._all_language_by_index.get(int(key), "")
                for key in actual_keys
                if str(key).isdigit()
            }
            untranslated_diagnostic = inspect_target_script(
                response_dict,
                subtitle_dict,
                target_language=self.target_language,
                source_language_by_key=source_languages,
            )
            if untranslated_diagnostic is not None:
                self._record_shadow_repair_plan((untranslated_diagnostic,))
                return False, untranslated_diagnostic.message

        reasoning_leak_diagnostic = inspect_reasoning_leaks(
            response_dict,
            subtitle_dict,
        )
        if reasoning_leak_diagnostic is not None:
            self._record_shadow_repair_plan((reasoning_leak_diagnostic,))
            return False, reasoning_leak_diagnostic.message

        script_ok, script_error = self._validate_target_chinese_script(
            response_dict,
            subtitle_dict,
            _extract_text,
        )
        if not script_ok:
            return False, script_error

        missing_epithets = []
        for key in sort_keys(actual_keys):
            source = subtitle_dict.get(key, "")
            target = self._confirmed_context_epithet_target(source)
            translated = _extract_text(response_dict[key])
            if target and target not in translated:
                missing_epithets.append(f"{key}:{target}")
        if missing_epithets:
            return (
                False,
                "Use each reviewed cultural or geographic epithet target exactly in the key "
                "that contains its source phrase. Do not replace it with a literal calque. "
                f"Missing contextual epithet targets: {missing_epithets[:20]}",
            )

        missing_identifiers = []
        for key in sort_keys(actual_keys):
            source = subtitle_dict.get(key, "")
            target = self._confirmed_context_official_identifier_target(source)
            translated = _extract_text(response_dict[key])
            if target and target not in translated:
                missing_identifiers.append(f"{key}:{target}")
        if missing_identifiers:
            return (
                False,
                "Preserve each manufacturer-introduced official identifier exactly as supplied "
                "by global terminology. Translate only its generic head noun, such as seat, "
                f"system, package, or mode. Missing identifiers: {missing_identifiers[:20]}",
            )

        editorial_ok, editorial_error = self._validate_no_added_editorial_labels(
            response_dict,
            subtitle_dict,
            _extract_text,
        )
        if not editorial_ok:
            return False, editorial_error

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

        key_label_ok, key_label_error = self._validate_no_key_label_leakage(
            response_dict,
            subtitle_dict,
            _extract_text,
        )
        if not key_label_ok:
            return False, key_label_error

        price_band_ok, price_band_error = self._validate_natural_price_bands(
            response_dict,
            subtitle_dict,
            _extract_text,
        )
        if not price_band_ok:
            return False, price_band_error

        natural_chinese_ok, natural_chinese_error = (
            self._validate_natural_chinese_degree_constructions(
                response_dict,
                subtitle_dict,
                _extract_text,
            )
        )
        if not natural_chinese_ok:
            return False, natural_chinese_error

        technical_terms_ok, technical_terms_error = self._validate_no_weakened_technical_compounds(
            response_dict,
            subtitle_dict,
            _extract_text,
        )
        if not technical_terms_ok:
            return False, technical_terms_error

        contextual_chinese_ok, contextual_chinese_error = (
            self._validate_natural_chinese_contextual_constructions(
                response_dict,
                subtitle_dict,
                _extract_text,
            )
        )
        if not contextual_chinese_ok:
            return False, contextual_chinese_error

        idiom_ok, idiom_error = self._validate_contextual_idioms(
            response_dict,
            subtitle_dict,
            _extract_text,
        )
        if not idiom_ok:
            return False, idiom_error

        medium_ok, medium_error = self._validate_contextual_demonstration_medium(
            response_dict,
            subtitle_dict,
            _extract_text,
        )
        if not medium_ok:
            return False, medium_error

        automotive_ok, automotive_error = self._validate_contextual_automotive_terms(
            response_dict,
            subtitle_dict,
            _extract_text,
        )
        if not automotive_ok:
            return False, automotive_error

        automotive_numbers_ok, automotive_numbers_error = (
            self._validate_elliptical_automotive_numbers(
                response_dict,
                subtitle_dict,
                _extract_text,
            )
        )
        if not automotive_numbers_ok:
            return False, automotive_numbers_error

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

        latin_names_ok, latin_names_error = self._validate_no_unowned_latin_names(
            response_dict,
            subtitle_dict,
            _extract_text,
        )
        if not latin_names_ok:
            return False, latin_names_error

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
            reflect_diagnostic = inspect_reflect_value_schema(response_dict)
            if reflect_diagnostic is not None:
                self._record_shadow_repair_plan((reflect_diagnostic,))
                return False, reflect_diagnostic.message

        return True, ""

    def _validate_target_chinese_script(
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        """Keep an explicitly Simplified-Chinese task in the requested script."""
        if self.target_language != TargetLanguage.SIMPLIFIED_CHINESE:
            return True, ""

        # These forms are unambiguous Traditional-script variants of common subtitle
        # vocabulary. A compact rejection list avoids a runtime conversion dependency
        # and lets the model rewrite the complete sentence instead of altering names.
        traditional_only = re.compile(
            r"[為這開來個裡麼說應會點無體總聲轉檔覺給過從們實經變讓較並進發現後時別將區見萬僅與還車種臺輛學]"
        )
        leaked = [
            str(key)
            for key in subtitle_dict
            if traditional_only.search(str(extract_text(response_dict.get(key, "")) or ""))
        ]
        if not leaked:
            return True, ""
        return (
            False,
            "The target is Simplified Chinese, but Traditional Chinese characters leaked into "
            "the final subtitle text. Rewrite the complete affected values in Simplified "
            f"Chinese while preserving names, numbers, and meaning. Keys: {leaked[:20]}",
        )

    def _validate_contextual_demonstration_medium(
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        """Reject a vague visual-show calque when local source proves an audio demo."""
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return True, ""

        invalid: list[str] = []
        generic_demo = re.compile(
            r"\bshow\s+(?:you|everyone|them)\s+what\s+(?:this|that|it)(?:'s|\s+is)\s+all\s+about\b",
            flags=re.IGNORECASE,
        )
        audio_context = re.compile(
            r"\b(?:audio|listen|music|sound(?:\s+system|\s+test)?|speaker|stereo|volume)\b",
            flags=re.IGNORECASE,
        )
        concrete_audio_target = re.compile(
            r"(?:听|聽|声音|聲音|音响|音響|音质|音質|声效|聲效|效果|表现|表現)"
        )
        for key, source in subtitle_dict.items():
            if not generic_demo.search(str(source or "")) or not str(key).isdigit():
                continue
            index = int(key)
            local_source = " ".join(
                self._all_source_by_index.get(position, "")
                for position in range(max(1, index - 4), index + 1)
            )
            if not audio_context.search(local_source):
                continue
            translated = str(extract_text(response_dict.get(key, "")) or "")
            if not concrete_audio_target.search(translated):
                invalid.append(str(key))
        if not invalid:
            return True, ""
        return (
            False,
            "Local source explicitly establishes an audio demonstration. Render the audience "
            "as hearing/listening to the sound or its performance; do not use a vague visual "
            f"'show what this is about' calque. Keys: {invalid[:20]}",
        )

    def _validate_contextual_automotive_terms(
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        """Require established Chinese for unambiguous vehicle controls and operating noise."""
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return True, ""

        quiet_component: list[str] = []
        one_touch_window: list[str] = []
        for key, source in subtitle_dict.items():
            source_text = str(source or "")
            translated = str(extract_text(response_dict.get(key, "")) or "")
            if (
                re.search(r"\bquiet\s+(?:air\s+)?vents?\b", source_text, re.IGNORECASE)
                and re.search(r"出风口.{0,8}(?:安静|安靜)", translated)
                and not re.search(r"(?:风噪|風噪|噪音|声音|聲音)", translated)
            ):
                quiet_component.append(str(key))
            if re.search(
                r"\bauto[- ]down\s+windows?\b", source_text, re.IGNORECASE
            ) and not re.search(r"一[键鍵]", translated):
                one_touch_window.append(str(key))

        problems: list[str] = []
        if quiet_component:
            problems.append(
                "A quiet vent describes low operating or wind noise, not an anthropomorphically "
                f"quiet component. State the noise property explicitly. Keys: {quiet_component[:20]}"
            )
        if one_touch_window:
            problems.append(
                "Use the established one-touch-down Chinese term for an auto-down window; do "
                f"not use the literal generic 'automatic lowering' wording. Keys: {one_touch_window[:20]}"
            )
        return (False, " ".join(problems)) if problems else (True, "")

    def _validate_elliptical_automotive_numbers(
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        """Protect unambiguous spoken scales commonly omitted in vehicle reviews."""
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return True, ""

        missing_percent: list[str] = []
        wrong_rpm_range: list[str] = []
        percent_source = re.compile(
            r"\b(\d{1,2})\s+(?:percent\s+)?(?:softer|stiffer|firmer|faster|slower)\b",
            flags=re.IGNORECASE,
        )
        rpm_source = re.compile(
            r"\bbetween\s+([1-9])\s+and\s+([1-9]),?000\s*rpm\b",
            flags=re.IGNORECASE,
        )
        for key, source in subtitle_dict.items():
            source_text = str(source or "")
            translated = str(extract_text(response_dict.get(key, "")) or "")
            percent = percent_source.search(source_text)
            if percent and not re.search(
                rf"(?<!\d){re.escape(percent.group(1))}\s*(?:%|％|百分之)",
                translated,
            ):
                missing_percent.append(str(key))

            rpm_range = rpm_source.search(source_text)
            if rpm_range:
                lower = str(int(rpm_range.group(1)) * 1000)
                upper = str(int(rpm_range.group(2)) * 1000)
                compact = re.sub(r"[,，\s]", "", translated)
                if lower not in compact or upper not in compact:
                    wrong_rpm_range.append(str(key))

        problems: list[str] = []
        if missing_percent:
            problems.append(
                "In an explicit comparative tuning statement, preserve the elliptical "
                "percentage: for example, '20 softer' is '软约20%', not merely '软20'. "
                f"Keys: {missing_percent[:20]}"
            )
        if wrong_rpm_range:
            problems.append(
                "Restore the omitted lower scale in an RPM range: 'between 1 and 2,000 RPM' "
                "means '1000到2000转/分', not '1到2000转/分'. "
                f"Keys: {wrong_rpm_range[:20]}"
            )
        if problems:
            return False, " ".join(problems)
        return True, ""

    def _validate_contextual_idioms(
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        """Reject literal or polarity-shifting readings of explicit common constructions."""
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return True, ""

        literal_archetype: list[str] = []
        overstated_equality: list[str] = []
        literal_traffic: list[str] = []
        for key, source in subtitle_dict.items():
            source_text = str(source or "")
            translated = str(extract_text(response_dict.get(key, "")) or "")
            if re.search(r"\bbiblically\s+accurate\b", source_text, re.IGNORECASE) and re.search(
                r"(?:圣经|聖經|宗教)",
                translated,
            ):
                literal_archetype.append(str(key))
            if (
                re.search(r"\bon\s+par\s+with\b", source_text, re.IGNORECASE)
                and re.search(r"(?:不错|不錯|优秀|優秀|出色|很好)", translated)
                and not re.search(
                    r"(?:同一水平|同等|不相上下|相若|持平|差不多|基本一致)",
                    translated,
                )
            ):
                overstated_equality.append(str(key))
            if re.search(
                r"\btraffic\s+situation\b.{0,100}\bget\s+(?:get\s+)?through\s+it\b",
                source_text,
                re.IGNORECASE,
            ) and re.search(r"(?:通过|通過)(?:车辆|車輛|它|这个|這個)", translated):
                literal_traffic.append(str(key))

        problems: list[str] = []
        if literal_archetype:
            problems.append(
                "'Biblically accurate' is figurative here: express fidelity to the archetype "
                "or category, without introducing the Bible or religion. "
                f"Keys: {literal_archetype[:20]}"
            )
        if overstated_equality:
            problems.append(
                "'On par with' is a neutral equality comparison. Express the same level or "
                "broad comparability without upgrading it to praise. "
                f"Keys: {overstated_equality[:20]}"
            )
        if literal_traffic:
            problems.append(
                "In a traffic situation, render 'get through it' as maneuvering or threading "
                "through congestion, not literally passing a vehicle or demonstrative. "
                f"Keys: {literal_traffic[:20]}"
            )
        return (False, " ".join(problems)) if problems else (True, "")

    def _validate_no_added_editorial_labels(
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        """Reject translator-authored stage directions that leak into subtitles."""
        labels: list[str] = []
        label_pattern = re.compile(
            r"^\s*[\[【（(]\s*(?:讽刺(?:地|语气)?|反讽(?:地|语气)?|"
            r"挖苦(?:地|语气)?|sarcastic(?:ally)?|ironic(?:ally)?)\s*[\]】）)]",
            flags=re.IGNORECASE,
        )
        source_label_pattern = re.compile(
            r"[\[【（(]\s*(?:讽刺|反讽|挖苦|sarcastic|ironic)",
            flags=re.IGNORECASE,
        )
        for key, source in subtitle_dict.items():
            translated = str(extract_text(response_dict.get(key, "")) or "")
            if label_pattern.search(translated) and not source_label_pattern.search(str(source)):
                labels.append(str(key))
        if not labels:
            return True, ""
        return (
            False,
            "Express irony through natural wording. Do not add bracketed stage directions or "
            f"translator commentary. Keys: {labels[:20]}",
        )

    @staticmethod
    def _validate_no_key_label_leakage(
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        """Reject JSON/SRT item labels accidentally copied into translation text."""
        leaked: list[str] = []
        for key, source in subtitle_dict.items():
            translated = str(extract_text(response_dict.get(key, "")) or "")
            pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(str(key))}\s*[:：]\s*"
                r"(?=[A-Za-z\u3400-\u9fff])"
            )
            if pattern.search(translated) and not pattern.search(str(source or "")):
                leaked.append(str(key))
        if not leaked:
            return True, ""
        return (
            False,
            "Subtitle key labels must not appear inside translated text. Remove accidental "
            f"JSON/SRT index prefixes from keys: {leaked[:20]}",
        )

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

    def _validate_natural_chinese_degree_constructions(
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        """Reject a narrow resultative-degree calque while preserving normal `有多` usage."""
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return True, ""

        awkward: list[str] = []
        source_pattern = re.compile(
            r"\b(?:that|this)\s+is\s+how\s+(?P<degree>[a-z]+)\s+"
            r"(?:this|it|that)\s+gets?\s+in\s+(?P=degree)\s+mode\b",
            flags=re.IGNORECASE,
        )
        target_pattern = re.compile(r"这(?:就)?是.{0,24}有多")
        for key, source in subtitle_dict.items():
            translated = str(extract_text(response_dict.get(key, "")) or "")
            if source_pattern.search(str(source or "")) and target_pattern.search(translated):
                awkward.append(str(key))
        if not awkward:
            return True, ""
        return (
            False,
            "Use natural resultative Chinese for demonstrative degree statements. Render "
            "'That is how quiet this gets in quiet mode' as '静音模式下 它的声音就是这么轻', "
            "not the literal calque '这就是安静模式下它有多安静'. "
            f"Unnatural degree keys: {awkward[:20]}",
        )

    def _validate_no_weakened_technical_compounds(
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        """Reject vague Chinese placeholders for explicit technical compounds."""
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return True, ""

        concrete_compound = re.compile(
            r"\b(?:[a-z][a-z-]*\s+){1,3}"
            r"(?:assembly|beam|core|corridor|damper|foundation|podium|sensor|"
            r"shaft|system|wall)\b",
            flags=re.IGNORECASE,
        )
        explicit_vagueness = re.compile(
            r"(?:特定|某种|某个|一般|普通)(?:的)?"
            r"[\u3400-\u9fff]{0,8}"
            r"(?:系统|裝置|装置|組件|组件|結構|结构|牆|墙|阻尼器|核心筒|"
            r"樑|梁|豎井|竖井|感測器|传感器|走廊|基礎|基础|裙樓|裙楼)"
        )
        source_allows_vagueness = re.compile(
            r"\b(?:a\s+kind\s+of|a\s+sort\s+of|certain|common|general|generic|"
            r"ordinary|particular|some\s+kind\s+of|specific)\b",
            flags=re.IGNORECASE,
        )
        weakened: list[str] = []
        for key, source in subtitle_dict.items():
            source_text = str(source or "")
            translated = str(extract_text(response_dict.get(key, "")) or "")
            if (
                concrete_compound.search(source_text)
                and not source_allows_vagueness.search(source_text)
                and explicit_vagueness.search(translated)
            ):
                weakened.append(str(key))

        if not weakened:
            return True, ""
        return (
            False,
            "The source names a concrete technical compound, but the translation weakens its "
            "modifier into an unsupported vague label such as 特定的/某种/一般的. Use the "
            "established professional target-language term and preserve the compound's exact "
            f"technical sense. Keys: {weakened[:20]}",
        )

    def _validate_natural_chinese_contextual_constructions(  # noqa: C901
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        """Reject narrow Chinese calques confirmed by full-run subtitle audits."""
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return True, ""

        use_case_keys: list[str] = []
        numeric_shorthand_keys: list[str] = []
        discourse_now_keys: list[str] = []
        awkward_softness_keys: list[str] = []
        literal_irl_keys: list[str] = []
        literal_yes_answer_keys: list[str] = []
        literal_fresh_slate_keys: list[str] = []
        literal_racing_line_keys: list[str] = []
        literal_price_argument_keys: list[str] = []
        repeated_ride_quality_keys: list[str] = []
        numeric_self_correction_keys: list[str] = []
        incomplete_phone_aside_keys: list[str] = []
        collapsed_reference_keys: list[str] = []
        literal_limo_stop_keys: list[str] = []
        literal_body_adhesive_keys: list[str] = []
        literal_run_cooler_keys: list[str] = []
        ambiguous_downshift_second_keys: list[str] = []
        fuel_economy_how_do_keys: list[str] = []
        literal_bat_out_of_hell_keys: list[str] = []
        parking_direction_keys: list[str] = []
        rev_match_first_keys: list[str] = []
        revised_component_keys: list[str] = []
        document_trim_variant_keys: list[str] = []
        middle_seat_doubt_keys: list[str] = []
        model_year_generation_keys: list[str] = []
        literal_fundamental_keys: list[str] = []
        literal_discourse_filler_keys: list[str] = []
        stacked_connective_keys: list[str] = []
        almost_two_dozen_keys: list[str] = []
        literal_literacy_keys: list[str] = []
        inconsistent_post_literacy_keys: list[str] = []
        malformed_email_keys: list[str] = []
        reversed_negative_valuation_keys: list[str] = []
        use_case_source = re.compile(
            r"\b\d+%\s+of\s+what\s+you['’]re\s+going\s+to\s+use\s+"
            r"this\s+(?:truck|car|vehicle)\s+for\b.*\blike\s+the\b.*"
            r"\bis\s+a\s+good\s+thing\b",
            flags=re.IGNORECASE,
        )
        numeric_shorthand_source = re.compile(
            r"\b(?:came|comes|coming)\s+in\s+with\s+(?:this|the|a)\s+"
            r"\d{2,4}\s*,?$",
            flags=re.IGNORECASE,
        )
        for key, source in subtitle_dict.items():
            translated = str(extract_text(response_dict.get(key, "")) or "").strip()
            source_text = str(source or "").strip()
            compact_target = re.sub(r"[\s，。！？；：、,.!?;:]+", "", translated)
            if use_case_source.search(source_text) and (
                re.search(r"使用.{0,12}\d+%", compact_target)
                or re.search(r"\d+%.{0,24}(?:比如|例如).{0,24}好事", compact_target)
                or re.search(
                    r"(?:这个)?转向(?:机|齿条).{0,8}(?:都)?是(?:件)?好(?:事|东西)",
                    compact_target,
                )
            ):
                use_case_keys.append(str(key))
            if numeric_shorthand_source.search(source_text) and compact_target.endswith("的"):
                numeric_shorthand_keys.append(str(key))
            if re.fullmatch(r"now[.!?]?", source_text, flags=re.IGNORECASE) and compact_target in {
                "现在",
                "如今",
                "目前",
            }:
                discourse_now_keys.append(str(key))
            if re.search(
                r"\b20\s+softer\s+for\s+26\s+from\s+the\s+25\s+model\b",
                source_text,
                flags=re.IGNORECASE,
            ) and (
                "针对26款来说" in compact_target
                or re.search(r"26款.{0,12}25款.{0,8}26款", compact_target)
                or compact_target.count("25款") > 1
                or compact_target.endswith("26款相比25款")
                or compact_target.endswith("26款")
            ):
                awkward_softness_keys.append(str(key))
            if re.fullmatch(r"\s*merging\s+irl[.!?]?\s*", source_text, re.IGNORECASE) and re.search(
                r"(?<![A-Za-z])IRL(?![A-Za-z])",
                translated,
                re.IGNORECASE,
            ):
                literal_irl_keys.append(str(key))
            if re.search(
                r"\b(?:the\s+)?answer(?:\s+here)?\s+is\s+yes\b",
                source_text,
                flags=re.IGNORECASE,
            ) and re.search(r"答案是\s*(?:容易|简单|好开|可以)$", translated):
                literal_yes_answer_keys.append(str(key))
            if re.search(r"\bfresh\s+slate\b", source_text, re.IGNORECASE) and re.search(
                r"(?:全新|新的|新)\s*(?:汽车|车辆)?平台",
                translated,
            ):
                literal_fresh_slate_keys.append(str(key))
            if re.search(r"\bracing\s+line\b", source_text, re.IGNORECASE) and re.search(
                r"接近.{0,6}赛道.{0,6}路线|赛道附近.{0,6}路线",
                translated,
            ):
                literal_racing_line_keys.append(str(key))
            if re.search(
                r"\bargue\s+for\s+(?:your\s+)?[$]\s*\d",
                source_text,
                re.IGNORECASE,
            ) and re.search(r"争", translated):
                literal_price_argument_keys.append(str(key))
            if re.search(
                r"\bstiff\b.*\bbouncy\b.*\bcrashy\b",
                source_text,
                re.IGNORECASE,
            ) and re.search(r"(?:有点)?颠\s*(?:有点)?颠簸", translated):
                repeated_ride_quality_keys.append(str(key))

            numeric_correction = re.search(
                r"\b(\d+(?:\.\d+)?)\s*,\s+(?:a|uh|um|er)\s+(\d+\.\d+)\b",
                source_text,
                re.IGNORECASE,
            )
            if numeric_correction:
                abandoned, corrected = numeric_correction.groups()
                if (
                    re.search(
                        rf"(?<![\d.]){re.escape(abandoned)}(?![\d.])",
                        translated,
                    )
                    and corrected in translated
                ):
                    numeric_self_correction_keys.append(str(key))
            if (
                re.search(
                    r"\bsmaller\s+this\s+is\s+a\s+pro\s+ma[ctx]\b",
                    source_text,
                    re.IGNORECASE,
                )
                and re.search(r"Pro\s*Max", translated, re.IGNORECASE)
                and not re.search(
                    r"手机|電話|电话|手提电话|行動電話",
                    translated,
                )
            ):
                incomplete_phone_aside_keys.append(str(key))
            reference_shift = re.match(
                r"^that\s+is\s+(?:(?:quite|really|pretty|so|very)\s+)?"
                r"(?:good|nice|great|cool|beautiful)[,;]\s*(?:a\s+)?"
                r"(?:fellow\s+)?([A-Za-z][A-Za-z0-9-]*)\b",
                source_text,
                flags=re.IGNORECASE,
            )
            if reference_shift:
                new_subject = reference_shift.group(1)
                if re.search(
                    rf"{re.escape(new_subject)}.{{0,6}}(?:很|真|相当|非常)?"
                    r"(?:不错|很好|漂亮|很棒|真棒|酷)",
                    translated,
                    flags=re.IGNORECASE,
                ):
                    collapsed_reference_keys.append(str(key))
            if re.search(r"\blimo\s+stops?\b", source_text, re.IGNORECASE) and re.search(
                r"礼宾(?:式)?(?:停车|刹车|制动)",
                translated,
            ):
                literal_limo_stop_keys.append(str(key))
            if re.search(
                r"\b(?:feet|foot)\b.*\b(?:goop|adhesive)\b|"
                r"\b(?:goop|adhesive)\b.*\bbody\s+panels?\b",
                source_text,
                re.IGNORECASE,
            ) and re.search(r"胶状物|黏糊糊|粘糊糊|浆糊", translated):
                literal_body_adhesive_keys.append(str(key))
            if re.search(r"\brun\s+(?:even\s+)?cooler\b", source_text, re.IGNORECASE) and re.search(
                r"跑得.{0,4}(?:更)?凉|跑得凉快",
                translated,
            ):
                literal_run_cooler_keys.append(str(key))
            if re.search(
                r"\bdownshift\s+a\s+second\b",
                source_text,
                re.IGNORECASE,
            ) and re.search(r"降\s*一\s*[档挡]", translated):
                ambiguous_downshift_second_keys.append(str(key))
            previous_source = ""
            next_source = ""
            if str(key).isdigit():
                numeric_key = int(key)
                previous_source = self._all_source_by_index.get(numeric_key - 1, "")
                next_source = self._all_source_by_index.get(numeric_key + 1, "")
            if (
                re.search(r"\bsee\s+how\s+we\s+do\b", source_text, re.IGNORECASE)
                and re.search(
                    r"\b(?:mpg|fuel\s+economy|miles\s+per\s+gallon)\b",
                    previous_source,
                    re.IGNORECASE,
                )
                and re.search(r"开起来|驾驶表现|开着怎么样", translated)
            ):
                fuel_economy_how_do_keys.append(str(key))
            if re.search(r"\bbat\s+out\s+of\s+hell\b", source_text, re.IGNORECASE) and re.search(
                r"地狱.{0,4}蝙蝠|蝙蝠.{0,4}地狱",
                translated,
            ):
                literal_bat_out_of_hell_keys.append(str(key))
            if re.search(
                r"\brolling\s+down\s+(?:this|the)\s+parking\s+(?:structure|garage)\b",
                source_text,
                re.IGNORECASE,
            ) and re.search(r"驶出|开出|离开", translated):
                parking_direction_keys.append(str(key))
            if re.search(r"\brev\s+match\s+first\b", source_text, re.IGNORECASE) and re.search(
                r"先.{0,3}(?:挂|切|换)(?:入|上|到)?一[档挡]|先挂一[档挡]",
                translated,
            ):
                rev_match_first_keys.append(str(key))
            if (
                re.search(r"\bnewly\s+revised\s*$", source_text, re.IGNORECASE)
                and re.search(r"\b(?:speaker|JBL|Bose|audio|sound)\b", next_source, re.IGNORECASE)
                and re.search(r"车型|车款|车辆", translated)
            ):
                revised_component_keys.append(str(key))
            if (
                re.search(r"\bElantra\s+M\b", source_text, re.IGNORECASE)
                and sum(
                    bool(re.search(r"\bElantra\s+N\b", value, re.IGNORECASE))
                    for value in self._all_source_by_index.values()
                )
                >= 1
                and re.search(
                    r"(?<![A-Za-z0-9])Elantra\s+M(?=\s|[，。！？；：、,.!?;:]|$|上|的)",
                    translated,
                    re.IGNORECASE,
                )
            ):
                document_trim_variant_keys.append(str(key))
            if re.search(
                r"\bi\s+don['’]t\s+know\s+that\s+you['’]re\s+putting\s+somebody\s+"
                r"in\s+the\s+middle\b",
                source_text,
                re.IGNORECASE,
            ) and re.search(r"不确定.{0,8}(?:让|叫)?谁", translated):
                middle_seat_doubt_keys.append(str(key))
            if re.search(r"\bfourth\s+model\s+year\b", source_text, re.IGNORECASE) and re.search(
                r"第四代|第4代|第四个车型年份|第4个车型年份",
                translated,
            ):
                model_year_generation_keys.append(str(key))
            if re.search(r"\bfundamental\s+to\b", source_text, re.IGNORECASE) and re.search(
                r"对.{0,24}来说(?:太|很|非常)?根本|(?:具有|有)根本(?:性)?意义",
                translated,
            ):
                literal_fundamental_keys.append(str(key))
            if (
                re.search(
                    r"(?:^|[,;]\s*|[.!?]\s+|(?:and|but|so)\s+)"
                    r"you\s+know(?:\s*[,;]|\s+(?!(?:that|what|who|whom|whose|"
                    r"where|when|why|how|whether|if|the|this|that|him|her|them)\b))",
                    source_text,
                    re.IGNORECASE,
                )
                and "你知道" in translated
            ):
                literal_discourse_filler_keys.append(str(key))
            if re.search(
                r"^(?:所以|因此|不过|但是|但|而且|并且)"
                r"(?:但|但是|不过|所以|因此|而且|并且|是的)",
                compact_target,
            ):
                stacked_connective_keys.append(str(key))
            if re.search(r"\balmost\s+two\s+dozen\b", source_text, re.IGNORECASE) and re.search(
                r"(?:将近|接近)?(?:二十|20)多个",
                compact_target,
            ):
                almost_two_dozen_keys.append(str(key))
            if re.search(
                r"\bbecome\s+literate\s+again\b", source_text, re.IGNORECASE
            ) and re.search(
                r"重新(?:变得)?(?:有文化|文明)",
                compact_target,
            ):
                literal_literacy_keys.append(str(key))
            if re.search(r"\bpost[ -]?literacy\b", source_text, re.IGNORECASE) and re.search(
                r"后读写(?:时代|社会)?",
                compact_target,
            ):
                inconsistent_post_literacy_keys.append(str(key))
            if (
                re.search(r"\b(?:drop\s+us\s+a\s+line|email\s+us)\s+at\b", source_text, re.I)
                and "@" not in translated
                and re.search(r"[A-Za-z0-9._%+-]+at[A-Za-z0-9.-]+\.(?:com|org|net)\b", translated)
            ):
                malformed_email_keys.append(str(key))
            if re.search(
                r"\bnot(?:\s+kind\s+of)?\s+(?:valuing|appreciating|using|choosing|accessing)\b",
                source_text,
                re.IGNORECASE,
            ) and re.search(r"(?:不是|并非|并不是)不", compact_target):
                reversed_negative_valuation_keys.append(str(key))

        problems: list[str] = []
        if use_case_keys:
            problems.append(
                "For percentage use-case statements, make the feature the subject: "
                "'在90%的使用场景里 更快的转向响应都有帮助'. Do not present the "
                f"feature as an example of a scenario. Keys: {use_case_keys[:20]}"
            )
        if numeric_shorthand_keys:
            problems.append(
                "Complete colloquial numeric shorthand with its implied noun, for example "
                "'福特推出了720马力版本'; do not leave an attributive 的 at the cue end. "
                f"Keys: {numeric_shorthand_keys[:20]}"
            )
        if discourse_now_keys:
            problems.append(
                "A standalone 'Now.' is a discourse transition here. Translate it as '那么' "
                f"or '接下来', not the temporal adverb '现在'. Keys: {discourse_now_keys[:20]}"
            )
        if awkward_softness_keys:
            problems.append(
                "Render the model-year comparison directly as '26款比25款大约软20%' without "
                f"a trailing '针对26款来说' or duplicated model year. Keys: {awkward_softness_keys[:20]}"
            )
        if literal_irl_keys:
            problems.append(
                "Standalone conversational IRL means 'in real life'. Translate its meaning "
                f"naturally instead of leaving an unexplained acronym. Keys: {literal_irl_keys[:20]}"
            )
        if literal_yes_answer_keys:
            problems.append(
                "Translate a direct yes/no answer as '答案是肯定的' or '答案是 是的'. "
                "Do not replace yes with an adjective copied from the preceding question. "
                f"Keys: {literal_yes_answer_keys[:20]}"
            )
        if literal_fresh_slate_keys:
            problems.append(
                "'Fresh slate' means a clean starting point here, not a vehicle platform. "
                f"Keys: {literal_fresh_slate_keys[:20]}"
            )
        if literal_racing_line_keys:
            problems.append(
                "Use the established motorsport sense of 'racing line' (赛车线/最佳过弯线路), "
                f"not a route near a racetrack. Keys: {literal_racing_line_keys[:20]}"
            )
        if literal_price_argument_keys:
            problems.append(
                "In 'argue for your $X', express what a buyer may reasonably expect at that "
                f"price; do not say the buyer argues for the money. Keys: {literal_price_argument_keys[:20]}"
            )
        if repeated_ride_quality_keys:
            problems.append(
                "Keep stiff, bouncy, and crashy as distinct ride qualities; do not collapse "
                f"them into repeated '颠/颠簸' wording. Keys: {repeated_ride_quality_keys[:20]}"
            )
        if numeric_self_correction_keys:
            problems.append(
                "A spoken number followed by a hesitation and a more precise number is a "
                "self-correction. State only the corrected final value once. "
                f"Keys: {numeric_self_correction_keys[:20]}"
            )
        if incomplete_phone_aside_keys:
            problems.append(
                "Resolve the unambiguous omitted noun in the phone-fit aside: say that a "
                "smaller phone would fit and that the speaker's phone is a Pro Max. "
                f"Keys: {incomplete_phone_aside_keys[:20]}"
            )
        if collapsed_reference_keys:
            problems.append(
                "Keep a demonstrative evaluation of the previously mentioned object separate "
                "from the newly announced vehicle. Translate 'That is quite good, fellow "
                "Corolla coming up' as '那辆真不错 有辆Corolla开过来了', not as praise "
                f"attached to the Corolla. Keys: {collapsed_reference_keys[:20]}"
            )
        if literal_limo_stop_keys:
            problems.append(
                "In driving commentary, a 'limo stop' is a smooth chauffeur-style stop for "
                "passenger comfort. Express the smooth braking action naturally; do not use "
                f"the literal phrase 礼宾式停车. Keys: {literal_limo_stop_keys[:20]}"
            )
        if literal_body_adhesive_keys:
            problems.append(
                "When measured goop or adhesive is added inside body panels, use the automotive "
                "sense 车身结构胶 (and preserve the stated length), not the literal 胶状物. "
                f"Keys: {literal_body_adhesive_keys[:20]}"
            )
        if literal_run_cooler_keys:
            problems.append(
                "For an engine or vehicle that can 'run cooler', describe a lower operating "
                "temperature or improved cooling; do not say it 跑得更凉快. "
                f"Keys: {literal_run_cooler_keys[:20]}"
            )
        if ambiguous_downshift_second_keys:
            problems.append(
                "In 'downshift a second', 'a second' is a brief duration, not one gear. "
                "Translate it as 稍微降一下挡/短暂降挡, not 降一挡. "
                f"Keys: {ambiguous_downshift_second_keys[:20]}"
            )
        if fuel_economy_how_do_keys:
            problems.append(
                "After MPG or fuel-economy ratings, 'see how we do' asks how measured fuel "
                "economy performs. Do not translate it as how the car drives. "
                f"Keys: {fuel_economy_how_do_keys[:20]}"
            )
        if literal_bat_out_of_hell_keys:
            problems.append(
                "'Bat out of hell' is an idiom for moving or driving extremely fast. Translate "
                "the behavior naturally; never mention a literal hell bat. "
                f"Keys: {literal_bat_out_of_hell_keys[:20]}"
            )
        if parking_direction_keys:
            problems.append(
                "'Rolling down a parking structure/garage' describes proceeding downward inside "
                "it, not necessarily exiting it. Preserve that direction. "
                f"Keys: {parking_direction_keys[:20]}"
            )
        if rev_match_first_keys:
            problems.append(
                "In 'Rev match first? Yes, indeed it will', do not turn first into an instruction "
                "to select first gear. Express that the car will rev-match. "
                f"Keys: {rev_match_first_keys[:20]}"
            )
        if revised_component_keys:
            problems.append(
                "A source key ending in 'newly revised' modifies the audio/component named in "
                "the next source key. Do not turn it into a revised vehicle/model. "
                f"Keys: {revised_component_keys[:20]}"
            )
        if document_trim_variant_keys:
            problems.append(
                "The document repeatedly confirms Elantra N. Correct the one-off phonetic "
                "Elantra M variant instead of preserving a nonexistent trim. "
                f"Keys: {document_trim_variant_keys[:20]}"
            )
        if middle_seat_doubt_keys:
            problems.append(
                "'I don't know that you're putting somebody in the middle' doubts that anyone "
                "will sit there; it does not ask which person. "
                f"Keys: {middle_seat_doubt_keys[:20]}"
            )
        if model_year_generation_keys:
            problems.append(
                "A 'fourth model year' is the fourth annual model-year iteration, not the "
                "vehicle's fourth generation. "
                f"Keys: {model_year_generation_keys[:20]}"
            )
        if literal_fundamental_keys:
            problems.append(
                "Translate 'fundamental to' as 对……至关重要/不可或缺 or describe its "
                "foundational influence. Do not use the literal calque 对……来说太根本. "
                f"Keys: {literal_fundamental_keys[:20]}"
            )
        if literal_discourse_filler_keys:
            problems.append(
                "Omit parenthetical 'you know' when it is only a speech filler. Do not repeat "
                f"你知道 mechanically in Chinese subtitles. Keys: {literal_discourse_filler_keys[:20]}"
            )
        if stacked_connective_keys:
            problems.append(
                "Use one coherent Chinese discourse connector. Remove stacked combinations such "
                f"as 所以但 or 不过是的. Keys: {stacked_connective_keys[:20]}"
            )
        if almost_two_dozen_keys:
            problems.append(
                "'Almost two dozen' means close to 24. Use 接近24个/近24个, not the "
                f"contradictory 将近二十多个. Keys: {almost_two_dozen_keys[:20]}"
            )
        if literal_literacy_keys:
            problems.append(
                "In reading context, 'become literate again' means restoring literacy or becoming "
                "a society that values reading and writing again. Do not render it as 重新变得有文化. "
                f"Keys: {literal_literacy_keys[:20]}"
            )
        if inconsistent_post_literacy_keys:
            problems.append(
                "Use the document term 后文字时代 consistently for post-literacy; do not switch "
                f"to the misleading 后读写时代. Keys: {inconsistent_post_literacy_keys[:20]}"
            )
        if malformed_email_keys:
            problems.append(
                "An address introduced by 'email us' or 'drop us a line' is an email address. "
                f"Normalize its unambiguous at form with @ instead of a .com website. Keys: {malformed_email_keys[:20]}"
            )
        if reversed_negative_valuation_keys:
            problems.append(
                "Preserve the source negation directly. 'not valuing or choosing to access' "
                "means '不再重视 也不愿主动获取'; do not introduce a double negative such "
                "as '不是不重视', which reverses the meaning. "
                f"Keys: {reversed_negative_valuation_keys[:20]}"
            )
        if problems:
            return False, " ".join(problems)
        return True, ""

    def _validate_cross_key_boundaries(  # noqa: C901
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
            for token in self._context_ownership_tokens(source):
                source_owners.setdefault(token, set()).add(key)
        for token, owners in source_owners.items():
            owners.update(
                key
                for key, source in subtitle_dict.items()
                if self._source_owns_equivalent_token(token, source)
            )

        translated_owners: dict[str, set[str]] = {}
        for key, value in response_dict.items():
            translated = extract_text(value)
            compact = re.sub(r"[\s,，.。-]+", "", translated).lower()
            for token, owners in source_owners.items():
                token_compact = re.sub(r"[\s,，.。-]+", "", token)
                if token_compact.isdigit():
                    token_pattern = rf"(?<![a-z0-9]){re.escape(token_compact)}(?![a-z0-9])"
                else:
                    token_pattern = rf"(?<![a-z0-9]){re.escape(token_compact)}(?![a-z0-9])"
                rendered = bool(re.search(token_pattern, compact, flags=re.IGNORECASE))
                if not rendered:
                    rendered = any(
                        re.sub(r"[\s,，.。-]+", "", alias).lower() in compact
                        for alias in _CHINESE_TOKEN_EQUIVALENTS.get(token.upper(), ())
                    )
                if not rendered and token_compact.isdigit():
                    rendered = any(
                        self._localized_magnitude_rendered(
                            subtitle_dict.get(owner, ""),
                            token,
                            translated,
                        )
                        for owner in owners
                    )
                if rendered:
                    translated_owners.setdefault(token, set()).add(str(key))

        leaks = []
        for token, owners in source_owners.items():
            output_keys = translated_owners.get(token, set())
            leaked_keys = {
                key
                for key in output_keys - owners
                if not self._ownership_token_belongs_to_source(
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
            # Chinese may naturally omit or reposition a speech filler while the
            # material clause remains correctly aligned. Treat this as diagnostic
            # evidence instead of invalidating and retranslating a complete batch.
            logger.debug(
                "Ignoring non-semantic discourse-marker movement in keys: %s",
                discourse_leaks[:20],
            )

        if not check_adjacent_repetition:
            return True, ""

        ordered_keys = sorted(
            subtitle_dict,
            key=lambda key: int(key) if str(key).isdigit() else str(key),
        )

        missing_conditions: list[str] = []
        condition_translation_pattern = re.compile(
            r"(?:如果|若|要是|需要|必要|的话|一旦|只要|否则|假如|时候|"
            r"即使|即便|尽管|儘管|虽然|雖然|虽说|縱使|纵使)"
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

        misplaced_quantities: list[str] = []
        source_quantities = {
            key: self._source_quantity_tokens(source) for key, source in subtitle_dict.items()
        }
        for position, key in enumerate(ordered_keys):
            if not key.isdigit():
                continue
            translated_quantities = self._localized_quantity_tokens(
                extract_text(response_dict.get(key, ""))
            )
            if not translated_quantities:
                continue
            current_source_quantities = source_quantities.get(key, set())
            current_values = {value for value, _unit in current_source_quantities}
            neighbor_keys = [
                neighbor
                for neighbor in (
                    ordered_keys[position - 1] if position else "",
                    ordered_keys[position + 1] if position + 1 < len(ordered_keys) else "",
                )
                if neighbor and neighbor.isdigit() and abs(int(neighbor) - int(key)) == 1
            ]
            current_speaker = self._all_speaker_by_index.get(int(key), "")
            for quantity in sorted(translated_quantities):
                if quantity in current_source_quantities:
                    continue
                adjacent_owner = any(
                    quantity in source_quantities.get(neighbor, set())
                    and (
                        not current_speaker
                        or not self._all_speaker_by_index.get(int(neighbor), "")
                        or self._all_speaker_by_index.get(int(neighbor), "") == current_speaker
                    )
                    for neighbor in neighbor_keys
                )
                if quantity[0] in current_values or adjacent_owner:
                    misplaced_quantities.append(f"{key}:{quantity[0]}{quantity[1]}")
        if misplaced_quantities:
            return (
                False,
                "A translated number was paired with a semantic unit owned by a different "
                "source key. Keep each number-unit or number-counted-noun atom under the key "
                "that contains that complete source fact. "
                f"Misplaced quantities: {misplaced_quantities[:10]}",
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

        anticipated_imperatives: list[str] = []
        for left_key, right_key in zip(ordered_keys, ordered_keys[1:]):
            if not left_key.isdigit() or not right_key.isdigit():
                continue
            if int(right_key) != int(left_key) + 1:
                continue
            left_source = subtitle_dict[left_key]
            right_source = subtitle_dict[right_key]
            if not re.match(
                r"^why\s+don['’]t\s+we\s+(?:show|look|check)\b",
                right_source.strip(),
                flags=re.IGNORECASE,
            ):
                continue
            if re.search(r"\b(?:show|look|check)\b", left_source, flags=re.IGNORECASE):
                continue
            left_target = extract_text(response_dict.get(left_key, ""))
            if re.search(r"(?:不如|要不|何不).{0,10}(?:看看|展示|看一下)", left_target):
                anticipated_imperatives.append(f"{left_key}-{right_key}")
        if anticipated_imperatives:
            return (
                False,
                "An invitation owned by the next source key was translated early under the "
                "previous key. Keep 'why don't we show/look' under its owning key. "
                f"Anticipated invitations: {anticipated_imperatives[:10]}",
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
                "just",
                "kind",
                "know",
                "like",
                "really",
                "she",
                "sort",
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
            repeated_semantic_frame = (
                target_common >= 7
                and common_share >= 0.30
                and source_ratio < 0.30
                and common_share - source_ratio >= 0.08
            )
            repetitive_paraphrase = (
                shorter_target >= 10
                and target_ratio >= 0.60
                and source_ratio <= 0.25
                and target_ratio - source_ratio >= 0.30
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
                or repeated_semantic_frame
                or repetitive_paraphrase
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
        "年": ("year", "years"),
        "项目": ("project", "projects"),
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
            rf"(?:个|项|座|条|段|家|名|辆)?\s*"
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
        return quantity in cls._source_quantity_tokens(source)

    @classmethod
    def _source_quantity_tokens(cls, source: str) -> set[tuple[int, str]]:
        """Extract source number-unit atoms using the target unit's canonical key."""
        source_text = str(source or "")
        number_values = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        number_pattern = r"(?P<number>\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
        quantities: set[tuple[int, str]] = set()
        for canonical_unit, aliases in cls._QUANTITY_UNITS.items():
            unit_pattern = "|".join(sorted(map(re.escape, aliases), key=len, reverse=True))
            for match in re.finditer(
                rf"\b{number_pattern}\s*[-–—]?\s*"
                rf"(?:{unit_pattern})\b",
                source_text,
                flags=re.IGNORECASE,
            ):
                raw_number = match.group("number").lower()
                value = int(raw_number) if raw_number.isdigit() else number_values[raw_number]
                quantities.add((value, canonical_unit))
        return quantities

    @staticmethod
    def _boundary_tokens(text: str) -> set[str]:
        """Extract numbers and alphanumeric model/spec tokens with locked ownership."""
        return {
            match.group().lower()
            for match in re.finditer(
                r"\b(?:[A-Za-z][A-Za-z0-9]*[-/]\d+[A-Za-z0-9.-]*|"
                r"[A-Za-z]+\d+[A-Za-z0-9.-]*|"
                r"\d+(?:\.\d+)+[A-Za-z]+[A-Za-z0-9.-]*|"
                r"(?<![\w.])\d+[A-Za-z]+[A-Za-z0-9.-]*|"
                r"(?<![\d.])\d{2,4}(?![\d.]))\b",
                str(text or ""),
            )
        }

    @classmethod
    def _context_ownership_tokens(cls, text: str) -> set[str]:
        """Include numeric aliases for plural forms such as `37s` during context checks."""
        tokens = cls._boundary_tokens(text)
        tokens.update(
            match.group(1).lower()
            for match in re.finditer(
                r"\b([A-Z][A-Z0-9]{1,9})(?:s)?\b",
                str(text or ""),
            )
            if match.group(1).upper() in _CHINESE_TOKEN_EQUIVALENTS
        )
        aliases = {
            match.group(1)
            for token in tokens
            if (match := re.fullmatch(r"(\d{2,4})s", token, flags=re.IGNORECASE))
        }
        return tokens | aliases

    @staticmethod
    def _source_owns_equivalent_token(token: str, source: str) -> bool:
        """Recognize an acronym's expanded English form under the same source key."""
        phrases = _SOURCE_TOKEN_EQUIVALENTS.get(token.upper(), ())
        if not phrases:
            return False
        normalized = re.sub(r"[^a-z0-9]+", " ", str(source or "").lower()).strip()
        for phrase in phrases:
            phrase_normalized = re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()
            if re.search(
                rf"(?<![a-z0-9]){re.escape(phrase_normalized)}(?![a-z0-9])",
                normalized,
            ):
                return True
        if token.upper() == "RPM" and re.search(
            r"\b(?:power|torque|everything)\s+"
            r"(?:comes?|kicks?)\s+(?:in|on)\s+(?:(?:kind\s+of|really)\s+)?low\s+down\b",
            normalized,
        ):
            return True
        return False

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

    @classmethod
    def _ownership_token_belongs_to_source(
        cls,
        token: str,
        source: str,
        translated: str,
    ) -> bool:
        """Recognize exact source ownership across harmless identifier formatting.

        Host-read URLs and promo codes commonly alternate between ``B1M`` and
        ``theB1M``. Treating the fused English article as a different model token
        makes a correct URL look borrowed from the neighboring subtitle. Keep the
        exception narrow: only alphanumeric identifiers containing both letters
        and digits qualify.
        """
        if cls._numeric_token_belongs_to_compound_model(token, source, translated):
            return True

        if token.isdigit() and 1 <= int(token) <= 12:
            months = (
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            )
            if re.search(rf"(?<!\d){re.escape(token)}月", translated) and re.search(
                rf"\b{months[int(token) - 1]}\b",
                source,
                flags=re.IGNORECASE,
            ):
                return True

        if token.isdigit() and len(token) in {2, 4}:
            decade = token[-2:]
            source_decades = {
                match.group(1)[-2:]
                for match in re.finditer(
                    r"\b((?:\d{2}|\d{4}))s\b",
                    source,
                    flags=re.IGNORECASE,
                )
            }
            if decade in source_decades and re.search(
                rf"(?:19|20)?{re.escape(decade)}\s*年代",
                translated,
            ):
                return True

        if token.isdigit():
            target_value = Decimal(token)
            normalized_source_numbers = normalize_grouped_numbers(source)
            for source_number in re.findall(
                r"(?<!\d)\d+(?:\.\d+)?(?!\d)", normalized_source_numbers
            ):
                try:
                    source_value = Decimal(source_number)
                except InvalidOperation:
                    continue
                for multiplier, unit in (
                    (Decimal(100000000), "亿"),
                    (Decimal(10000), "万"),
                    (Decimal(1000), "千"),
                ):
                    if source_value == target_value * multiplier and re.search(
                        rf"(?<!\d){re.escape(token)}\s*{unit}", translated
                    ):
                        return True

        source_tokens = cls._context_ownership_tokens(source)
        token_compact = re.sub(r"[^a-z0-9]", "", token.lower())
        translated_compact = re.sub(r"[^a-z0-9]", "", translated.lower())
        if not token_compact or token_compact not in translated_compact:
            return False

        if re.search(r"[a-z]", token_compact) and re.search(r"\d", token_compact):
            identifier_parts = re.findall(r"[a-z]+|\d+", token_compact)
            identifier_pattern = r"[\s./_-]*".join(re.escape(part) for part in identifier_parts)
            if identifier_parts and re.search(
                rf"(?<![A-Za-z0-9]){identifier_pattern}(?![A-Za-z0-9])",
                source,
                flags=re.IGNORECASE,
            ):
                return True

        def is_identifier(value: str) -> bool:
            return bool(re.search(r"[a-z]", value) and re.search(r"\d", value))

        source_compacts = {
            re.sub(r"[^a-z0-9]", "", source_token.lower()) for source_token in source_tokens
        }
        if token_compact in source_compacts:
            return True
        if token_compact.isdigit():
            measurement_units = {
                "km": ("公里", "千米"),
                "kmh": ("公里", "千米", "时速"),
                "kph": ("公里", "千米", "时速"),
                "mph": ("英里", "时速"),
                "mpg": ("英里每加仑", "英里/加仑", "mpg"),
                "rpm": ("转速", "转/分", "rpm"),
                "hp": ("马力", "hp"),
                "lb": ("磅", "lb"),
                "lbs": ("磅", "lb", "lbs"),
                "ft": ("英尺", "ft"),
            }
            for source_compact in source_compacts:
                match = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z]+)", source_compact)
                if not match or match.group(1) != token_compact:
                    continue
                aliases = measurement_units.get(match.group(2))
                if aliases and any(alias.lower() in translated.lower() for alias in aliases):
                    return True
        for article in ("the", "an", "a"):
            if token_compact.startswith(article):
                bare = token_compact[len(article) :]
                if is_identifier(bare) and bare in source_compacts:
                    return True
            prefixed = f"{article}{token_compact}"
            if is_identifier(token_compact) and prefixed in source_compacts:
                return True
        return False

    @staticmethod
    def _localized_magnitude_rendered(source: str, token: str, translated: str) -> bool:
        """Detect an exact Chinese rendering of a source-owned magnitude."""
        compact_match = re.fullmatch(
            r"(\d+(?:\.\d+)?)(K|M|MM|B|BN|T)",
            token,
            flags=re.IGNORECASE,
        )
        if compact_match:
            token_match = re.search(
                rf"\b{re.escape(token)}\b",
                source,
                flags=re.IGNORECASE,
            )
            if not token_match:
                return False
            prefix = source[max(0, token_match.start() - 2) : token_match.start()]
            has_currency = bool(re.search(r"[$€£¥]\s*$", prefix))
            has_financial_context = bool(
                re.search(
                    r"\b(?:cost|costing|funding|finance|financing|investment|invested|"
                    r"budget|price|priced|worth|revenue|spending|expenditure|loan|debt)\b",
                    source,
                    flags=re.IGNORECASE,
                )
            )
            if not has_currency and not has_financial_context:
                return False
            raw_value = compact_match.group(1)
            suffix = compact_match.group(2).upper()
            multiplier = {
                "K": Decimal(1000),
                "M": Decimal(1000000),
                "MM": Decimal(1000000),
                "B": Decimal(1000000000),
                "BN": Decimal(1000000000),
                "T": Decimal(1000000000000),
            }[suffix]
        else:
            if not re.fullmatch(r"\d+(?:\.\d+)?", token):
                return False
            match = re.search(
                rf"\b{re.escape(token)}\s+(hundred|thousand|million|billion|trillion)\b",
                source,
                flags=re.IGNORECASE,
            )
            if not match:
                return False
            raw_value = token
            multiplier = {
                "hundred": Decimal(100),
                "thousand": Decimal(1000),
                "million": Decimal(1000000),
                "billion": Decimal(1000000000),
                "trillion": Decimal(1000000000000),
            }[match.group(1).lower()]
        value = Decimal(raw_value) * multiplier

        def render(number: Decimal) -> str:
            text = format(number, "f")
            return text.rstrip("0").rstrip(".") if "." in text else text

        candidates = {render(value)}
        for divisor, unit in (
            (Decimal(100000000), "亿"),
            (Decimal(10000), "万"),
            (Decimal(1000), "千"),
        ):
            coefficient = value / divisor
            exponent = coefficient.as_tuple().exponent
            if coefficient == coefficient.to_integral_value() or (
                isinstance(exponent, int) and exponent >= -3
            ):
                candidates.add(f"{render(coefficient)}{unit}")
        compact = re.sub(r"[\s,，。-]+", "", translated).lower()
        number_chars = r"0-9零〇一二两三四五六七八九十百千万亿点."
        for candidate in candidates:
            compact_candidate = re.sub(r"[\s,，。-]+", "", candidate).lower()
            if re.search(
                rf"(?<![{number_chars}]){re.escape(compact_candidate)}"
                rf"(?![{number_chars}])",
                compact,
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
            lowercase_runs = re.findall(
                r"(?<![A-Za-z-])[a-z]{2,}(?:\s+[a-z]{2,})+(?![A-Za-z-])",
                translated,
            )
            residue.extend(f"{key}:{run}" for run in lowercase_runs)
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

    def _validate_no_unowned_latin_names(
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        """Prevent a global proper noun from leaking into an unrelated subtitle key."""
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return True, ""

        leaked: list[str] = []
        latin_name_pattern = (
            r"(?<![A-Za-z0-9])(?:[A-Z][a-z]{2,}|"
            r"(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)"
            r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)(?![A-Za-z0-9])"
        )
        source_name_pattern = (
            r"[A-Za-z][A-Za-z0-9'’-]*|"
            r"(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)"
            r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"
        )
        for key, source in subtitle_dict.items():
            source_text = str(source or "")
            original_source = source_text
            if str(key).isdigit():
                original_source = self._all_source_by_index.get(int(key), source_text)
            source_tokens = {
                token.casefold() for token in re.findall(source_name_pattern, original_source)
            }
            source_tokens.update(
                part
                for token in tuple(source_tokens)
                for part in re.split(r"[-'’]", token)
                if len(part) >= 2
            )
            allowed_context = " ".join(
                value
                for value in (
                    self._confirmed_context_canonical(original_source),
                    self._confirmed_context_official_identifier_target(original_source),
                )
                if value
            ).casefold()
            translated = str(extract_text(response_dict.get(key, "")) or "")
            translated_tokens = re.findall(latin_name_pattern, translated)
            compact_source_text = re.sub(r"[^a-z0-9]", "", original_source.casefold())
            for token in translated_tokens:
                folded = token.casefold()
                compact = re.sub(r"[^a-z0-9]", "", folded)
                forms = {folded, folded.removesuffix("s"), compact}
                source_forms = source_tokens | {item.removesuffix("s") for item in source_tokens}
                source_forms.update(re.sub(r"[^a-z0-9]", "", item) for item in source_tokens)
                allowed_compact = re.sub(r"[^a-z0-9]", "", allowed_context)
                if forms & source_forms or folded in allowed_context or compact in allowed_compact:
                    continue
                if re.search(r"\d", compact) and compact in compact_source_text:
                    continue
                if str(key).isdigit() and re.search(
                    r"\b(?:it|this|that|these|those|they|one|car|truck|vehicle|thing|model)\b",
                    original_source,
                    flags=re.IGNORECASE,
                ):
                    index = int(key)
                    adjacent_source = " ".join(
                        self._all_source_by_index.get(neighbor, "")
                        for neighbor in (index - 1, index + 1)
                    )
                    adjacent_tokens = {
                        re.sub(r"[^a-z0-9]", "", item.casefold())
                        for item in re.findall(source_name_pattern, adjacent_source)
                    }
                    if compact in adjacent_tokens:
                        continue
                if str(key).isdigit():
                    index = int(key)
                    previous_source = self._all_source_by_index.get(index - 1, "")
                    next_source = self._all_source_by_index.get(index + 1, "")
                    dependent_adjacent = (
                        assess_english_boundary(previous_source, original_source).unstable
                        or assess_english_boundary(original_source, next_source).unstable
                    )
                    adjacent_source = f"{previous_source} {next_source}"
                    adjacent_tokens = {
                        re.sub(r"[^a-z0-9]", "", item.casefold())
                        for item in re.findall(source_name_pattern, adjacent_source)
                    }
                    if dependent_adjacent and compact in adjacent_tokens:
                        continue
                leaked.append(f"{key}:{token}")
        if not leaked:
            return True, ""
        return (
            False,
            "A Latin proper name appears in a subtitle whose source does not own it. Global "
            "context is read-only and must not leak a nearby model, person, or product into the "
            f"current key. Remove or replace these unowned names: {leaked[:20]}",
        )

    @staticmethod
    def _looks_like_placeholder_translation(text: str) -> bool:
        return BaseTranslator._looks_like_placeholder_translation(text)

    def _validate_no_placeholder_translations(
        self,
        response_dict: Dict[str, Any],
        subtitle_dict: Dict[str, str],
        extract_text,
    ) -> Tuple[bool, str]:
        diagnostics = inspect_placeholders(response_dict, subtitle_dict)
        if diagnostics:
            self._record_shadow_repair_plan(diagnostics)
            return False, diagnostics[0].message
        return True, ""

    def _looks_untranslated_for_cjk(
        self,
        text: str,
        original: str,
        source_language: str = "",
    ) -> bool:
        text = str(text or "").strip()
        original = str(original or "").strip()
        if not text:
            return True
        return is_untranslated_output(
            text,
            original,
            self.target_language,
            source_language,
        )

    def _inspect_preserved_tokens(
        self, response_dict: Dict[str, Any], subtitle_dict: Dict[str, str], extract_text
    ) -> tuple[QualityDiagnostic, ...]:
        """Delegate exact-value checks to the focused preservation module."""
        return inspect_preserved_tokens(
            response_dict,
            subtitle_dict,
            extract_text,
            target_language_value=self.target_language.value,
            localized_magnitude_rendered=self._localized_magnitude_rendered,
        )

    def _validate_preserved_tokens(
        self, response_dict: Dict[str, Any], subtitle_dict: Dict[str, str], extract_text
    ) -> Tuple[bool, str]:
        """Retain the legacy validator contract during typed-rule migration."""
        diagnostics = self._inspect_preserved_tokens(
            response_dict,
            subtitle_dict,
            extract_text,
        )
        if diagnostics:
            self._record_shadow_repair_plan(diagnostics)
            return False, legacy_preserved_token_message(diagnostics)
        return True, ""

    def _translate_locked_batch(
        self,
        subtitle_chunk: List[SubtitleProcessData],
        initial_feedback: str = "",
    ) -> List[SubtitleProcessData]:
        """Recover a failed reflective batch without discarding key ownership."""
        subtitle_dict = {
            str(data.index): self._source_for_translation(data.original_text)
            for data in subtitle_chunk
        }
        system_prompt = get_prompt(
            self._batch_translation_prompt_name(reflect=False),
            target_language=self.target_language.value,
            custom_prompt=self.custom_prompt,
        )
        system_prompt += self._dialogue_prompt_rules(subtitle_dict)
        style_rules = self._target_language_style_rules(subtitle_dict.values())
        if not is_glm_53_model(self.model):
            system_prompt += style_rules
        context_text = self.translation_context.render()
        if context_text:
            system_prompt += f"\n\n<global_context>\n{context_text}\n</global_context>"
        if is_glm_53_model(self.model):
            system_prompt += style_rules
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
                cache_namespace=self.cache_namespace,
                client=self.llm_client,
                reasoning_mode="disabled",
                max_output_tokens=4096,
            )
            try:
                response_dict = parse_json_object(get_response_text(response))
                self._normalize_chinese_response_connectives(response_dict)
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
                self._record_shadow_legacy_action(
                    strategy=(
                        RepairStrategy.RETRY
                        if _attempt + 1 < self.SINGLE_FALLBACK_MAX_ATTEMPTS
                        else RepairStrategy.NONE
                    ),
                    reasoning_mode=PlannedReasoningMode.DISABLED,
                )
                last_error = error_message
                messages.append(
                    self._assistant_followup_message(
                        response,
                        json.dumps(response_dict, ensure_ascii=False),
                    )
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
        own_tokens = set().union(
            *(self._context_ownership_tokens(text) for text in current.values())
        )
        neighbors = [
            *self._neighbor_context(current, before=True),
            *self._neighbor_context(current, before=False),
        ]
        neighbor_tokens = set().union(
            *(self._context_ownership_tokens(item["source"]) for item in neighbors),
            set(),
        )
        compact = re.sub(r"[\s,，.。-]+", "", translated).lower()
        borrowed = []
        for token in neighbor_tokens - own_tokens:
            if any(
                self._ownership_token_belongs_to_source(
                    token,
                    own_source,
                    translated,
                )
                for own_source in current.values()
            ):
                continue
            if any(
                self._localized_magnitude_rendered(
                    item["source"],
                    token,
                    translated,
                )
                for item in neighbors
            ):
                borrowed.append(token)
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

        single_prompt = get_prompt(
            "translate/single",
            target_language=self.target_language.value,
        )

        def _looks_untranslated(
            text: str,
            original: str,
            source_language: str = "",
        ) -> bool:
            if self.target_language.value not in {"简体中文", "繁体中文", "日本語", "韩语", "粤语"}:
                return False
            return self._looks_untranslated_for_cjk(text, original, source_language)

        failures: list[int] = []
        translated_items: list[SubtitleProcessData] = []
        provisional_items: list[SubtitleProcessData] = []

        for data in subtitle_chunk:
            if self._fatal_provider_error.is_set():
                raise RuntimeError(self._fatal_provider_message or "LLM provider request rejected")
            current = {str(data.index): data.original_text}
            messages = [
                {
                    "role": "system",
                    "content": (
                        single_prompt
                        + self._target_language_style_rules(current.values())
                        + self._dialogue_prompt_rules(current)
                    ),
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
            best_effort_item: SubtitleProcessData | None = None
            for attempt in range(self.SINGLE_FALLBACK_MAX_ATTEMPTS):
                try:
                    response = call_llm(
                        messages=messages,
                        model=self.model,
                        temperature=self.TRANSLATION_TEMPERATURE,
                        use_cache=self.use_cache,
                        cache_namespace=self.cache_namespace,
                        client=self.llm_client,
                        reasoning_mode="disabled",
                        max_output_tokens=2048,
                    )
                    translated_text = get_response_text(response)
                    if self.target_language.value in {"简体中文", "繁体中文", "粤语"}:
                        translated_text = self._normalize_stacked_chinese_connectives(
                            translated_text
                        )
                    if _looks_untranslated(
                        translated_text,
                        data.original_text,
                        data.source_language,
                    ):
                        raise RuntimeError(
                            f"Single item translation did not produce {self.target_language.value}: "
                            f"{translated_text!r}"
                        )
                    if self._looks_like_placeholder_translation(translated_text):
                        raise RuntimeError(
                            "Single item translation returned a placeholder instead of a "
                            f"translation: {translated_text!r}"
                        )
                    # Keep the newest genuine target-language answer as a recovery
                    # candidate. A later strict fact/alignment check may still reject
                    # it, but discarding it would leave an avoidable empty subtitle.
                    best_effort_item = replace(data, translated_text=translated_text)
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
                    natural_chinese_ok, natural_chinese_error = (
                        self._validate_natural_chinese_degree_constructions(
                            current_response,
                            current_source,
                            lambda value: str(value),
                        )
                    )
                    if not natural_chinese_ok:
                        raise RuntimeError(natural_chinese_error)
                    technical_terms_ok, technical_terms_error = (
                        self._validate_no_weakened_technical_compounds(
                            current_response,
                            current_source,
                            lambda value: str(value),
                        )
                    )
                    if not technical_terms_ok:
                        raise RuntimeError(technical_terms_error)
                    contextual_chinese_ok, contextual_chinese_error = (
                        self._validate_natural_chinese_contextual_constructions(
                            current_response,
                            current_source,
                            lambda value: str(value),
                        )
                    )
                    if not contextual_chinese_ok:
                        raise RuntimeError(contextual_chinese_error)
                    self._validate_single_context_ownership(current, translated_text)
                    translated_items.append(replace(data, translated_text=translated_text))
                    last_error = None
                    break
                except Exception as error:
                    self._record_shadow_legacy_action(
                        strategy=(
                            RepairStrategy.RETRY
                            if attempt + 1 < self.SINGLE_FALLBACK_MAX_ATTEMPTS
                            else RepairStrategy.NONE
                        ),
                        reasoning_mode=PlannedReasoningMode.DISABLED,
                    )
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
                if best_effort_item is not None:
                    provisional_items.append(best_effort_item)

        if failures:
            raise PartialTranslationError(
                f"Single item translation failed for {len(failures)}/{len(subtitle_chunk)} entries: {failures}",
                completed=translated_items,
                failed_indices=failures,
                provisional=provisional_items,
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
            self._record_shadow_legacy_action(
                strategy=RepairStrategy.BATCH_REBUILD,
                reasoning_mode=PlannedReasoningMode.DISABLED,
            )
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
        elif not fallback_ok:
            self._record_shadow_legacy_action(
                strategy=RepairStrategy.NONE,
                reasoning_mode=PlannedReasoningMode.DISABLED,
            )
        return translated_items

    def _widen_batch_tail_shift_repair(
        self,
        source_list: List[SubtitleProcessData],
        boundary: int,
        response_dict: Dict[str, str],
        repair_sources: List[SubtitleProcessData],
        repetition_failure: bool,
    ) -> List[SubtitleProcessData]:
        """Rebuild a batch when a near-identical tail pair can terminate a key shift."""
        if not repetition_failure or self.batch_num < 4:
            return repair_sources
        left, right = source_list[boundary - 1], source_list[boundary]
        offset = (right.index - 1) % self.batch_num
        if offset < self.batch_num - 2:
            return repair_sources

        left_target = self._normalized_target_text(response_dict.get(str(left.index), ""))
        right_target = self._normalized_target_text(response_dict.get(str(right.index), ""))
        left_source = self._normalized_source_text(left.original_text)
        right_source = self._normalized_source_text(right.original_text)
        if min(len(left_target), len(right_target)) < 6:
            return repair_sources
        target_ratio = difflib.SequenceMatcher(None, left_target, right_target).ratio()
        source_ratio = difflib.SequenceMatcher(None, left_source, right_source).ratio()
        if target_ratio < 0.82 or source_ratio >= 0.45:
            return repair_sources

        start = max(0, boundary - offset)
        end = min(len(source_list), start + self.batch_num)
        widened = source_list[start:end]
        if len(widened) < 2:
            return repair_sources
        logger.warning(
            "Near-identical translations at batch tail %s-%s may terminate a multi-key "
            "shift; rebuilding original batch %s-%s without native reasoning",
            left.index,
            right.index,
            widened[0].index,
            widened[-1].index,
        )
        return widened

    def _finalize_translated_list(
        self,
        source_list: List[SubtitleProcessData],
        translated_list: List[SubtitleProcessData],
    ) -> List[SubtitleProcessData]:
        """Repair confirmed alignment errors and high-confidence adjacent repetition."""
        if self.is_reflect and is_lmstudio_qwen_38_request(
            self.model,
            self.llm_client,
        ):
            return self._finalize_local_qwen_translated_list(
                source_list,
                translated_list,
            )
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
                                source_key=str(item.index),
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
            ownership_failure = not valid and "Cross-key duplicates:" in error
            if not (dependent_boundary or repetition_failure or ownership_failure):
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
            repair_sources = self._widen_batch_tail_shift_repair(
                source_list,
                boundary,
                response_dict,
                repair_sources,
                repetition_failure,
            )
            repair_source_dict = {str(item.index): item.original_text for item in repair_sources}
            try:
                repaired = self._translate_locked_batch(
                    repair_sources,
                    initial_feedback=(
                        error
                        if repetition_failure or ownership_failure
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
                try:
                    repaired = self._translate_locked_batch(
                        repair_sources,
                        initial_feedback=(
                            "The previous boundary repair still duplicated or anticipated "
                            "neighboring meaning: " + repaired_error
                        ),
                    )
                except Exception as repair_error:
                    logger.warning(
                        "Adjacent translation retry failed for subtitles %s and %s; "
                        "retaining the previous translations: %s",
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
            previous_source = self._all_source_by_index.get(index - 1, "")
            next_source = self._all_source_by_index.get(index + 1, "")
            try:
                repaired_text = self._translate_alignment_item(
                    source_dict[key],
                    source_key=key,
                    previous_source=previous_source,
                    next_source=next_source,
                    repair_hint=self._selective_semantic_repair_hint(
                        source_dict[key],
                        translated_dict.get(key, ""),
                        previous_source=previous_source,
                        next_source=next_source,
                    ),
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
        self._repair_high_confidence_semantic_asr_fallbacks(
            source_list,
            translated_by_index,
        )

        # Recheck deterministic prose risks after broad boundary-fluency rewrites
        # so a late polish pass cannot reintroduce a confirmed calque.
        translated_dict = {
            str(item.index): translated_by_index[item.index].translated_text
            for item in source_list
            if item.index in translated_by_index
        }
        prose_candidates = self._strong_chinese_prose_candidates(
            source_dict,
            translated_dict,
        )
        for key in prose_candidates:
            index = int(key)
            try:
                repaired_text = self._translate_alignment_item(
                    source_dict[key],
                    source_key=key,
                    previous_source=self._all_source_by_index.get(index - 1, ""),
                    next_source=self._all_source_by_index.get(index + 1, ""),
                    repair_hint=self._chinese_prose_repair_hint(
                        source_dict[key],
                        translated_dict.get(key, ""),
                    ),
                )
            except Exception as error:
                fallback_text = self._deterministic_chinese_prose_fallback(
                    source_dict[key],
                    translated_dict.get(key, ""),
                )
                if fallback_text != translated_dict.get(
                    key, ""
                ) and not self._chinese_prose_repair_hint(source_dict[key], fallback_text):
                    translated_by_index[index] = replace(
                        translated_by_index[index],
                        translated_text=fallback_text,
                    )
                    logger.info(
                        "Deterministic prose fallback corrected key after LLM failure: %s",
                        key,
                    )
                    continue
                logger.warning(
                    "Final prose repair failed for subtitle %s: %s",
                    key,
                    error,
                )
                continue
            translated_by_index[index] = replace(
                translated_by_index[index],
                translated_text=repaired_text,
            )
            logger.info("Final prose repair corrected key: %s", key)
        # A final style rewrite may reintroduce a fragment that an earlier
        # boundary pass had already completed. Reapply only deterministic,
        # source-anchored repairs after all LLM output has settled.
        self._repair_high_confidence_semantic_asr_fallbacks(
            source_list,
            translated_by_index,
        )
        return [
            translated_by_index[item.index]
            for item in source_list
            if item.index in translated_by_index
        ]

    def _finalize_local_qwen_translated_list(
        self,
        source_list: List[SubtitleProcessData],
        translated_list: List[SubtitleProcessData],
    ) -> List[SubtitleProcessData]:
        """Run only evidence-backed repairs that justify local inference cost.

        A full reflective audit can issue nearly one request per subtitle on a
        memory-bound local model. The Qwen path instead trusts the validated
        batch result, applies deterministic corrections, and invokes native
        reasoning only when an existing semantic detector proves a concrete
        role or ASR mismatch.
        """
        translated_by_index = {item.index: item for item in translated_list}
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
            if not key.isdigit() or int(key) not in translated_by_index:
                continue
            index = int(key)
            previous_source = self._all_source_by_index.get(index - 1, "")
            next_source = self._all_source_by_index.get(index + 1, "")
            repair_hint = self._selective_semantic_repair_hint(
                source_dict[key],
                translated_dict.get(key, ""),
                previous_source=previous_source,
                next_source=next_source,
            )
            if not repair_hint:
                continue
            try:
                repaired_text = self._translate_alignment_item(
                    source_dict[key],
                    source_key=key,
                    previous_source=previous_source,
                    next_source=next_source,
                    repair_hint=repair_hint,
                )
            except Exception as error:
                logger.warning(
                    "Local Qwen semantic repair failed for subtitle %s; retaining "
                    "the validated translation: %s",
                    key,
                    error,
                )
                continue
            translated_by_index[index] = replace(
                translated_by_index[index],
                translated_text=repaired_text,
            )
            logger.info("Local Qwen semantic repair corrected key: %s", key)

        self._repair_contextual_nuclear_plant_terms(
            source_list,
            translated_by_index,
        )
        self._remove_stranded_chinese_subject_tails(
            source_list,
            translated_by_index,
        )
        self._repair_high_confidence_semantic_asr_fallbacks(
            source_list,
            translated_by_index,
        )
        return [
            translated_by_index[item.index]
            for item in source_list
            if item.index in translated_by_index
        ]

    def _allows_deterministic_boundary_edit(
        self, left: SubtitleProcessData, right: SubtitleProcessData
    ) -> bool:
        """Local text moves cannot cross a known turn, language, or visible pause."""
        left_speaker = self._all_speaker_by_index.get(left.index)
        right_speaker = self._all_speaker_by_index.get(right.index)
        return (
            right.index == left.index + 1
            and not (left_speaker and right_speaker and left_speaker != right_speaker)
            and not (
                left.source_language
                and right.source_language
                and left.source_language != right.source_language
            )
            and self._gap_after_index.get(left.index, 0) < self.SEPARATED_DISPLAY_GAP_MS
        )

    def _remove_stranded_chinese_subject_tails(
        self,
        source_list: List[SubtitleProcessData],
        translated_by_index: Dict[int, SubtitleProcessData],
    ) -> None:
        """Remove only a duplicated Chinese subject split from its following auxiliary."""
        if self.target_language.value not in {"简体中文", "繁体中文", "粤语"}:
            return
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
        pronoun_forms = {
            "i": {"i", "me", "my", "mine", "myself"},
            "you": {"you", "your", "yours", "yourself", "yourselves"},
            "he": {"he", "him", "his", "himself"},
            "she": {"she", "her", "hers", "herself"},
            "it": {"it", "its", "itself"},
            "we": {"we", "us", "our", "ours", "ourselves"},
            "they": {"they", "them", "their", "theirs", "themselves"},
        }
        pronouns = "我们|你们|他们|她们|它们|我們|你們|他們|她們|它們|我|你|他|她|它"
        for left, right in zip(source_list, source_list[1:]):
            left_item = translated_by_index.get(left.index)
            right_item = translated_by_index.get(right.index)
            if (
                left_item is None
                or right_item is None
                or not self._allows_deterministic_boundary_edit(left, right)
                or re.search(r"[.!?][\"')\]]*\s*$", left.original_text)
            ):
                continue
            left_tokens = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", left.original_text.lower())
            right_tokens = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", right.original_text.lower())
            subject = re.search(r"[.!?]\s+(i|you|he|she|it|we|they)\s*$", left.original_text, re.I)
            if (
                len(left_tokens) < 2
                or not right_tokens
                or right_tokens[0] not in auxiliary_heads
                or subject is None
            ):
                continue
            # A separated tail plus the same following subject is evidence of
            # duplication. A final pronoun alone can be the preceding verb's object.
            tail = re.search(rf"[\s，,]+({pronouns})\s*$", left_item.translated_text)
            head = re.match(rf"({pronouns})", right_item.translated_text.lstrip())
            if tail is None or head is None or tail.group(1) != head.group(1):
                continue
            cleaned = left_item.translated_text[: tail.start()].strip()
            # Whitespace alone does not prove the tail is dangling: it may be
            # the preceding verb's object. Keep all earlier pronoun mentions.
            forms = pronoun_forms[subject.group(1).lower()]
            required_mentions = sum(
                token.replace("’", "'").partition("'")[0] in forms for token in left_tokens[:-1]
            )
            remaining_mentions = re.findall(pronouns, cleaned).count(tail.group(1))
            if remaining_mentions < required_mentions:
                continue
            if cleaned and cleaned != left_item.translated_text.strip():
                translated_by_index[left.index] = replace(
                    left_item,
                    translated_text=cleaned,
                )

    def _repair_high_confidence_semantic_asr_fallbacks(
        self,
        source_list: List[SubtitleProcessData],
        translated_by_index: Dict[int, SubtitleProcessData],
    ) -> None:
        """Apply local text corrections without guessing missing source facts."""
        if self.target_language != TargetLanguage.SIMPLIFIED_CHINESE:
            return

        for item in source_list:
            current = translated_by_index.get(item.index)
            if current is None:
                continue
            source = item.original_text.strip()
            translated = current.translated_text.strip()
            repaired = translated
            if (
                re.search(r"\bCobalt\b", source, re.I)
                and not re.search(r"\bCruze\b", source, re.I)
                and "科鲁兹" in translated
                and not re.search(r"\bCobalt\b", translated, re.I)
            ):
                # Cobalt and Cruze are distinct Chevrolet models; a localized
                # Cruze name is not a valid rendering of source-owned Cobalt.
                repaired = translated.replace("科鲁兹", "Cobalt")
            elif re.fullmatch(
                r"(?:i|we|you|he|she|they)\s+(?:bought|drove|kept|sold|took)\s+this[.!?]?",
                source,
                re.I,
            ):
                if re.search(r"(?:这|那)辆$", translated):
                    repaired = f"{translated}车"
            if repaired != translated:
                translated_by_index[item.index] = replace(current, translated_text=repaired)

    def _pending_alignment_repair_windows(
        self,
        source_list: List[SubtitleProcessData],
        translated_by_index: Dict[int, SubtitleProcessData],
        pending_keys: List[int],
    ) -> List[List[SubtitleProcessData]]:
        """Return tight context windows around confirmed alignment errors.

        Pending keys have already failed isolated repair. Include one adjacent key on
        each side so the model can restore a local shift, but do not retranslate an
        otherwise valid 20-key batch because one item failed a strict validator.
        """
        pending = set(pending_keys)
        valid_positions = sorted(
            position
            for position, item in enumerate(source_list)
            if item.index in pending and item.index in translated_by_index
        )
        if not valid_positions:
            return []

        clusters: list[list[int]] = []
        for position in valid_positions:
            if clusters and position - clusters[-1][-1] <= 2:
                clusters[-1].append(position)
            else:
                clusters.append([position])

        windows: list[List[SubtitleProcessData]] = []
        for cluster in clusters:
            start = max(0, cluster[0] - self.ALIGNMENT_REPAIR_CONTEXT)
            end = min(len(source_list), cluster[-1] + self.ALIGNMENT_REPAIR_CONTEXT + 1)
            window = [item for item in source_list[start:end] if item.index in translated_by_index]
            if window:
                windows.append(window)
        return windows

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
        """Correct a locally explicit power plant, not an ambiguous neighboring plant."""
        if self.target_language != TargetLanguage.SIMPLIFIED_CHINESE:
            return
        for item in source_list:
            source = item.original_text.lower()
            if (
                not re.search(r"\bnuclear(?:\s+power)?\s+plants?\b", source)
                or len(re.findall(r"\bplants?\b", source)) != 1
                or re.search(r"\b(?:factor(?:y|ies)|manufactur\w*|assembly)\b", source)
            ):
                continue
            translated_item = translated_by_index.get(item.index)
            if (
                translated_item is None
                or translated_item.translated_text.count("工厂") != 1
                or "核电" in translated_item.translated_text
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
        if not windows:
            return
        # Only disjoint groups can run concurrently. Shared cues are committed
        # sequentially and rechecked against already accepted neighboring windows.
        groups: list[list[List[SubtitleProcessData]]] = []
        for window in windows:
            if groups and window[0].index <= groups[-1][-1][-1].index:
                groups[-1].append(window)
            else:
                groups.append([window])
        workers = min(self.thread_num, len(groups))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self._repair_chinese_fluency_group,
                    group,
                    {
                        item.index: translated_by_index[item.index]
                        for window in group
                        for item in window
                    },
                )
                for group in groups
            ]
            for future in as_completed(futures):
                translated_by_index.update(future.result())

    def _repair_chinese_fluency_group(
        self,
        windows: list[List[SubtitleProcessData]],
        current: Dict[int, SubtitleProcessData],
    ) -> Dict[int, SubtitleProcessData]:
        accepted: list[List[SubtitleProcessData]] = []
        for window in windows:
            if not self.is_running:
                break
            _, repaired, repair_error = self._repair_chinese_fluency_window_with_retries(
                window, [current[item.index] for item in window]
            )
            if repaired is None:
                logger.warning(
                    "Chinese boundary fluency repair failed for subtitles %s; retaining "
                    "the previous translations: %s",
                    [item.index for item in window],
                    repair_error,
                )
                continue
            candidate = {item.index: item for item in repaired}
            if set(candidate) != {item.index for item in window} or len(candidate) != len(repaired):
                logger.warning("Discarding fluency repair with changed cue ownership")
                continue
            changed = {
                index
                for index, item in candidate.items()
                if item.translated_text != current[index].translated_text
            }
            proposed = {**current, **candidate}
            try:
                for previous in accepted:
                    if changed.intersection(item.index for item in previous):
                        self._validate_chinese_fluency_repair(
                            previous,
                            [current[item.index] for item in previous],
                            [proposed[item.index] for item in previous],
                        )
            except Exception as error:
                logger.warning(
                    "Discarding overlapping repair that invalidates an accepted window: %s",
                    error,
                )
                continue
            current.update(candidate)
            accepted.append(window)
            logger.info(
                "Chinese boundary fluency repair corrected subtitles: %s",
                [item.index for item in window],
            )
        return current

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
        deterministic = self._deterministic_chinese_fluency_fallback(
            window,
            current,
            multispeaker=self._is_multispeaker_document(),
            protected_boundaries=frozenset(
                left.index
                for left, right in zip(window, window[1:])
                if not self._allows_deterministic_boundary_edit(left, right)
            ),
        )
        if deterministic is not None:
            return window, deterministic, None
        repair_error: Exception | None = None
        feedback = ""
        fresh_reasoning_attempted = self._should_reason_about_chinese_fluency_window(
            window,
            current,
        )
        if fresh_reasoning_attempted:
            try:
                candidate = self._rewrite_chinese_fluency_window_fresh(
                    window,
                    reasoning=True,
                )
                self._validate_chinese_fluency_repair(window, current, candidate)
                self._record_reasoning_metric("accepted_repairs")
                return window, candidate, None
            except Exception as error:
                self._record_shadow_legacy_action(
                    strategy=RepairStrategy.LOCAL_REWRITE,
                    reasoning_mode=PlannedReasoningMode.DISABLED,
                )
                repair_error = error
                feedback = str(error)
                self._record_reasoning_metric("rejected_repairs")
                self._record_reasoning_metric("fallback_requests")
        for attempt in range(self.CHINESE_FLUENCY_ANCHORED_MAX_ATTEMPTS):
            use_reasoning = (
                not fresh_reasoning_attempted
                and self._should_reason_about_chinese_fluency_window(
                    window, current, feedback=feedback
                )
            )
            if attempt:
                # The first confirmed hard repair may use native reasoning. Later
                # attempts already have concrete validator feedback; repeating a
                # long reasoning pass is slower and can exhaust the output budget
                # before DeepSeek emits the final JSON.
                use_reasoning = False
            try:
                candidate = self._rewrite_chinese_fluency_window(
                    window,
                    current,
                    feedback=feedback,
                    reasoning_override=use_reasoning,
                )
                self._validate_chinese_fluency_repair(window, current, candidate)
                if use_reasoning:
                    self._record_reasoning_metric("accepted_repairs")
                return window, candidate, None
            except Exception as error:
                self._record_shadow_legacy_action(
                    strategy=RepairStrategy.LOCAL_REWRITE,
                    reasoning_mode=PlannedReasoningMode.DISABLED,
                )
                repair_error = error
                feedback = str(error)
                if use_reasoning:
                    self._record_reasoning_metric("rejected_repairs")
                    self._record_reasoning_metric("fallback_requests")
        # Repeated feedback can anchor the model to the defective wording. Make
        # one fresh non-reasoning attempt without showing the old Chinese after
        # the conservative repair path is exhausted.
        fresh_feedback = ""
        for _attempt in range(self.CHINESE_FLUENCY_FRESH_MAX_ATTEMPTS):
            try:
                candidate = self._rewrite_chinese_fluency_window_fresh(
                    window,
                    feedback=fresh_feedback,
                )
                self._validate_chinese_fluency_repair(window, current, candidate)
                return window, candidate, None
            except Exception as error:
                self._record_shadow_legacy_action(
                    strategy=RepairStrategy.NONE,
                    reasoning_mode=PlannedReasoningMode.DISABLED,
                )
                repair_error = error
                fresh_feedback = str(error)
        # A failed rewrite does not prove that the validated text was defective:
        # the candidate itself may have introduced the rejected structure. Before
        # retaining the old text, independently verify its combined fidelity and
        # per-cue readability. Only a confirmed residual defect receives one final,
        # issue-guided rewrite, keeping the extra cost off normal windows.
        try:
            # The previous text must pass the same structural gate as a rewrite.
            # A combined-fidelity verdict alone can overlook an individually
            # unreadable cue such as ``真正的 / 全球场馆网络``.
            self._validate_chinese_fluency_repair(window, current, current)
            return window, current, None
        except Exception as current_error:
            repair_error = current_error

        final_reasoning = self._should_reason_about_chinese_fluency_window(
            window,
            current,
            feedback=str(repair_error or ""),
        )
        self._record_shadow_legacy_action(
            strategy=RepairStrategy.LOCAL_REWRITE,
            reasoning_mode=(
                PlannedReasoningMode.ENABLED if final_reasoning else PlannedReasoningMode.DISABLED
            ),
        )
        try:
            candidate = self._rewrite_chinese_fluency_window_fresh(
                window,
                feedback=(
                    "The existing translations also failed independent readability and "
                    f"fidelity validation: {repair_error}"
                ),
                reasoning=final_reasoning,
            )
            self._validate_chinese_fluency_repair(window, current, candidate)
            if final_reasoning:
                self._record_reasoning_metric("accepted_repairs")
            return window, candidate, None
        except Exception as error:
            self._record_shadow_legacy_action(
                strategy=RepairStrategy.NONE,
                reasoning_mode=PlannedReasoningMode.DISABLED,
            )
            repair_error = error
            if final_reasoning:
                self._record_reasoning_metric("rejected_repairs")
        return window, None, repair_error

    def _rewrite_chinese_fluency_window_fresh(
        self,
        source_items: List[SubtitleProcessData],
        *,
        feedback: str = "",
        reasoning: bool = False,
    ) -> List[SubtitleProcessData]:
        """Retranslate one confirmed broken window without defective target anchoring."""
        payload: Dict[str, Dict[str, Any]] = {}
        for item in source_items:
            canonical = self._confirmed_context_canonical(item.original_text)
            value = {
                "source": self._source_for_translation(item.original_text),
                "source_language": item.source_language,
            }
            if canonical:
                value["confirmed_canonical_name"] = canonical
            speaker = self._all_speaker_by_index.get(item.index, "")
            if speaker:
                value["speaker"] = speaker
            value.update(self._display_boundary_metadata(item.index))
            payload[str(item.index)] = value
        first_index = source_items[0].index
        last_index = source_items[-1].index
        request = {
            "items": payload,
            "readonly_context": {
                "previous_source": self._all_source_by_index.get(first_index - 1, ""),
                "next_source": self._all_source_by_index.get(last_index + 1, ""),
            },
        }
        japanese_window = any(item.source_language == "ja" for item in source_items)
        if japanese_window:
            request["combined_source"] = "".join(
                self._source_for_translation(item.original_text) for item in source_items
            )
        japanese_window_guidance = (
            " For this Japanese window, combined_source is the reconstructed utterance and item "
            "keys are timing slots rather than semantic ownership boundaries. Translate the full "
            "utterance first, then repartition natural Chinese across every key."
            if japanese_window
            else ""
        )
        retry_instruction = (
            " The previous fresh rewrite was rejected for this reason: "
            + feedback
            + ". Correct that exact defect without copying any old translation."
            if feedback
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"Translate this confirmed broken subtitle window into "
                    f"{self.target_language.value} from scratch. Read all source items as one "
                    "continuous utterance, reconstruct its complete meaning, then distribute it "
                    "across the original keys so each cue is as readable as source ownership "
                    "permits. gap_after_ms describes the pause after an item: continuous items may "
                    "form one display sequence, while separated items must not end with a naked "
                    "number, unfinished predicate, modifier, or function word. Keep every "
                    "fact, name, number, negation, comparison, speaker turn, and conclusion exactly "
                    "once. Do not mirror an English boundary that strands a Chinese subject, "
                    "predicate, object, modifier, connective, or vague filler-only frame. Omit oral "
                    "fillers that carry no meaning. You may add only minimal non-material Chinese "
                    "grammatical scaffolding, such as a pronoun, demonstrative, classifier, or an "
                    "already established head noun, when a cue would otherwise be unreadable. Such "
                    "scaffolding must add no fact and must never repeat a name, number, distinct "
                    "action, opinion, or conclusion. You may move only a copula, function word, "
                    "pronoun, classifier, or other non-material grammatical scaffolding between "
                    "adjacent keys. Never move a material action or event such as starting "
                    "operation, construction, lifting, growth, approval, or completion into a key "
                    "whose source does not contain it. Exception: when adjacent source keys split "
                    "one inseparable semantic atom such as predicate-complement, modifier-head, or "
                    "number-unit, you may realize that atom across those two keys with minimal "
                    "shared grammatical or lexical scaffolding so separated displays are readable. "
                    "Do not move it beyond that pair or duplicate the combined fact. If the source "
                    "splits a modifier from its head noun, keep a concise continuation instead of "
                    "anticipating the next key's action. Preserve the combined meaning exactly once. "
                    "For a three-key chain "
                    "shaped as 将/把 + object, "
                    "locative phrase, then action, redistribute it into independently readable cues "
                    "such as 这里使用 + object, 它们会安装在 + location, then 并逐步 + action; "
                    "never return "
                    "the original stranded chain. Speaker and readonly_context values are context "
                    "only and must not appear in the output. When an item has "
                    "confirmed_canonical_name, reproduce that exact Latin string in the same key; "
                    "never translate, transliterate, abbreviate, or respell it. Return only "
                    '{"translations": {"key": "text"}} with every input key exactly once.'
                    + japanese_window_guidance
                    + retry_instruction
                )
                + self._target_language_style_rules(
                    self._source_for_translation(item.original_text) for item in source_items
                ),
            },
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ]
        response = call_llm(
            messages=messages,
            model=self.model,
            temperature=self.TRANSLATION_TEMPERATURE,
            use_cache=self.use_cache,
            cache_namespace=self.cache_namespace,
            client=self.llm_client,
            reasoning_mode="enabled" if reasoning else "disabled",
            max_output_tokens=(self.REASONING_REWRITE_MAX_OUTPUT_TOKENS if reasoning else 4096),
            **({"reasoning_effort": "low"} if reasoning else {}),
        )
        if reasoning:
            self._record_reasoning_metric("rewrite_requests")
        try:
            response_text = get_response_text(response)
            if reasoning:
                self._record_reasoning_metric("final_answers")
        except ValueError:
            if reasoning:
                self._record_reasoning_metric("no_final_answers")
            raise
        result = parse_json_object(response_text).get("translations")
        expected = set(payload)
        if not isinstance(result, dict) or set(map(str, result)) != expected:
            raise ValueError("fresh fluency repair must return every input key exactly once")
        repaired: list[SubtitleProcessData] = []
        for item in source_items:
            text = str(result[str(item.index)]).strip()
            text = self._normalize_stacked_chinese_connectives(text)
            if not text or self._looks_like_placeholder_translation(text):
                raise ValueError(f"invalid fresh fluency repair for key {item.index}")
            repaired.append(replace(item, translated_text=text))
        return repaired

    @staticmethod
    def _deterministic_chinese_fluency_fallback(
        window: List[SubtitleProcessData],
        current: List[SubtitleProcessData],
        *,
        multispeaker: bool = False,
        protected_boundaries: frozenset[int] = frozenset(),
    ) -> Optional[List[SubtitleProcessData]]:
        """Move existing local wording; never synthesize a remembered translation."""
        current_by_index = {item.index: item for item in current}
        repaired_by_index = dict(current_by_index)
        changed = False

        for left, right in zip(window, window[1:]):
            if left.index in protected_boundaries:
                continue
            left_source = left.original_text.strip()
            right_source = right.original_text.strip()
            left_translation = repaired_by_index[left.index].translated_text.strip()
            right_translation = repaired_by_index[right.index].translated_text.strip()

            if LLMTranslator._source_ends_with_parallel_plural_attributes(
                left_source
            ) and re.search(r"的[\s，。！？；：、,.!?;:…]*$", left_translation):
                cleaned_left = re.sub(
                    r"的(?P<punct>[\s，。！？；：、,.!?;:…]*)$",
                    r"\g<punct>",
                    left_translation,
                ).rstrip()
                if cleaned_left:
                    repaired_by_index[left.index] = replace(
                        repaired_by_index[left.index],
                        translated_text=cleaned_left,
                    )
                    left_translation = cleaned_left
                    changed = True

            # English can place a copular emphasis immediately before a long
            # pause while natural Chinese carries the same emphasis with the
            # result construction in the following cue (for example 一...就...).
            # Move no facts: remove only the redundant dangling Chinese copula
            # when the following cue already contains the corresponding 就.
            if re.search(
                r"\b(?:it|this|that)\s+(?:is|was)\s+"
                r"(?:(?:really|actually)\s+)?just[,.!?]?\s*$",
                left_source,
                re.IGNORECASE,
            ) and re.match(r"^一.{1,16}就", right_translation):
                cleaned_left = re.sub(
                    r"[\s，,]*(?:它|这|那|这辆车|那辆车)"
                    r"(?:其实|基本|大概|完全|确实)?就是$",
                    "",
                    left_translation,
                ).rstrip()
                if cleaned_left and cleaned_left != left_translation:
                    repaired_by_index[left.index] = replace(
                        repaired_by_index[left.index],
                        translated_text=cleaned_left,
                    )
                    changed = True

            # Relocate a conjunction that the English word order stranded at
            # the previous cue end. This preserves the same relation and text
            # ownership while making both Chinese cues independently readable.
            connector_match = re.search(
                r"[\s，,]*(但|但是|不过|而且|并且|所以|因此|另一方面|与此同时|相比之下|相较之下)"
                r"[\s，。！？；：、,.!?;:…]*$",
                repaired_by_index[left.index].translated_text,
            )
            if connector_match and re.search(
                r"\b(?:and(?:\s+then)?|but|so|however|meanwhile|on\s+the\s+other\s+hand|"
                r"by\s+contrast|in\s+contrast)[,.!?]?\s*$",
                left_source,
                re.IGNORECASE,
            ):
                connector = connector_match.group(1)
                cleaned_left = (
                    repaired_by_index[left.index]
                    .translated_text[: connector_match.start()]
                    .rstrip()
                )
                current_right = repaired_by_index[right.index].translated_text.lstrip()
                if connector in {"而且", "并且"} and re.match(
                    r"^(?:(?:也|还|又)|[\u3400-\u9fffA-Za-z0-9]{1,8}(?:也|还|又))",
                    current_right,
                ):
                    repaired_right = current_right
                elif connector in {"但", "但是", "不过"} and re.match(
                    r"^(?:好吧|行吧)", current_right
                ):
                    repaired_right = re.sub(r"^(?:好吧|行吧)[\s，,]*", "不过 ", current_right)
                elif re.match(
                    r"^(?:但|但是|不过|然而|而且|并且|所以|因此|"
                    r"另一方面|与此同时|相比之下|相较之下)",
                    current_right,
                ):
                    repaired_right = current_right
                else:
                    repaired_right = f"{connector} {current_right}"
                if cleaned_left and repaired_right:
                    repaired_by_index[left.index] = replace(
                        repaired_by_index[left.index],
                        translated_text=cleaned_left,
                    )
                    repaired_by_index[right.index] = replace(
                        repaired_by_index[right.index],
                        translated_text=repaired_right,
                    )
                    changed = True

            if re.search(r"\bnewly\s+revised$", left_source, re.IGNORECASE) and re.match(
                r"^(?:\w+\s+){0,3}(?:speaker|JBL|Bose|audio|sound)\b",
                right_source,
                re.IGNORECASE,
            ):
                left_translation = repaired_by_index[left.index].translated_text.rstrip()
                right_translation = repaired_by_index[right.index].translated_text.lstrip()
                cleaned_left = re.sub(
                    r"\s*(?:这个|这款|这套)?(?:新|全新)?(?:改版|改款|修订)(?:的|版本)?$",
                    "",
                    left_translation,
                    count=1,
                ).rstrip()
                if (
                    cleaned_left
                    and cleaned_left != left_translation
                    and not re.match(r"^(?:这套)?(?:新|全新)?改版", right_translation)
                ):
                    repaired_by_index[left.index] = replace(
                        repaired_by_index[left.index],
                        translated_text=cleaned_left,
                    )
                    repaired_by_index[right.index] = replace(
                        repaired_by_index[right.index],
                        translated_text=f"这套新改版的{right_translation}",
                    )
                    changed = True
                    continue
            ordinal_gear = re.search(
                r"\b(first|second|third|fourth|fifth|sixth)$",
                left_source,
                flags=re.IGNORECASE,
            )
            if ordinal_gear and re.match(r"^gear\b", right_source, re.IGNORECASE):
                gear_names = {
                    "first": "一挡",
                    "second": "二挡",
                    "third": "三挡",
                    "fourth": "四挡",
                    "fifth": "五挡",
                    "sixth": "六挡",
                }
                gear_name = gear_names[ordinal_gear.group(1).lower()]
                left_translation = repaired_by_index[left.index].translated_text
                right_translation = repaired_by_index[right.index].translated_text
                if gear_name not in left_translation:
                    continue
                cleaned_right = re.sub(
                    r"^\s*(?:(?:挂|切|换)(?:入|上|到)?\s*)?"
                    rf"(?:{gear_name[0]})?挡(?:位)?(?:时)?\s*[,，、]?\s*",
                    "",
                    right_translation,
                    count=1,
                )
                if cleaned_right and cleaned_right != right_translation:
                    repaired_by_index[left.index] = replace(
                        repaired_by_index[left.index],
                        translated_text=left_translation,
                    )
                    repaired_by_index[right.index] = replace(
                        repaired_by_index[right.index],
                        translated_text=cleaned_right,
                    )
                    changed = True
                    continue
        if not changed:
            return None
        return [repaired_by_index[item.index] for item in window]

    def _chinese_fluency_candidates(
        self,
        source_list: List[SubtitleProcessData],
        translated_by_index: Dict[int, SubtitleProcessData],
    ) -> list[int]:
        """Return boundaries that need a cheap, conservative readability audit.

        A single-speaker passage can reorder grammar across any source clause that
        continues into the next cue. Auditing every such open boundary is more
        general than accumulating English-prefix regexes, while the independent
        auditor still prevents an open boundary from becoming an automatic rewrite.
        Dialogue remains risk-gated because natural turn-taking contains many short
        fragments and a speaker change is normally a hard semantic boundary.
        """
        candidates: list[int] = []
        multispeaker = self._is_multispeaker_document()
        for position in range(len(source_list) - 1):
            left_source = source_list[position]
            right_source = source_list[position + 1]
            left_item = translated_by_index.get(left_source.index)
            right_item = translated_by_index.get(right_source.index)
            if left_item is None or right_item is None:
                continue
            left_speaker = self._all_speaker_by_index.get(left_source.index, "")
            right_speaker = self._all_speaker_by_index.get(right_source.index, "")
            speaker_changed = bool(left_speaker and right_speaker and left_speaker != right_speaker)
            same_speaker = bool(left_speaker and left_speaker == right_speaker)
            edited_handoff = self._is_edited_speaker_handoff(left_source, right_source)
            if speaker_changed and not edited_handoff:
                continue
            target_signal = self._target_boundary_signal(
                left_source.index,
                left_item.translated_text,
                right_item.translated_text,
            )
            source_signal = self._source_boundary_signal(
                left_source.original_text,
                right_source.original_text,
                left_item.translated_text,
                right_item.translated_text,
            )
            source_clause_is_open = self._is_open_source_boundary(
                left_source.original_text,
                right_source.original_text,
            )
            readable_reported_topic = self._is_readable_reported_topic_handoff(
                left_source.original_text,
                right_source.original_text,
                left_item.translated_text,
                right_item.translated_text,
            )
            if readable_reported_topic:
                should_audit = False
            elif not multispeaker or same_speaker:
                should_audit = bool(target_signal or source_signal or source_clause_is_open)
            else:
                assessment = assess_english_boundary(
                    left_source.original_text,
                    right_source.original_text,
                )
                # Dialogue contains many short, natural continuations. Audit
                # only a visible Chinese defect, a strongly dependent source
                # boundary, or a tightly edited cross-speaker handoff.
                should_audit = bool(
                    # A visible Chinese boundary signal is cheap to classify
                    # and is not itself permission to rewrite. Send soft
                    # signals through the non-reasoning auditor as well so
                    # dialogue fragments do not bypass quality control merely
                    # because the English boundary scorer considers them valid.
                    target_signal or assessment.risk >= 30 or edited_handoff
                )
            if should_audit:
                candidates.append(left_source.index)
        return candidates

    @staticmethod
    def _is_open_source_boundary(left_source: str, right_source: str) -> bool:
        """Return whether two cues continue the same unclosed source clause.

        This is language-structural rather than vocabulary-specific. Terminal
        sentence punctuation closes the clause; commas, dashes, colons, and
        absent ASR punctuation leave it open for the conservative LLM audit.
        """
        left = str(left_source or "").strip()
        right = str(right_source or "").strip()
        if not left or not right:
            return False
        if re.search(r"[.!?][\"')\]]*$", left):
            return False
        assessment = assess_english_boundary(left, right)
        if assessment.unstable:
            return True
        # A capitalized acronym can still be the head noun of a modifier that
        # ends the previous cue (for example ``fully commercial land-based / SMR``).
        # Do not let the generic capital-letter shortcut split that phrase.
        if re.search(r"\b[A-Za-z]+-[A-Za-z]+[\"')\]]*$", left) and re.match(
            r"^[A-Z][A-Z0-9-]{1,9}\b",
            right,
        ):
            return True
        # ASR punctuation can be absent even when a complete clause is followed
        # by an unmistakable new sentence. Do not grow a repair window across
        # that boundary merely because the period was omitted.
        if right[:1].isupper() and re.search(
            r"\b(?:am|are|became|become|can|could|did|does|had|has|have|is|made|"
            r"makes|played|provides?|seems?|was|were|will|would)\b",
            left,
            flags=re.IGNORECASE,
        ):
            return False
        return True

    @staticmethod
    def _is_readable_reported_topic_handoff(
        left_source: str,
        right_source: str,
        left_translation: str,
        right_translation: str,
    ) -> bool:
        """Recognize a complete reported topic followed by its shared predicate."""
        if not re.search(
            r"\b(?:note|mention|observe|report|say|show)\b.*\bthat\b.*\band\b",
            left_source,
            re.IGNORECASE,
        ):
            return False
        if not re.match(
            r"^(?:become|becomes|became|is|are|was|were|have|has|had)\b",
            right_source.strip(),
            re.IGNORECASE,
        ):
            return False
        if not re.search(
            r"(?:我.{0,8})?(?:在.{0,8})?(?:提到|指出|注意到|观察到|写到|写道|显示|发现)",
            left_translation,
        ):
            return False
        if not re.search(r"(?:和|与|以及|、)", left_translation):
            return False
        return bool(
            re.match(
                r"^(?:历来|一直|通常|往往|曾经|目前|现在|这些|上述|该|他们|她们|它们)",
                right_translation.strip(),
            )
        )

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
            if (
                left_speaker
                and right_speaker
                and left_speaker != right_speaker
                and not self._is_edited_speaker_handoff(left, right)
            ):
                continue

            long_gap_signal = self._long_gap_chinese_boundary_signal(
                left.index,
                left_item.translated_text,
                right_item.translated_text,
            )
            target_signal = long_gap_signal or self._chinese_boundary_signal(
                left_item.translated_text,
                right_item.translated_text,
            )
            reasons = set(assess_english_boundary(left.original_text, right.original_text).reasons)
            strong_dependency_reasons = {
                "numeric value separated from its unit or noun",
                "numeric value separated from its multiplier or unit",
                "numeric compound modifier separated from its head noun",
                "attributive or comparative modifier separated from its head",
                "participle separated from its complement",
            }
            strong_dependency = bool(reasons & strong_dependency_reasons)
            separated_display = (
                self._gap_after_index.get(left.index, 0) >= self.SEPARATED_DISPLAY_GAP_MS
            )
            repeated_dependency = strong_dependency and self._dependent_boundary_repeats_meaning(
                left_item.translated_text,
                right_item.translated_text,
            )
            source_is_open = not re.search(
                r"[.!?][\"')\]]*$",
                left.original_text.strip(),
            )
            if strong_dependency and (target_signal or repeated_dependency):
                mandatory.append(left.index)
            elif separated_display and long_gap_signal:
                mandatory.append(left.index)
            elif (
                separated_display
                and target_signal == "connective stranded at previous subtitle end"
            ):
                mandatory.append(left.index)
            elif (
                source_is_open
                and target_signal
                and target_signal
                not in {
                    "possible function-word split",
                    "possible demonstrative split",
                    "possible pronoun boundary",
                    "possible duplicated boundary phrase",
                    "possible duplicated boundary meaning",
                    "possible copular bridge",
                    "possible reporting frame",
                    "material subject may be stranded",
                    "coordinated modifier may be stranded",
                }
            ):
                mandatory.append(left.index)
            elif (
                target_signal == "existential predicate is separated from its object"
                and re.search(
                    r"\b(?:didn't|doesn't|don't|isn't|aren't|wasn't|weren't|"
                    r"hasn't|haven't|hadn't|won't|wouldn't|can't|couldn't|shouldn't)"
                    r"[.!?]?\s*$",
                    left.original_text,
                    re.IGNORECASE,
                )
                and re.search(
                    r"(?:没有|不是|不会|不能)[\s，。！？；：、,.!?;:…]*$",
                    left_item.translated_text,
                )
            ):
                mandatory.append(left.index)
            elif target_signal == "possible pronoun boundary" and any(
                reason.startswith("dangling subject") for reason in reasons
            ):
                mandatory.append(left.index)
            elif reasons.intersection(
                {
                    "comparative marker separated from its example",
                    "numeric range split at conjunction",
                    "place name split between city and state",
                    "proper-name subject separated from its predicate",
                    "clause-final subject separated from its finite predicate",
                    "split lexical unit 'experimental standards'",
                    "split lexical unit 'other socks'",
                    "split lexical unit 'rev matching'",
                    "split phrasal construction 'take ... away'",
                    "transitive predicate separated from its object",
                    "transitive predicate separated from its pronoun object",
                    "incomplete predicate 'have'",
                    "new-clause connective stranded at previous cue end",
                }
            ):
                mandatory.append(left.index)
            elif any(
                reason.startswith(
                    (
                        "dangling modifier 'other'",
                        "dangling modifier 'really'",
                        "dangling modifier 'revised'",
                        "dangling function word 'to'",
                        "subject and auxiliary stranded at 'it's'",
                    )
                )
                for reason in reasons
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
            r"and\s+(?:go|see|test|buy|purchase)\b|"
            r"is\b|are\b|was\b|were\b|be\b|been\b|being\b|"
            r"has\b|have\b|had\b|do\b|does\b|did\b|"
            r"can\b|could\b|will\b|would\b|should\b|may\b|might\b|must\b)",
            right_lower,
        ):
            return "source continuation may require different target-language order"

        if re.match(
            r"^so\s+(?:[a-z]+ly\b|far\b|long\b|many\b|much\b|strong\b|well\b|"
            r"widespread\b)",
            right_lower,
        ):
            return "degree complement crosses the subtitle boundary"

        if re.match(
            r"^at\s+(?:(?:much|far|considerably|significantly)\s+)?"
            r"(?:higher|lower|greater|smaller|faster|slower|stronger|weaker)\b",
            right_lower,
        ):
            return "degree complement crosses the subtitle boundary"

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
        features = ChineseBoundaryFeatures.from_text(left, right)
        left = features.left
        right = features.right
        if not left or not right:
            return ""

        for detector in ORDERED_CHINESE_BOUNDARY_DETECTORS:
            match = detector(features)
            if match is not None:
                return match.message
        return ""

    def _request_chinese_fluency_flags(
        self,
        candidate_indices: list[int],
        source_list: List[SubtitleProcessData],
        translated_by_index: Dict[int, SubtitleProcessData],
    ) -> list[int]:
        source_by_index = {item.index: item for item in source_list}
        items: Dict[str, Dict[str, Any]] = {}
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
            item: Dict[str, Any] = {
                "source_left": left.original_text,
                "source_right": right.original_text,
                "source_language_left": left.source_language,
                "source_language_right": right.source_language,
                "translation_left": left_translation.translated_text,
                "translation_right": right_translation.translated_text,
                "speaker_left": self._all_speaker_by_index.get(left.index, ""),
                "speaker_right": self._all_speaker_by_index.get(right.index, ""),
                "target_signal": self._target_boundary_signal(
                    left.index,
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
            item.update(self._display_boundary_metadata(left.index))
            items[key] = item
        if not items:
            return []

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a conservative Chinese subtitle boundary auditor. Decide whether "
                    "each boundary creates an unsuitable Chinese subtitle break. Read both source "
                    "keys as one continuous utterance. Use gap_after_ms and display_continuity to "
                    "judge how viewers actually encounter the two captions. A continuous boundary "
                    "may read as one sequence. At a separated boundary, each displayed cue must "
                    "avoid a naked number, unfinished predicate, modifier, function word, or other "
                    "fragment that depends on text appearing only after the visible pause. "
                    "Flag city/state or model-name splits, a relative or adverbial clause separated "
                    "from what it modifies, a subject separated from its predicate, a modifier from "
                    "its noun, 是 from its complement, an auxiliary or adverb from its predicate, a "
                    "coordinate noun phrase split in half, or a temporal/locative phrase placed after "
                    "its Chinese predicate. Also flag a translation that is grammatically complete "
                    "only during continuous playback but awkward when either cue is displayed alone, "
                    "or that follows source word order so literally that the pair is objectively "
                    "unnatural or assigns the wrong semantic role. Do not flag merely because a "
                    "different polished wording is possible. "
                    "For Japanese source fragments, follow Japanese particles and reconstruct a "
                    "complete Chinese topic-predicate or modifier-head relation instead of mapping "
                    "each short source fragment to an equally incomplete Chinese fragment. "
                    "When the left source owns a material noun-list subject and the right source "
                    "starts its predicate, flag the boundary if those subject identities are missing "
                    "from the left translation or if the right-key predicate was translated under "
                    "the left key. This is an ownership error, not acceptable pronoun omission. "
                    "Do not flag a natural conjunction, reason, qualification, or continuation merely "
                    "because the English sentence spans two cues. "
                    "Speaker fields are metadata only. A speaker change is normally a hard boundary. "
                    "Exception: if the source itself clearly cuts one dependent grammatical phrase "
                    "between two tightly edited voices, flag it when the two Chinese cues are not "
                    "independently readable. Do not merge or rename speakers. "
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
            cache_namespace=self.cache_namespace,
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
        """Build compact repair windows that include the full local dependency chain."""
        if not confirmed_positions:
            return []
        clusters: list[list[int]] = []
        for position in confirmed_positions:
            if clusters and position == clusters[-1][-1] + 1:
                clusters[-1].append(position)
            else:
                clusters.append([position])

        def linked(left_position: int, right_position: int) -> int:
            left = source_list[left_position]
            right = source_list[right_position]
            left_speaker = self._all_speaker_by_index.get(left.index, "")
            right_speaker = self._all_speaker_by_index.get(right.index, "")
            if left_speaker and right_speaker and left_speaker != right_speaker:
                return 0
            assessment = assess_english_boundary(left.original_text, right.original_text)
            if assessment.unstable:
                return assessment.risk
            source_signal = self._source_boundary_signal(
                left.original_text,
                right.original_text,
            )
            if source_signal:
                return 20
            if self._is_open_source_boundary(left.original_text, right.original_text):
                # Once one boundary is confirmed, include the remaining
                # same-speaker fragments so a multi-cue clause is repaired as
                # one idea instead of as a chain of isolated pairs.
                return 10
            return 0

        expanded: list[tuple[int, int]] = []
        for cluster in clusters:
            start = cluster[0]
            end = cluster[-1] + 1
            while end - start + 1 < self.CHINESE_FLUENCY_MAX_WINDOW:
                left_risk = linked(start - 1, start) if start > 0 else 0
                right_risk = linked(end, end + 1) if end + 1 < len(source_list) else 0
                if not left_risk and not right_risk:
                    break
                if right_risk >= left_risk and right_risk:
                    end += 1
                else:
                    start -= 1
            if expanded and start <= expanded[-1][1] + 1:
                previous_start, previous_end = expanded[-1]
                expanded[-1] = (previous_start, max(previous_end, end))
            else:
                expanded.append((start, end))

        windows: list[List[SubtitleProcessData]] = []
        for start, end in expanded:
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
        reasoning_override: bool | None = None,
    ) -> List[SubtitleProcessData]:
        current_by_index = {item.index: item for item in current_items}
        multispeaker = self._is_multispeaker_document()
        payload: Dict[str, Dict[str, Any]] = {}
        for item in source_items:
            canonical = self._confirmed_context_canonical(item.original_text)
            value = {
                "source": self._source_for_translation(item.original_text),
                "source_language": item.source_language,
                "current_translation": current_by_index[item.index].translated_text,
            }
            if canonical:
                value["confirmed_canonical_name"] = canonical
            if multispeaker:
                value["speaker"] = self._all_speaker_by_index.get(item.index, "")
            value.update(self._display_boundary_metadata(item.index))
            payload[str(item.index)] = value
        first_index = source_items[0].index
        last_index = source_items[-1].index
        request_payload: Dict[str, Any] = {"items": payload}
        japanese_window = any(item.source_language == "ja" for item in source_items)
        if japanese_window:
            request_payload["combined_source"] = "".join(
                self._source_for_translation(item.original_text) for item in source_items
            )
        readonly_context = {
            "previous_source": self._all_source_by_index.get(first_index - 1, ""),
            "next_source": self._all_source_by_index.get(last_index + 1, ""),
        }
        if any(readonly_context.values()):
            request_payload["readonly_context"] = readonly_context
        if multispeaker:
            request_payload["readonly_speakers"] = {
                "previous": self._all_speaker_by_index.get(first_index - 1, ""),
                "next": self._all_speaker_by_index.get(last_index + 1, ""),
            }
        retry_instruction = (
            "\nThe previous repair was rejected: "
            + feedback
            + ". Correct that exact failure while preserving all required keys."
            if feedback
            else ""
        )
        mode_guidance = repair_mode_guidance(multispeaker)
        japanese_window_guidance = (
            "For this Japanese window, combined_source is the reconstructed utterance and item "
            "keys are timing slots rather than semantic ownership boundaries. Translate the full "
            "utterance first, then repartition natural Chinese across every key. "
            if japanese_window
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"""You are repairing a confirmed Chinese subtitle syntax break for {self.target_language.value}.
Rewrite only the provided translations. Keep every key, timestamp boundary, fact, name, number, negation, comparison, and conclusion. Preserve the combined meaning exactly once. Return only {{"translations": {{"key": "text"}}}} with every input key exactly once. readonly_context is read-only and must never be copied into the output. Before answering, map each source clause to its owning key, reconstruct the complete local idea, write concise idiomatic Chinese, and verify that every fact appears once. Fidelity outranks clarity, and clarity outranks elegance. Never output reasoning, labels, notes, placeholders, source text, or punctuation-only entries. """
                    "When an item has confirmed_canonical_name, reproduce that exact Latin string "
                    "in the same key; never translate, transliterate, abbreviate, or respell it. "
                    "gap_after_ms describes the pause after each item. Treat continuous items as "
                    "one display sequence. At a separated boundary, do not leave a naked number, "
                    "unfinished predicate, modifier, or function word waiting across the pause. "
                    "When source_language is Japanese, use Japanese particles and the full local "
                    "clause to produce independently readable Chinese cues; do not preserve a "
                    "fragment boundary that leaves 把, a subject, predicate, object, or locative "
                    "complement unfinished. You may move only a copula, function word, pronoun, "
                    "classifier, or other non-material grammatical scaffolding between adjacent "
                    "keys. Never move a material action or event such as starting operation, "
                    "construction, lifting, growth, approval, or completion into a key whose source "
                    "does not contain it. Exception: if two adjacent source keys demonstrably split "
                    "one predicate-complement, modifier-head, or number-unit atom, use only minimal "
                    "shared scaffolding needed to make separated displays readable; keep its facts "
                    "within that pair and state them once in the combined window. If the source "
                    "splits a modifier from its head noun, keep a concise continuation instead of "
                    "anticipating the next key's action. For a "
                    "chain shaped as 将/把 + object, locative phrase, then "
                    "action, redistribute it into independently readable cues such as 这里使用 + "
                    "object, 它们会安装在 + location, then 并逐步 + action; never preserve the stranded "
                    "chain. "
                    + japanese_window_guidance
                    + mode_guidance
                    + self._target_language_style_rules(
                        self._source_for_translation(item.original_text) for item in source_items
                    )
                    + retry_instruction
                ),
            },
            {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
        ]
        use_reasoning = (
            reasoning_override
            if reasoning_override is not None
            else self._should_reason_about_chinese_fluency_window(
                source_items,
                current_items,
                feedback=feedback,
            )
        )
        if use_reasoning:
            self._record_reasoning_metric("rewrite_requests")
        response = call_llm(
            messages=messages,
            model=self.model,
            temperature=self.TRANSLATION_TEMPERATURE,
            use_cache=self.use_cache,
            cache_namespace=self.cache_namespace,
            client=self.llm_client,
            # Spend native reasoning only on the first rewrite. Formatting retries
            # use the validator's concrete feedback and do not benefit from another
            # long chain of thought.
            reasoning_mode="enabled" if use_reasoning else "disabled",
            max_output_tokens=(self.REASONING_REWRITE_MAX_OUTPUT_TOKENS if use_reasoning else 4096),
            **({"reasoning_effort": "low"} if use_reasoning else {}),
        )
        try:
            response_text = get_response_text(response)
            if use_reasoning:
                self._record_reasoning_metric("final_answers")
        except ValueError:
            if use_reasoning:
                self._record_reasoning_metric("no_final_answers")
            raise
        result = parse_json_object(response_text).get("translations")
        expected = set(payload)
        if not isinstance(result, dict) or set(map(str, result)) != expected:
            raise ValueError("fluency repair must return every input key exactly once")
        repaired: list[SubtitleProcessData] = []
        for item in source_items:
            text = str(result[str(item.index)]).strip()
            text = self._normalize_stacked_chinese_connectives(text)
            if not text or self._looks_like_placeholder_translation(text):
                raise ValueError(f"invalid fluency repair for key {item.index}")
            repaired.append(replace(item, translated_text=text))
        return repaired

    def _should_reason_about_chinese_fluency_window(
        self,
        source_items: List[SubtitleProcessData],
        current_items: List[SubtitleProcessData],
        *,
        feedback: str = "",
    ) -> bool:
        """Reserve native reasoning for hard semantic or word-order repairs."""
        if (
            feedback
            or len(source_items) > self.CHINESE_FLUENCY_MAX_WINDOW
            or not self._prefers_native_reasoning()
        ):
            return False

        current_by_index = {item.index: item for item in current_items}
        multispeaker = self._is_multispeaker_document()
        soft_target_signals = {
            "possible function-word split",
            "possible demonstrative split",
            "possible pronoun boundary",
            "possible duplicated boundary phrase",
            "possible duplicated boundary meaning",
            "possible copular bridge",
            "possible reporting frame",
        }
        routine_source_signals = {
            "attributive or comparative modifier separated from its head",
            "comparative clause separated after 'than'",
            "coordinate phrase crosses the subtitle boundary",
            "coordinated noun phrase split at conjunction",
            "numeric value separated from its unit or noun",
            "participle separated from its complement",
            "short source fragment crosses an unfinished sentence",
        }
        for left, right in zip(source_items, source_items[1:]):
            left_translation = current_by_index.get(left.index)
            right_translation = current_by_index.get(right.index)
            if not left_translation or not right_translation:
                continue
            target_signal = self._target_boundary_signal(
                left.index,
                left_translation.translated_text,
                right_translation.translated_text,
            )
            source_signal = self._source_boundary_signal(
                left.original_text,
                right.original_text,
                left_translation.translated_text,
                right_translation.translated_text,
            )
            if multispeaker:
                assessment = assess_english_boundary(
                    left.original_text,
                    right.original_text,
                )
                if self._is_edited_speaker_handoff(left, right):
                    return True
                if target_signal and target_signal not in soft_target_signals:
                    return True
                if target_signal in {
                    "possible duplicated boundary phrase",
                    "possible duplicated boundary meaning",
                    "possible copular bridge",
                    "possible reporting frame",
                }:
                    # These are simple surface repairs. A strong source signal
                    # may coexist with them, but native reasoning adds cost
                    # without improving the deterministic diagnosis.
                    continue
                if target_signal == "possible pronoun boundary" and any(
                    reason.startswith("dangling subject") for reason in assessment.reasons
                ):
                    return True
                if set(assessment.reasons) == {"auxiliary phrase separated from its participle"}:
                    # The dependency is deterministic and the repair prompt has
                    # concrete ownership guidance; native reasoning adds latency.
                    continue
                if assessment.risk >= 34:
                    return True
                continue
            if target_signal and target_signal not in soft_target_signals:
                return True
            if source_signal and source_signal not in routine_source_signals:
                return True
        return False

    def _validate_chinese_fluency_repair(
        self,
        source_items: List[SubtitleProcessData],
        current_items: List[SubtitleProcessData],
        repaired_items: List[SubtitleProcessData],
    ) -> None:
        source_dict = {
            str(item.index): self._source_for_translation(item.original_text)
            for item in source_items
        }
        repaired_dict = {str(item.index): item.translated_text for item in repaired_items}
        valid, error = self._validate_llm_response(
            repaired_dict,
            source_dict,
            require_reflect=False,
            # A confirmed fluency window may redistribute minimal Chinese
            # wording, but it must not introduce duplicated neighbor meaning.
            # The same validator also protects condition ownership before the
            # independent whole-window fidelity audit makes the final decision.
            check_adjacent_repetition=True,
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
        for item in source_items:
            canonical = self._confirmed_context_canonical(item.original_text)
            repaired = repaired_by_index.get(item.index)
            if (
                canonical
                and repaired
                and canonical.casefold() not in repaired.translated_text.casefold()
            ):
                raise ValueError(
                    f"fluency repair dropped confirmed canonical name '{canonical}' "
                    f"from key {item.index}"
                )
        remaining = [
            item.index
            for item, following in zip(source_items, source_items[1:])
            if item.index in repaired_by_index
            and following.index in repaired_by_index
            and self._target_boundary_signal(
                item.index,
                repaired_by_index[item.index].translated_text,
                repaired_by_index[following.index].translated_text,
            )
        ]
        remaining_details = {
            index: {
                "left": repaired_by_index[index].translated_text,
                "right": repaired_by_index[index + 1].translated_text,
                "signal": self._target_boundary_signal(
                    index,
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
                "possible duplicated boundary meaning",
                "possible copular bridge",
                "possible reporting frame",
            }
        }
        if hard_remaining:
            raise ValueError(f"fluency repair left structural boundary signals: {hard_remaining}")
        for left, right in zip(source_items, source_items[1:]):
            repaired_left = repaired_by_index.get(left.index)
            repaired_right = repaired_by_index.get(right.index)
            if repaired_left is None or repaired_right is None:
                continue
            cross_language_error = self._cross_language_chinese_boundary_error(
                left.original_text,
                right.original_text,
                repaired_left.translated_text,
                repaired_right.translated_text,
            )
            if cross_language_error:
                raise ValueError(cross_language_error)
        # Soft signals only shortlist the original defect. Re-running the same
        # classifier after a rewrite creates a circular veto and rejects valid
        # Chinese redistribution. Hard structural signals above remain binding;
        # one independent whole-window fidelity verdict is the final arbiter.
        self._validate_chinese_window_fidelity(source_items, repaired_items)

    @staticmethod
    def _cross_language_chinese_boundary_error(
        left_source: str,
        right_source: str,
        left_translation: str,
        right_translation: str,
    ) -> str:
        """Reject target text that preserves a known English-only fragment boundary."""
        reasons = set(assess_english_boundary(left_source, right_source).reasons)
        if (
            "new-clause connective stranded at previous cue end" in reasons
            and re.match(
                r"^(?:different|similar|distinct)\b",
                right_source.strip(),
                re.IGNORECASE,
            )
            and re.match(r"^(?:和|与|跟|比)", right_translation.strip())
        ):
            return (
                "comparison repair still lacks an explicit Chinese subject after a "
                "stranded English discourse connective"
            )
        return ""

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
        combined_source = " ".join(
            self._source_for_translation(item.original_text) for item in source_items
        )
        combined_target = " ".join(item.translated_text for item in repaired_items)
        if re.search(
            r"\bwithout\s+effort\s+in\s+the\s+same\s+way\s+as\s+speaking\s+is\b",
            combined_source,
            re.IGNORECASE,
        ) and re.search(r"说话.{0,8}(?:需要|付出).{0,4}努力", combined_target):
            raise ValueError(
                "fluency repair reversed the effort contrast: reading and writing require "
                "continued effort, while speaking does not"
            )
        allows_edited_handoff = any(
            self._is_edited_speaker_handoff(left, right)
            for left, right in zip(source_items, source_items[1:])
        )
        payload: Dict[str, Dict[str, Any]] = {}
        for item in source_items:
            value: Dict[str, Any] = {
                "source": self._source_for_translation(item.original_text),
                "translation": repaired_by_index[item.index].translated_text,
                "speaker": self._all_speaker_by_index.get(item.index, ""),
            }
            value.update(self._display_boundary_metadata(item.index))
            payload[str(item.index)] = value
        handoff_rule = (
            " One boundary is a tightly edited speaker handoff inside a dependent source "
            "phrase. Minimal repetition or restatement of a shared noun, function words, or "
            "grammatical frame is allowed solely to make both cues readable, but no name, number, "
            "distinct fact, opinion, answer, or conclusion may move between speakers."
            if allows_edited_handoff
            else " No meaning may move across a speaker turn."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an independent bilingual subtitle-window fidelity validator. "
                    "The ordered keys are one short continuous utterance. Minimal Chinese "
                    "surface reordering between adjacent keys is allowed when required by "
                    "Chinese grammar, but every source fact, name, number, model, negation, "
                    "comparison, qualification, and conclusion must appear exactly once in "
                    "the combined translations. A material action or event such as starting "
                    "operation, construction, lifting, growth, approval, or completion must stay "
                    "under a key whose source contains that action; Chinese word order is not "
                    "permission to anticipate it. Hard facts must not move to an unrelated key, "
                    "and no meaning may be invented, omitted, duplicated, or anticipated from "
                    "outside this window. A minimal pronoun, demonstrative, classifier, or already "
                    "established head noun may be restated solely as Chinese grammatical scaffolding; "
                    "do not count it as duplicated meaning unless it repeats or changes a material "
                    "fact. gap_after_ms describes the visible pause after each key. At a separated "
                    "boundary, an adjacent predicate-complement, modifier-head, or number-unit atom "
                    "may use minimal shared scaffolding across that exact pair so both cues are "
                    "readable, provided its combined facts remain complete and occur once. Judge combined "
                    "fidelity and per-cue readability, not English word-order similarity. "
                    "If a left source key owns a material coordinated noun subject and the next "
                    "source key begins that subject's predicate, keep the subject identities visible "
                    "under the left key. Do not accept replacing them with the next key's predicate, "
                    "a generic pronoun, or an omitted subject. "
                    + handoff_rule
                    + ' Return only {"valid": true_or_false, "issues": ["brief issue"]}.'
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = call_llm(
            messages=messages,
            model=self.model,
            temperature=0,
            use_cache=self.use_cache,
            cache_namespace=self.cache_namespace,
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
        base_url = str(
            getattr(self.llm_client, "_subforge_base_url", "") or os.getenv("OPENAI_BASE_URL", "")
        ).strip()
        provider_key = generate_cache_key(base_url)[:16]
        prompt_key = generate_cache_key(
            {
                "custom_prompt": self.custom_prompt,
                "reflect": self.is_reflect,
                "context": self.translation_context.fingerprint(),
                "dialogue_speakers": {
                    data.index: self._all_speaker_by_index.get(data.index, "") for data in chunk
                },
                "prompt_version": (
                    "context-v42-terminology-boundary-review"
                    if self._is_multispeaker_document()
                    else "context-v36-terminology-boundary-review"
                ),
            }
        )
        return f"{class_name}:{chunk_key}:{lang}:{provider_key}:{model}:{prompt_key}"
