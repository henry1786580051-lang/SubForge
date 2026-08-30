"""Pure, ordered detectors for Chinese subtitle boundaries."""

from collections.abc import Callable

from subforge.core.translate.quality.boundary_features import ChineseBoundaryFeatures

from .adverb_pronoun_attachment import detect_adverb_pronoun_attachment_boundary
from .clause_attachment import detect_clause_attachment_boundary
from .completion_frames import detect_completion_frame_boundary
from .consequence_predicate import detect_consequence_predicate_boundary
from .discourse_bridge import detect_discourse_bridge_boundary
from .foundation import detect_foundation_boundary
from .governing_attachment import detect_governing_attachment_boundary
from .incomplete_nominal_frames import detect_incomplete_nominal_frame_boundary
from .late_structural_frames import detect_late_structural_frame_boundary
from .nominal_attachment import detect_nominal_attachment_boundary
from .numeric_completion import detect_numeric_completion_boundary
from .predicate_completion import detect_predicate_completion_boundary
from .reason_construction import detect_reason_construction_boundary
from .semantic_attachment import detect_semantic_attachment_boundary
from .semantic_completion import detect_semantic_completion_boundary
from .structural_tail import detect_structural_tail_boundary
from .subject_attachment import detect_subject_attachment_boundary
from .subject_nominal_completion import detect_subject_nominal_completion_boundary
from .surface_fluency import detect_surface_fluency_boundary
from .temporal_locative_attachment import detect_temporal_locative_attachment_boundary
from .terminal_tokens import detect_terminal_token_boundary
from .unfinished_frames import detect_unfinished_frame_boundary
from .unfinished_predicate import detect_unfinished_predicate_boundary
from .visible_pause import BoundarySignalMatch, detect_visible_pause_boundary

ChineseBoundaryDetector = Callable[[ChineseBoundaryFeatures], BoundarySignalMatch | None]

ORDERED_CHINESE_BOUNDARY_DETECTORS: tuple[ChineseBoundaryDetector, ...] = (
    detect_foundation_boundary,
    detect_nominal_attachment_boundary,
    detect_governing_attachment_boundary,
    detect_surface_fluency_boundary,
    detect_subject_attachment_boundary,
    detect_discourse_bridge_boundary,
    detect_unfinished_frame_boundary,
    detect_reason_construction_boundary,
    detect_completion_frame_boundary,
    detect_numeric_completion_boundary,
    detect_consequence_predicate_boundary,
    detect_semantic_attachment_boundary,
    detect_unfinished_predicate_boundary,
    detect_predicate_completion_boundary,
    detect_temporal_locative_attachment_boundary,
    detect_incomplete_nominal_frame_boundary,
    detect_semantic_completion_boundary,
    detect_clause_attachment_boundary,
    detect_subject_nominal_completion_boundary,
    detect_structural_tail_boundary,
    detect_late_structural_frame_boundary,
    detect_adverb_pronoun_attachment_boundary,
    detect_terminal_token_boundary,
)

__all__ = [
    "BoundarySignalMatch",
    "ChineseBoundaryDetector",
    "ORDERED_CHINESE_BOUNDARY_DETECTORS",
    "detect_completion_frame_boundary",
    "detect_clause_attachment_boundary",
    "detect_adverb_pronoun_attachment_boundary",
    "detect_consequence_predicate_boundary",
    "detect_discourse_bridge_boundary",
    "detect_foundation_boundary",
    "detect_governing_attachment_boundary",
    "detect_incomplete_nominal_frame_boundary",
    "detect_late_structural_frame_boundary",
    "detect_nominal_attachment_boundary",
    "detect_numeric_completion_boundary",
    "detect_predicate_completion_boundary",
    "detect_reason_construction_boundary",
    "detect_semantic_attachment_boundary",
    "detect_semantic_completion_boundary",
    "detect_subject_attachment_boundary",
    "detect_subject_nominal_completion_boundary",
    "detect_structural_tail_boundary",
    "detect_surface_fluency_boundary",
    "detect_temporal_locative_attachment_boundary",
    "detect_terminal_token_boundary",
    "detect_unfinished_frame_boundary",
    "detect_unfinished_predicate_boundary",
    "detect_visible_pause_boundary",
]
