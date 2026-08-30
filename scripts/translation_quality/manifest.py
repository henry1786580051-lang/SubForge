"""Corpus manifest discovery and validation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from .srt import parse_srt

CorpusSplit = Literal["development", "validation", "holdout"]


class ManifestError(ValueError):
    """Raised when a corpus manifest is incomplete or unsafe."""


@dataclass(frozen=True)
class CorpusSample:
    id: str
    title: str
    split: CorpusSplit
    domain: str
    speaker_mode: str
    speaker_count: int | None
    source_languages: tuple[str, ...]
    target_language: str
    duration_ms: int | None
    source_srt: str
    machine_srt: str
    gold_srt: str
    source_media: str | None
    machine_model: str
    algorithm_version: str
    configuration: dict[str, Any]
    provenance: dict[str, Any]
    alignment: dict[str, Any]
    known_issues: tuple[str, ...]
    notes: str
    hashes: dict[str, str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorpusSample":
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            split=str(data["split"]),  # type: ignore[arg-type]
            domain=str(data.get("domain", "unknown")),
            speaker_mode=str(data.get("speaker_mode", "unknown")),
            speaker_count=data.get("speaker_count"),
            source_languages=tuple(str(item) for item in data.get("source_languages", [])),
            target_language=str(data.get("target_language", "zh-CN")),
            duration_ms=data.get("duration_ms"),
            source_srt=str(data["source_srt"]),
            machine_srt=str(data["machine_srt"]),
            gold_srt=str(data["gold_srt"]),
            source_media=str(data["source_media"]) if data.get("source_media") else None,
            machine_model=str(data.get("machine_model", "unknown")),
            algorithm_version=str(data.get("algorithm_version", "unknown")),
            configuration=dict(data.get("configuration", {})),
            provenance=dict(data.get("provenance", {})),
            alignment=dict(data.get("alignment", {})),
            known_issues=tuple(str(item) for item in data.get("known_issues", [])),
            notes=str(data.get("notes", "")),
            hashes={str(key): str(value) for key, value in data.get("hashes", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "split": self.split,
            "domain": self.domain,
            "speaker_mode": self.speaker_mode,
            "speaker_count": self.speaker_count,
            "source_languages": list(self.source_languages),
            "target_language": self.target_language,
            "duration_ms": self.duration_ms,
            "source_srt": self.source_srt,
            "machine_srt": self.machine_srt,
            "gold_srt": self.gold_srt,
            "source_media": self.source_media,
            "machine_model": self.machine_model,
            "algorithm_version": self.algorithm_version,
            "configuration": self.configuration,
            "provenance": self.provenance,
            "alignment": self.alignment,
            "known_issues": list(self.known_issues),
            "notes": self.notes,
            "hashes": self.hashes,
        }


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: int
    corpus_id: str
    created_at: str
    data_root_env: str
    samples: tuple[CorpusSample, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorpusManifest":
        return cls(
            schema_version=int(data["schema_version"]),
            corpus_id=str(data["corpus_id"]),
            created_at=str(data["created_at"]),
            data_root_env=str(data.get("data_root_env", "SUBFORGE_TRANSLATION_CORPUS_ROOT")),
            samples=tuple(CorpusSample.from_dict(item) for item in data.get("samples", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "corpus_id": self.corpus_id,
            "created_at": self.created_at,
            "data_root_env": self.data_root_env,
            "samples": [sample.to_dict() for sample in self.samples],
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_under_root(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ManifestError(f"Corpus path escapes data root: {relative_path}") from exc
    return candidate


def load_manifest(path: Path) -> CorpusManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManifestError("Manifest root must be a JSON object")
    return CorpusManifest.from_dict(data)


def write_manifest(manifest: CorpusManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_manifest(manifest: CorpusManifest, root: Path, *, verify_hashes: bool = True) -> None:
    errors: list[str] = []
    if manifest.schema_version != 1:
        errors.append(f"unsupported schema_version={manifest.schema_version}")
    if not manifest.samples:
        errors.append("manifest contains no samples")

    seen_ids: set[str] = set()
    valid_splits = {"development", "validation", "holdout"}
    for sample in manifest.samples:
        if sample.id in seen_ids:
            errors.append(f"duplicate sample id: {sample.id}")
        seen_ids.add(sample.id)
        if sample.split not in valid_splits:
            errors.append(f"{sample.id}: invalid split {sample.split}")
        if not sample.source_languages:
            errors.append(f"{sample.id}: source_languages is empty")

        paths = {
            "source_sha256": sample.source_srt,
            "machine_sha256": sample.machine_srt,
            "gold_sha256": sample.gold_srt,
        }
        for hash_key, relative in paths.items():
            if Path(relative).is_absolute():
                errors.append(f"{sample.id}: absolute corpus path is not allowed: {relative}")
                continue
            try:
                resolved = _resolve_under_root(root, relative)
            except ManifestError as exc:
                errors.append(f"{sample.id}: {exc}")
                continue
            if not resolved.is_file():
                errors.append(f"{sample.id}: missing file {relative}")
                continue
            expected = sample.hashes.get(hash_key)
            if verify_hashes and not expected:
                errors.append(f"{sample.id}: missing {hash_key}")
            elif verify_hashes and expected != sha256_file(resolved):
                errors.append(f"{sample.id}: hash mismatch for {relative}")

        if sample.source_media:
            if Path(sample.source_media).is_absolute():
                errors.append(f"{sample.id}: absolute media path is not allowed")
            else:
                try:
                    media_path = _resolve_under_root(root, sample.source_media)
                except ManifestError as exc:
                    errors.append(f"{sample.id}: {exc}")
                else:
                    if not media_path.is_file():
                        errors.append(f"{sample.id}: missing media {sample.source_media}")

    if errors:
        raise ManifestError("Manifest validation failed:\n- " + "\n- ".join(errors))


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-") or hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _choose_triplet(files: Iterable[Path]) -> tuple[Path, Path, Path]:
    candidates = sorted(files)
    # Both legacy *_chatgpt_edited and model-qualified *_<model>_edited exports.
    gold = [path for path in candidates if path.stem.casefold().endswith("_edited")]
    if len(gold) != 1:
        raise ManifestError(f"Expected one *_edited gold file, found {len(gold)}")
    remaining = [path for path in candidates if path != gold[0]]
    machine = [path for path in remaining if "processed" in path.name.casefold()]
    if len(machine) != 1:
        raise ManifestError(f"Expected one processed machine file, found {len(machine)}")
    source = [path for path in remaining if path != machine[0]]
    if len(source) != 1:
        raise ManifestError(f"Expected one source file, found {len(source)}")
    return source[0], machine[0], gold[0]


def _detect_languages(text: str) -> tuple[str, ...]:
    languages: list[str] = []
    if re.search(r"[A-Za-z]{2}", text):
        languages.append("en")
    if re.search(r"[\u3040-\u30ff\u31f0-\u31ff]", text):
        languages.append("ja")
    if re.search(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]", text):
        languages.append("ko")
    if re.search(r"[\u0400-\u04ff]", text):
        languages.append("ru")
    return tuple(languages or ["unknown"])


def _detect_speakers(text: str) -> tuple[str, int | None]:
    speaker_ids = {
        int(match)
        for match in re.findall(r"\[(?:Speaker|说话人)[ _]?(\d+)\]", text, re.IGNORECASE)
    }
    if not speaker_ids:
        return "unknown", None
    count = len(speaker_ids)
    if count == 1:
        return "monologue", count
    if count == 2:
        return "dialogue", count
    return "multi_speaker", count


def _infer_domain(title: str) -> str:
    folded = title.casefold()
    if any(token in folded for token in ("honda", "toyota", "bentley", "car", "driving")):
        return "automotive"
    if any(
        token in folded
        for token in (
            "airport",
            "canal",
            "tunnel",
            "tower",
            "building",
            "sea wall",
            "skytrain",
            "nile",
            "sphere",
        )
    ):
        return "infrastructure"
    return "other"


def _infer_model(filename: str) -> str:
    folded = filename.casefold()
    if "v4flash" in folded or "v4-flash" in folded:
        return "deepseek-v4-flash"
    if "nemotron3" in folded or "nemotron-3" in folded:
        return "nvidia/nemotron-3-ultra-550b-a55b"
    return "unknown"


def discover_corpus(
    root: Path,
    *,
    created_at: str,
    holdout_ids: set[str] | None = None,
    validation_ids: set[str] | None = None,
) -> CorpusManifest:
    root = root.expanduser().resolve()
    holdout_ids = holdout_ids or set()
    validation_ids = validation_ids or set()
    samples: list[CorpusSample] = []

    for triplet_dir in sorted(path for path in root.rglob("三件套") if path.is_dir()):
        source_path, machine_path, gold_path = _choose_triplet(triplet_dir.glob("*.srt"))
        title = triplet_dir.parent.name
        sample_id = _slugify(title)
        split: CorpusSplit = "development"
        if sample_id in holdout_ids:
            split = "holdout"
        elif sample_id in validation_ids:
            split = "validation"

        source_doc = parse_srt(source_path, layout="source_only")
        machine_doc = parse_srt(machine_path, layout="target_above")
        gold_doc = parse_srt(gold_path, layout="target_above")
        source_text = "\n".join(cue.source for cue in source_doc.cues)
        speaker_mode, speaker_count = _detect_speakers(source_text)
        duration_ms = max((cue.end_ms for cue in source_doc.cues), default=None)
        machine_signature = [(cue.index, cue.timeline, cue.source) for cue in machine_doc.cues]
        gold_signature = [(cue.index, cue.timeline, cue.source) for cue in gold_doc.cues]
        cue_structure = "exact" if machine_signature == gold_signature else "requires_alignment"
        source_kind = (
            "word_level_asr"
            if len(source_doc.cues) > max(1, len(machine_doc.cues)) * 1.5
            else "sentence_srt"
        )
        media_candidates = sorted(
            path
            for path in triplet_dir.parent.iterdir()
            if path.is_file() and path.suffix.casefold() in {".mp4", ".mkv", ".mov", ".avi"}
        )
        source_media = (
            media_candidates[0].relative_to(root).as_posix() if media_candidates else None
        )
        def relative(path: Path) -> str:
            return path.relative_to(root).as_posix()

        samples.append(
            CorpusSample(
                id=sample_id,
                title=title,
                split=split,
                domain=_infer_domain(title),
                speaker_mode=speaker_mode,
                speaker_count=speaker_count,
                source_languages=_detect_languages(source_text),
                target_language="zh-CN",
                duration_ms=duration_ms,
                source_srt=relative(source_path),
                machine_srt=relative(machine_path),
                gold_srt=relative(gold_path),
                source_media=source_media,
                machine_model=_infer_model(machine_path.name),
                algorithm_version="unknown",
                configuration={
                    "batch_size": None,
                    "concurrency": None,
                    "reflection": None,
                    "multi_speaker": None,
                },
                provenance={
                    "source_kind": source_kind,
                    "machine_kind": "subforge",
                    "gold_editor": (
                        "chatgpt_pro" if "chatgpt" in gold_path.name.casefold() else "unknown"
                    ),
                    "verified_by_user": None,
                },
                alignment={
                    "cue_structure": cue_structure,
                    "timing_changed": None,
                    "advertisements_removed": None,
                    "removed_ranges": [],
                },
                known_issues=(),
                notes="Metadata not inferable from files remains unknown until Phase 0 review.",
                hashes={
                    "source_sha256": sha256_file(source_path),
                    "machine_sha256": sha256_file(machine_path),
                    "gold_sha256": sha256_file(gold_path),
                },
            )
        )

    return CorpusManifest(
        schema_version=1,
        corpus_id="subforge-local-translation-quality",
        created_at=created_at,
        data_root_env="SUBFORGE_TRANSLATION_CORPUS_ROOT",
        samples=tuple(samples),
    )
