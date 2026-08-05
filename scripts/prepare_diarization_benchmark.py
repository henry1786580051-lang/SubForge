#!/usr/bin/env python3
"""Download and validate a compact AMI/VoxConverse diarization benchmark."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subforge.core.asr.diarization_benchmark import write_rttm
from subforge.core.asr.speaker_diarization import SpeakerTurn


DATASET_SHARDS = {
    "voxconverse-dev": (
        "diarizers-community/voxconverse",
        "data/dev-00000-of-00005.parquet",
    ),
    "ami-ihm-validation": (
        "diarizers-community/ami",
        "ihm/validation-00000-of-00003.parquet",
    ),
    "ami-sdm-validation": (
        "diarizers-community/ami",
        "sdm/validation-00000-of-00003.parquet",
    ),
}


@dataclass(frozen=True)
class PreparedRecording:
    dataset: str
    source_shard: str
    source_row: int
    uri: str
    audio: str
    reference_rttm: str
    duration_seconds: float
    speakers: int
    turns: int


def _safe_uri(value: str, fallback: str) -> str:
    stem = Path(value).stem if value else fallback
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.")
    return normalized or fallback


def _audio_payload(value: Any) -> tuple[bytes, str]:
    if not isinstance(value, dict):
        raise ValueError("audio column is not an embedded audio object")
    audio_bytes = value.get("bytes")
    source_path = str(value.get("path") or "")
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise ValueError("audio row does not contain embedded bytes")
    return audio_bytes, source_path


def _validated_turns(
    starts: Any,
    ends: Any,
    speakers: Any,
    *,
    duration_seconds: float,
) -> list[SpeakerTurn]:
    if not isinstance(starts, list) or not isinstance(ends, list) or not isinstance(speakers, list):
        raise ValueError("timestamp and speaker columns must be arrays")
    if not starts or len(starts) != len(ends) or len(starts) != len(speakers):
        raise ValueError("timestamp and speaker arrays are empty or have different lengths")
    turns: list[SpeakerTurn] = []
    for index, (start, end, speaker) in enumerate(zip(starts, ends, speakers)):
        try:
            start_value = float(start)
            end_value = float(end)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid timestamp at turn {index}") from exc
        speaker_value = str(speaker or "").strip()
        if start_value < 0 or end_value <= start_value or not speaker_value:
            raise ValueError(f"invalid interval or speaker at turn {index}")
        if end_value > duration_seconds + 0.25:
            raise ValueError(
                f"turn {index} ends at {end_value:.3f}s beyond {duration_seconds:.3f}s audio"
            )
        turns.append(
            SpeakerTurn(
                round(start_value * 1000),
                round(min(end_value, duration_seconds) * 1000),
                speaker_value,
            )
        )
    return turns


def _prepare_shard(
    dataset_name: str,
    shard: Path,
    output_root: Path,
    *,
    limit: int,
) -> tuple[list[PreparedRecording], list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as parquet
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Benchmark preparation requires pyarrow and soundfile. "
            "Run with: uv run --with pyarrow --with soundfile"
        ) from exc

    accepted: list[PreparedRecording] = []
    rejected: list[dict[str, Any]] = []
    columns = ["audio", "timestamps_start", "timestamps_end", "speakers"]
    parquet_file = parquet.ParquetFile(shard)
    row_number = 0
    for batch in parquet_file.iter_batches(batch_size=1, columns=columns):
        row = batch.to_pylist()[0]
        try:
            audio_bytes, source_path = _audio_payload(row["audio"])
            audio_info = sf.info(io.BytesIO(audio_bytes))
            if audio_info.frames <= 0 or audio_info.samplerate <= 0:
                raise ValueError("audio is empty")
            duration = audio_info.frames / audio_info.samplerate
            turns = _validated_turns(
                row["timestamps_start"],
                row["timestamps_end"],
                row["speakers"],
                duration_seconds=duration,
            )
            uri = _safe_uri(source_path, f"{dataset_name}-{row_number:04d}")
            suffix = Path(source_path).suffix.lower()
            if suffix not in {".wav", ".flac", ".ogg"}:
                suffix = ".wav"
            recording_dir = output_root / dataset_name / uri
            recording_dir.mkdir(parents=True, exist_ok=True)
            audio_path = recording_dir / f"audio{suffix}"
            rttm_path = recording_dir / "reference.rttm"
            audio_path.write_bytes(audio_bytes)
            write_rttm(rttm_path, {uri: turns})
            accepted.append(
                PreparedRecording(
                    dataset=dataset_name,
                    source_shard=shard.name,
                    source_row=row_number,
                    uri=uri,
                    audio=str(audio_path),
                    reference_rttm=str(rttm_path),
                    duration_seconds=round(duration, 3),
                    speakers=len({turn.speaker_id for turn in turns}),
                    turns=len(turns),
                )
            )
        except Exception as exc:
            rejected.append({"row": row_number, "reason": str(exc)})
        row_number += 1
        if len(accepted) >= limit:
            break
    return accepted, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diarization/benchmark-data"),
    )
    parser.add_argument("--limit", type=int, default=2, help="valid recordings per dataset")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_SHARDS),
        help="dataset to prepare; repeatable, defaults to all",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required") from exc

    output_root = args.output.expanduser().resolve()
    download_root = output_root / "downloads"
    selected = args.dataset or list(DATASET_SHARDS)
    manifest: dict[str, Any] = {
        "format_version": 1,
        "selection": selected,
        "limit_per_dataset": args.limit,
        "recordings": [],
        "rejected": {},
    }
    for dataset_name in selected:
        repo_id, filename = DATASET_SHARDS[dataset_name]
        shard = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=download_root / dataset_name,
            )
        )
        accepted, rejected = _prepare_shard(
            dataset_name,
            shard,
            output_root,
            limit=args.limit,
        )
        manifest["recordings"].extend(asdict(recording) for recording in accepted)
        manifest["rejected"][dataset_name] = rejected

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    print(f"prepared={len(manifest['recordings'])}")
    print(f"rejected={sum(len(items) for items in manifest['rejected'].values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
