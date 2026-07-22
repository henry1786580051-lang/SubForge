import math
import re

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, field_validator

from app.security import validate_path
from subforge.core.utils.atomic_write import atomic_write_text

_ASS_TAG_RE = re.compile(r"\{[^}]*\}")

router = APIRouter()

# Format → (converter, media_type) mapping — single source of truth
_FORMAT_MAP: dict[str, tuple] = {}  # populated below after converter defs


def _export_segments(segments: list[dict], fmt: str) -> tuple[str, str]:
    """Convert segments to the requested format. Returns (output, media_type)."""
    if fmt not in _FORMAT_MAP:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")
    converter, media_type = _FORMAT_MAP[fmt]
    return converter(segments), media_type


_TIMESTAMP_RE = re.compile(
    r"^(?:(?P<hours>\d+):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2})(?:[.,](?P<fraction>\d{1,3}))?$"
)


def _timestamp_to_ms(value: object, *, strict: bool = False) -> int:
    """Parse SRT/VTT/ASS-like timestamps into milliseconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if strict and (not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid timestamp: {value!r}")
        return max(0, int(value))

    text = str(value or "").strip()
    if not text:
        if strict:
            raise ValueError("Timestamp cannot be empty")
        return 0

    match = _TIMESTAMP_RE.fullmatch(text)
    if match:
        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        if minutes >= 60 or seconds >= 60:
            if strict:
                raise ValueError(f"Invalid timestamp: {value!r}")
            return 0
        fraction = (match.group("fraction") or "").ljust(3, "0")
        milliseconds = int(fraction or 0)
        return (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds

    parts = text.replace(",", ".").split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
        elif len(parts) == 2:
            hours = 0
            minutes = int(parts[0])
            seconds = float(parts[1])
        else:
            hours = 0
            minutes = 0
            seconds = float(parts[0])
    except ValueError:
        if strict:
            raise ValueError(f"Invalid timestamp: {value!r}") from None
        return 0


    if strict:
        raise ValueError(f"Invalid timestamp: {value!r}")

    return max(0, int(round((hours * 3600 + minutes * 60 + seconds) * 1000)))


def _ms_to_timestamp(ms: int) -> str:
    ms = max(0, int(ms))
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _normalize_segment_timing(
    segments: list[dict],
    min_duration_ms: int = 1,
    *,
    strict_timestamps: bool = False,
) -> list[dict]:
    """Return copied segments with sorted, non-overlapping timestamps."""
    normalized = []
    for order, seg in enumerate(segments):
        item = dict(seg)
        start_ms = _timestamp_to_ms(item.get("start"), strict=strict_timestamps)
        end_ms = _timestamp_to_ms(item.get("end"), strict=strict_timestamps)
        if end_ms < start_ms:
            end_ms = start_ms
        item["_order"] = order
        item["_start_ms"] = start_ms
        item["_end_ms"] = end_ms
        normalized.append(item)

    normalized.sort(key=lambda s: (s["_start_ms"], s["_end_ms"], s["_order"]))
    min_duration_ms = max(0, min_duration_ms)

    for i in range(1, len(normalized)):
        prev = normalized[i - 1]
        curr = normalized[i]
        if prev["_end_ms"] <= curr["_start_ms"]:
            continue

        prev_min_end = prev["_start_ms"] + min_duration_ms
        curr_max_start = curr["_end_ms"] - min_duration_ms
        if prev_min_end <= curr_max_start:
            split = (prev["_end_ms"] + curr["_start_ms"]) // 2
            split = max(prev_min_end, min(split, curr_max_start))
        else:
            split = max(prev["_start_ms"], min(curr["_start_ms"], curr["_end_ms"]))

        prev["_end_ms"] = min(prev["_end_ms"], split)
        curr["_start_ms"] = max(curr["_start_ms"], split)
        if curr["_end_ms"] < curr["_start_ms"]:
            curr["_end_ms"] = curr["_start_ms"]

    result = []
    for item in normalized:
        item["start"] = _ms_to_timestamp(item["_start_ms"])
        item["end"] = _ms_to_timestamp(item["_end_ms"])
        item.pop("_order", None)
        item.pop("_start_ms", None)
        item.pop("_end_ms", None)
        result.append(item)
    return result


def _segment_display_text(seg: dict) -> str:
    """Return bilingual display text using the app default: target above source."""
    text = str(seg.get("text", "") or "")
    translated = str(seg.get("translated", "") or "")
    if not translated:
        return text
    if not text:
        return translated
    return f"{translated}\n{text}"


@router.get("/load")
async def load_subtitle(path: str = Query(..., description="Subtitle file path")):
    """Load and parse a subtitle file (SRT/ASS/VTT)."""
    try:
        file_path = validate_path(path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    suffix = file_path.suffix.lower()

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="gbk")

    if suffix == ".srt":
        segments = parse_srt(content)
    elif suffix == ".vtt":
        segments = parse_vtt(content)
    elif suffix == ".ass":
        segments = parse_ass(content)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    segments = _normalize_segment_timing(segments)
    return {
        "file_path": str(file_path),
        "format": suffix.lstrip("."),
        "segments": segments,
        "count": len(segments),
    }


@router.get("/export")
async def export_subtitle(
    path: str = Query(...),
    format: str = Query("srt", pattern="^(srt|vtt|ass|txt|json)$"),
    mode: str = Query("bilingual", pattern="^(original|translated|bilingual)$"),
):
    """Export subtitles in the specified format and language mode."""
    try:
        file_path = validate_path(path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    src_suffix = file_path.suffix.lower().lstrip(".")

    # Convert using ASRData
    try:
        from subforge.core.asr.asr_data import ASRData

        asr_data = ASRData.from_subtitle_file(str(file_path))

        def _ms_to_srt(ms: int) -> str:
            """Convert milliseconds to SRT time format (HH:MM:SS.mmm)."""
            if ms < 0:
                ms = 0
            s, ms = divmod(ms, 1000)
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

        segments = [{"id": i + 1,
                      "start": _ms_to_srt(seg.start_time) if isinstance(seg.start_time, int) else str(seg.start_time),
                      "end": _ms_to_srt(seg.end_time) if isinstance(seg.end_time, int) else str(seg.end_time),
                      "text": seg.text, "translated": getattr(seg, "translated_text", "")}
                     for i, seg in enumerate(asr_data.segments)]
    except ImportError:
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="gbk")

        if src_suffix == "srt":
            segments = parse_srt(content)
        elif src_suffix == "vtt":
            segments = parse_vtt(content)
        elif src_suffix == "ass":
            segments = parse_ass(content)
        else:
            raise HTTPException(status_code=400, detail=f"Cannot read {src_suffix}")

    segments = _normalize_segment_timing(segments)
    # Apply language mode to segments
    segments = _apply_language_mode(segments, mode)

    # Generate output
    output, media_type = _export_segments(segments, format)
    filename = file_path.stem + f".{format}"
    return Response(
        content=output.encode("utf-8"),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _apply_language_mode(segments: list[dict], mode: str) -> list[dict]:
    """Adjust segment text based on language mode."""
    if mode == "original":
        return [{**s, "translated": ""} for s in segments]
    elif mode == "translated":
        result = []
        for s in segments:
            if s.get("translated"):
                result.append({**s, "text": s["translated"], "translated": ""})
            else:
                result.append(s)
        return result
    # bilingual: keep as-is
    return segments


class SegmentData(BaseModel):
    id: int = 0
    start: str = "00:00:00.000"
    end: str = "00:00:00.000"
    text: str = ""
    translated: str = ""


class ExportRequest(BaseModel):
    segments: list[SegmentData]
    format: str = "srt"
    mode: str = "bilingual"
    filename: str = "export.srt"

    @field_validator("filename")
    @classmethod
    def sanitize_filename(cls, v: str) -> str:
        v = re.sub(r'[/\\:*?"<>|\x00-\x1f]', '_', v)
        return v or "export.srt"


@router.post("/export")
async def export_subtitle_post(req: ExportRequest):
    """Export subtitles from POST data (for pywebview/DMG where GET download doesn't work)."""
    segments = [s.model_dump() for s in req.segments]
    try:
        segments = _normalize_segment_timing(segments, strict_timestamps=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    segments = _apply_language_mode(segments, req.mode)
    output, media_type = _export_segments(segments, req.format)
    return Response(
        content=output.encode("utf-8"),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{req.filename}"'},
    )


class SaveRequest(BaseModel):
    file_path: str
    segments: list[dict]


@router.post("/save")
async def save_subtitle(req: SaveRequest):
    """Save edited subtitle segments back to file."""
    try:
        file_path = validate_path(req.file_path)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Original file not found")

    suffix = file_path.suffix.lower()
    try:
        segments = _normalize_segment_timing(req.segments, strict_timestamps=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if suffix == ".srt":
        content = segments_to_srt(segments)
    elif suffix == ".vtt":
        content = segments_to_vtt(segments)
    elif suffix == ".ass":
        content = segments_to_ass(segments)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    atomic_write_text(file_path, content)
    return {"status": "ok", "file_path": str(file_path), "count": len(segments)}


def segments_to_srt(segments: list[dict]) -> str:
    """Convert segments to SRT format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", "00:00:00.000").replace(".", ",")
        end = seg.get("end", "00:00:00.000").replace(".", ",")
        display = _segment_display_text(seg)
        lines.append(f"{i}\n{start} --> {end}\n{display}")
    return "\n\n".join(lines) + "\n"


def segments_to_vtt(segments: list[dict]) -> str:
    """Convert segments to WebVTT format."""
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", "00:00:00.000")
        end = seg.get("end", "00:00:00.000")
        display = _segment_display_text(seg)
        lines.append(f"{i}\n{start} --> {end}\n{display}")
    return "\n\n".join(lines) + "\n"


def segments_to_ass(segments: list[dict]) -> str:
    """Convert segments to ASS format."""
    header = """[Script Info]
Title: SubForge Export
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,18,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for seg in segments:
        start = _time_to_ass(seg.get("start", "00:00:00.000"))
        end = _time_to_ass(seg.get("end", "00:00:00.000"))
        display = _segment_display_text(seg).replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{display}")
    return "\n".join(lines) + "\n"


def _time_to_ass(time_str: str) -> str:
    """Convert SRT time (00:00:00,000) to ASS time (0:00:00.00)."""
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return f"{int(h)}:{m}:{s[:5]}"
    return time_str


def segments_to_txt(segments: list[dict]) -> str:
    """Convert segments to plain text."""
    lines = []
    for seg in segments:
        lines.append(_segment_display_text(seg))
    return "\n".join(lines) + "\n"


def segments_to_json(segments: list[dict]) -> str:
    """Convert segments to JSON format."""
    import json
    data = []
    for seg in segments:
        data.append({
            "start": seg.get("start", "00:00:00.000"),
            "end": seg.get("end", "00:00:00.000"),
            "text": seg.get("text", ""),
            "translated": seg.get("translated", ""),
        })
    return json.dumps(data, ensure_ascii=False, indent=2)


def parse_srt(content: str) -> list[dict]:
    """Parse SRT format into segments."""
    from subforge.core.asr.asr_data import ASRData

    asr_data = ASRData.from_srt(content)
    return [
        {
            "id": idx,
            "start": _ms_to_timestamp(seg.start_time),
            "end": _ms_to_timestamp(seg.end_time),
            "text": seg.text,
            "translated": seg.translated_text,
        }
        for idx, seg in enumerate(asr_data.segments, 1)
    ]


def parse_vtt(content: str) -> list[dict]:
    """Parse WebVTT format into segments."""
    segments = []
    lines = content.strip().split("\n")
    idx = 0
    i = 0

    # Skip WEBVTT header (skip until first --> line)
    while i < len(lines) and " --> " not in lines[i]:
        i += 1

    while i < len(lines):
        line = lines[i].strip()

        if " --> " in line:
            idx += 1
            start, end = line.split(" --> ")
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and " --> " not in lines[i]:
                text_lines.append(lines[i].strip())
                i += 1

            segments.append({
                "id": idx,
                "start": start.strip(),
                "end": end.strip().split(" ")[0],
                "text": "\n".join(text_lines),
                "translated": "",
            })
        else:
            i += 1

    return segments


def parse_ass(content: str) -> list[dict]:
    """Parse ASS format into segments."""
    segments = []
    idx = 0
    in_events = False
    format_fields = []

    for line in content.split("\n"):
        line = line.strip()

        if line == "[Events]":
            in_events = True
            continue

        if line.startswith("[") and in_events:
            break

        if in_events and line.startswith("Format:"):
            format_fields = [f.strip() for f in line[7:].split(",")]
            continue

        if in_events and line.startswith("Dialogue:"):
            parts = line[9:].strip().split(",", len(format_fields) - 1)
            if len(parts) >= len(format_fields):
                start_idx = format_fields.index("Start") if "Start" in format_fields else 1
                end_idx = format_fields.index("End") if "End" in format_fields else 2
                text_idx = format_fields.index("Text") if "Text" in format_fields else -1

                idx += 1
                text = parts[text_idx] if text_idx >= 0 else ""
                text = _ASS_TAG_RE.sub("", text).replace("\\N", "\n")

                segments.append({
                    "id": idx,
                    "start": parts[start_idx],
                    "end": parts[end_idx],
                    "text": text,
                    "translated": "",
                })

    return segments


# Populate format map after all converters are defined
_FORMAT_MAP.update({
    "srt": (segments_to_srt, "application/x-subrip"),
    "vtt": (segments_to_vtt, "text/vtt"),
    "ass": (segments_to_ass, "text/x-ssa"),
    "txt": (segments_to_txt, "text/plain"),
    "json": (segments_to_json, "application/json"),
})
