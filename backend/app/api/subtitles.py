import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

_ASS_TAG_RE = re.compile(r"\{[^}]*\}")

router = APIRouter()


@router.get("/load")
async def load_subtitle(path: str = Query(..., description="Subtitle file path")):
    """Load and parse a subtitle file (SRT/ASS/VTT)."""
    file_path = Path(path)
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
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    src_suffix = file_path.suffix.lower().lstrip(".")

    # Convert using ASRData
    try:
        from subforge.core.asr.asr_data import ASRData

        asr_data = ASRData.from_subtitle_file(str(file_path))
        segments = [{"id": i + 1, "start": seg.start_time, "end": seg.end_time,
                      "text": seg.text, "translated": getattr(seg, "translated", "")}
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

    # Apply language mode to segments
    segments = _apply_language_mode(segments, mode)

    # Generate output
    if format == "srt":
        output = segments_to_srt(segments)
        media_type = "application/x-subrip"
    elif format == "vtt":
        output = segments_to_vtt(segments)
        media_type = "text/vtt"
    elif format == "ass":
        output = segments_to_ass(segments)
        media_type = "text/x-ssa"
    elif format == "txt":
        output = segments_to_txt(segments)
        media_type = "text/plain"
    elif format == "json":
        output = segments_to_json(segments)
        media_type = "application/json"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

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


class SaveRequest(BaseModel):
    file_path: str
    segments: list[dict]


@router.post("/save")
async def save_subtitle(req: SaveRequest):
    """Save edited subtitle segments back to file."""
    file_path = Path(req.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Original file not found")

    suffix = file_path.suffix.lower()
    segments = req.segments

    if suffix == ".srt":
        content = segments_to_srt(segments)
    elif suffix == ".vtt":
        content = segments_to_vtt(segments)
    elif suffix == ".ass":
        content = segments_to_ass(segments)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    file_path.write_text(content, encoding="utf-8")
    return {"status": "ok", "file_path": str(file_path), "count": len(segments)}


def _srt_to_vtt(srt_content: str) -> str:
    """Convert SRT string to VTT."""
    return "WEBVTT\n\n" + srt_content.replace(",", ".")


def segments_to_srt(segments: list[dict]) -> str:
    """Convert segments to SRT format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", "00:00:00.000").replace(".", ",")
        end = seg.get("end", "00:00:00.000").replace(".", ",")
        text = seg.get("text", "")
        translated = seg.get("translated", "")
        display = text
        if translated:
            display = f"{text}\n{translated}"
        lines.append(f"{i}\n{start} --> {end}\n{display}")
    return "\n\n".join(lines) + "\n"


def segments_to_vtt(segments: list[dict]) -> str:
    """Convert segments to WebVTT format."""
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", "00:00:00.000")
        end = seg.get("end", "00:00:00.000")
        text = seg.get("text", "")
        translated = seg.get("translated", "")
        display = text
        if translated:
            display = f"{text}\n{translated}"
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
        text = seg.get("text", "")
        translated = seg.get("translated", "")
        display = text.replace("\n", "\\N")
        if translated:
            display = f"{display}\\N{translated}".replace("\n", "\\N")
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
        text = seg.get("text", "")
        translated = seg.get("translated", "")
        if translated:
            lines.append(f"{text}\n{translated}")
        else:
            lines.append(text)
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
    segments = []
    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue

        time_line = lines[1].strip()
        if " --> " not in time_line:
            continue

        start, end = time_line.split(" --> ")
        text = "\n".join(lines[2:]).strip()

        segments.append({
            "id": idx,
            "start": start.strip().replace(",", "."),
            "end": end.strip().replace(",", "."),
            "text": text,
            "translated": "",
        })

    return segments


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
