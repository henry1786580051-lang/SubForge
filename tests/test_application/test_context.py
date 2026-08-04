import asyncio

import pytest

from subforge.application import PipelineContext, subtitle_preview_segments
from subforge.core.asr.asr_data import ASRData, ASRDataSeg


def test_pipeline_context_forwards_progress_preview_and_cancellation_scope():
    events = []
    callbacks = []
    context = PipelineContext(
        task_id="task",
        _progress=lambda progress, message, path: events.append(
            ("progress", progress, message, path)
        ),
        _preview=lambda segments, path, message: events.append(
            ("preview", segments, path, message)
        ),
        _register_cancel=callbacks.append,
        _unregister_cancel=callbacks.remove,
    )
    def stop() -> None:
        return None

    context.report(25, "working", subtitle_file="partial.srt")
    context.publish_preview([{"id": 1}], message="ready")
    with context.cancellation_scope(stop):
        assert callbacks == [stop]
    assert callbacks == []
    assert events == [
        ("progress", 25, "working", "partial.srt"),
        ("preview", [{"id": 1}], None, "ready"),
    ]


def test_pipeline_context_checkpoint_raises_asyncio_cancellation():
    context = PipelineContext(_is_cancelled=lambda: True)

    with pytest.raises(asyncio.CancelledError):
        context.checkpoint()


def test_preview_serializer_is_shared_and_hides_placeholder_notes():
    data = ASRData(
        [
            ASRDataSeg(
                "Hello",
                0,
                1000,
                translated_text="（此句合并至上一句）",
                speaker_id="SPEAKER_00",
            )
        ]
    )

    assert subtitle_preview_segments(data) == [
        {
            "id": 1,
            "start": "00:00:00,000",
            "end": "00:00:01,000",
            "text": "Hello",
            "translated": "",
            "speaker": "SPEAKER_00",
        }
    ]
