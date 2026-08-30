"""Long-video previews must share the final merge's time coordinate system."""

from threading import Event

import pytest

from subforge.core.asr.asr_data import ASRData, ASRDataSeg, ASRWord
from subforge.core.asr.chunked_asr import ChunkedASR


@pytest.mark.parametrize("concurrency", [1, 2])
def test_preview_is_cumulative_global_and_does_not_mutate_chunk_results(concurrency):
    snapshots = []
    second_done = Event()

    class FakeASR:
        def __init__(self, audio, segment_callback):
            self.audio = audio
            self.callback = segment_callback

        def run(self, callback=None):
            if concurrency == 2 and self.audio == b"first":
                assert second_done.wait(3)
            word = ASRWord(self.audio.decode(), 1000, 2000)
            data = ASRData([ASRDataSeg(word.text, 1000, 2000, words=[word])])
            self.callback(data)
            if self.audio == b"second":
                second_done.set()
            return data

    chunks = [(b"first", 0), (b"second", 1_790_000)]
    asr = ChunkedASR(
        FakeASR,
        b"unused",
        asr_kwargs={"segment_callback": snapshots.append},
        chunk_concurrency=concurrency,
    )
    results = asr._transcribe_chunks(chunks, None)
    final = asr._merge_results(results, chunks)
    expected = [("first", 1000, 2000), ("second", 1_791_000, 1_792_000)]
    assert [(s.text, s.start_time, s.end_time) for s in snapshots[-1].segments] == expected
    assert [(s.text, s.start_time, s.end_time) for s in final.segments] == expected
    assert snapshots[-1].segments[-1].words[0].start_time == 1_791_000
    assert all(data.segments[0].start_time == 1000 for data in results)
    assert all(data.segments[0].words[0].start_time == 1000 for data in results)
    if concurrency == 2:
        assert snapshots[0].segments[0].start_time == 1_791_000


def test_retry_preview_remains_relative_to_its_parent(monkeypatch):
    snapshots = []

    class FakeASR:
        def __init__(self, audio, segment_callback):
            self.audio = audio
            self.callback = segment_callback

        def run(self, callback=None):
            if self.audio == b"second":
                raise RuntimeError("coverage")
            return ASRData([ASRDataSeg("first", 1000, 2000)])

    asr = ChunkedASR(
        FakeASR, b"unused", asr_kwargs={"segment_callback": snapshots.append}, chunk_concurrency=1
    )
    monkeypatch.setattr(asr, "_can_retry_chunk", lambda *args: True)

    def retry(chunk, callback, error, *, asr_kwargs):
        data = ASRData([ASRDataSeg("recovered", 3000, 4000)])
        asr_kwargs["segment_callback"](data)
        return data

    monkeypatch.setattr(asr, "_retry_failed_chunk", retry)
    results = asr._transcribe_chunks([(b"first", 0), (b"second", 1_790_000)], None)
    assert snapshots[-1].segments[-1].start_time == 1_793_000
    assert results[-1].segments[0].start_time == 3000
    assert snapshots[-1].segments[0].text == "first"


def test_preview_consumer_failure_does_not_abort_transcription():
    class FakeASR:
        def __init__(self, audio, segment_callback):
            self.callback = segment_callback

        def run(self, callback=None):
            data = ASRData([ASRDataSeg("Complete result", 1000, 2000)])
            self.callback(data)
            return data

    def broken_preview(data):
        raise RuntimeError("Disconnected editor")

    asr = ChunkedASR(
        FakeASR, b"unused", asr_kwargs={"segment_callback": broken_preview}, chunk_concurrency=1
    )
    results = asr._transcribe_chunks([(b"first", 0), (b"second", 1_790_000)], None)
    assert len(results) == 2
    assert all(data.segments[0].text == "Complete result" for data in results)
