import logging
from pathlib import Path

from subforge.core.asr.asr_data import ASRData
from subforge.core.asr.bcut import BcutASR
from subforge.core.asr.chunked_asr import ChunkedASR
from subforge.core.asr.faster_whisper import FasterWhisperASR
from subforge.core.asr.jianying import JianYingASR
from subforge.core.asr.whisper_api import WhisperAPI
from subforge.core.asr.whisper_cpp import WhisperCppASR
from subforge.core.asr.whisperx_asr import WhisperXASR
from subforge.core.entities import TranscribeConfig, TranscribeModelEnum

logger = logging.getLogger(__name__)


def _noop_callback(x, y):
    pass


def transcribe(audio_path: str, config: TranscribeConfig, callback=None, on_segment=None) -> ASRData:
    """Transcribe audio file using specified configuration.

    Args:
        audio_path: Path to audio file
        config: Transcription configuration
        callback: Progress callback function(progress: int, message: str)
        on_segment: Optional callback(ASRData) for partial results during VAD-segmented transcription

    Returns:
        ASRData: Transcription result data
    """
    if callback is None:
        callback = _noop_callback

    if config.transcribe_model is None:
        raise ValueError("Transcription model not set")

    # Enhance audio with DeepFilterNet3 when the optional denoise stack is
    # installed. This is especially useful before VAD on noisy in-car footage.
    enhanced_path = None
    audio_for_asr = audio_path
    if config.enable_audio_enhancement:
        try:
            from subforge.core.asr.audio_enhancer import enhance_audio, is_available

            if is_available():
                callback(6, "Enhancing audio with DeepFilterNet3...")
                logger.info("Enhancing audio with DeepFilterNet3...")
                enhanced_path = enhance_audio(audio_path)
                audio_for_asr = enhanced_path
                logger.info("Using enhanced audio: %s", enhanced_path)
            else:
                logger.info("DeepFilterNet3 not available, using original audio")
        except Exception as e:
            logger.warning("Audio enhancement failed, using original: %s", e, exc_info=True)
            audio_for_asr = audio_path
    else:
        logger.info("Audio enhancement disabled, using original audio")
        audio_for_asr = audio_path

    try:
        # Try Silero VAD preprocessing to skip silence
        asr_data = None
        try:
            from subforge.core.asr.silero_vad import detect_speech_segments
            from subforge.core.asr.silero_vad import is_available as vad_available
            if _should_use_outer_vad(config) and vad_available():
                speech_segments = detect_speech_segments(audio_for_asr)
                if speech_segments and len(speech_segments) > 1:
                    asr_data = _transcribe_segments(audio_for_asr, speech_segments, config, callback, on_segment=on_segment)
        except Exception as e:
            logger.warning(f"Silero VAD preprocessing failed, using full transcription: {e}", exc_info=True)

        # Fallback: transcribe full audio
        if asr_data is None:
            callback(10, "Preparing transcription...")
            asr = _create_asr_instance(audio_for_asr, config, on_segment=on_segment)
            callback(30, "Sending audio to ASR engine...")
            asr_data = asr.run(callback=callback)
            callback(90, "Processing results...")

        if asr_data.is_word_timestamp():
            try:
                from subforge.core.asr.speech_vad import detect_speech_segments
                from subforge.core.asr.speech_vad import is_available as vad_available

                if vad_available():
                    speech_segments = detect_speech_segments(
                        audio_for_asr,
                        threshold=0.5,
                        min_speech_ms=160,
                        min_silence_ms=180,
                        speech_pad_ms=0,
                    )
                    asr_data.refine_word_edges_with_speech_segments(speech_segments)
            except Exception as e:
                logger.debug("Word-edge VAD refinement skipped: %s", e, exc_info=True)

        asr_data.cap_abnormal_word_durations()

        # Fix boundary overlaps before text/energy post-processing sees the data.
        asr_data.fix_boundary_overlaps()

        # Filter hallucinated segments using audio energy analysis
        asr_data.filter_hallucinations(audio_path=audio_for_asr)

        # Remove duplicate text emitted around VAD/chunk boundaries before the
        # final timing pass, so exports do not keep short repeated fragments.
        asr_data.deduplicate_adjacent_text()

        # whisper.cpp sometimes cuts a spoken sentence into adjacent subtitle
        # fragments. Merge conservative mid-phrase splits before timing export,
        # but never collapse word-level timelines used by smart splitting.
        if not asr_data.is_word_timestamp():
            asr_data.merge_sentence_fragments()

        # Optimize subtitle timing if not using word timestamps
        if not config.need_word_time_stamp:
            asr_data.optimize_timing()

        # Keep the final exported timeline monotonic even if a post-processor
        # changed segment boundaries.
        asr_data.fix_boundary_overlaps()

        return asr_data
    finally:
        if enhanced_path:
            try:
                Path(enhanced_path).unlink(missing_ok=True)
            except Exception:
                pass


def _should_use_outer_vad(config: TranscribeConfig) -> bool:
    """Return whether SubForge should split audio before invoking ASR.

    whisper.cpp already has its own VAD and can process long audio with Metal
    after loading the model once. Running SubForge's outer VAD around
    whisper.cpp is expensive because it starts a new whisper-cli process for
    every speech segment, repeatedly loading large models and creating
    duplicate boundary text from overlap context.
    """
    return config.transcribe_model not in {
        TranscribeModelEnum.WHISPER_CPP,
        TranscribeModelEnum.WHISPERX,
    }


def _transcribe_segments(
    audio_path: str,
    speech_segments: list,
    config: TranscribeConfig,
    callback=None,
    on_segment=None,
    overlap_ms: int = 1500,
) -> ASRData:
    """Transcribe only speech segments detected by VAD.

    Each segment is extended by `overlap_ms` on both sides before sending to
    Whisper, giving the model full context at segment boundaries. After
    transcription, results are clipped back to the original VAD boundaries so
    the overlap audio does not appear in the final output.

    Args:
        on_segment: Optional callback(ASRData) called after each segment is transcribed,
                    used for saving partial results during processing.
        overlap_ms: Milliseconds of context overlap on each side (default 1500).
    """
    import tempfile

    from pydub import AudioSegment

    if callback is None:
        callback = _noop_callback

    audio = AudioSegment.from_file(audio_path)
    audio_len_ms = len(audio)
    all_segments = []
    total = len(speech_segments)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, (start_ms, end_ms) in enumerate(speech_segments):
            # Extend segment boundaries for Whisper context
            ext_start = max(0, start_ms - overlap_ms)
            ext_end = min(audio_len_ms, end_ms + overlap_ms)

            chunk_audio = audio[ext_start:ext_end]
            chunk_path = str(Path(tmp_dir) / f"vad_{idx:04d}.wav")
            chunk_audio.export(chunk_path, format="wav")

            logger.info(
                f"Transcribing segment {idx+1}/{total}: "
                f"{start_ms/1000:.1f}s - {end_ms/1000:.1f}s "
                f"(context: {ext_start/1000:.1f}s - {ext_end/1000:.1f}s)"
            )

            asr = _create_single_asr(chunk_path, config)
            chunk_result = asr.run()

            # Shift timestamps back to original timeline
            for seg in chunk_result.segments:
                seg.start_time += ext_start
                seg.end_time += ext_start
                # For models without timestamps (e.g. mimo-omni), use VAD boundaries
                if seg.end_time <= seg.start_time:
                    seg.end_time = end_ms
                # Clip to original VAD boundary — discard overlap context
                seg.start_time = max(seg.start_time, start_ms)
                seg.end_time = min(seg.end_time, end_ms)
            all_segments.extend(chunk_result.segments)

            progress = int((idx + 1) / total * 100)
            callback(progress, f"Transcribing segment {idx+1}/{total}")

            # Notify with partial results
            if on_segment:
                on_segment(ASRData(list(all_segments)))

    callback(100, "All segments transcribed")
    return ASRData(all_segments)


# --- ASR factory: shared kwargs builders ---

def _build_whisper_cpp_kwargs(
    config: TranscribeConfig,
    use_vad: bool = True,
    use_cache: bool = False,
    segment_callback=None,
) -> dict:
    # whisper.cpp VAD trims audio before decoding and then maps the compacted
    # audio back to the original timeline. In quiet intros this can either drop
    # speech or shift the first sentence several seconds late. Decode the full
    # audio and let JSON token timestamps plus post-processing preserve gaps.
    effective_use_vad = False
    return {
        "use_cache": use_cache,
        "need_word_time_stamp": config.need_word_time_stamp,
        "language": config.transcribe_language,
        "whisper_cpp_path": config.whisper_cpp_path or None,
        "whisper_model": config.whisper_model.value if config.whisper_model else None,
        "n_threads": getattr(config, "whisper_n_threads", 4),
        "use_vad": effective_use_vad,
        "segment_callback": segment_callback,
    }


def _build_whisper_api_kwargs(config: TranscribeConfig, use_cache: bool = True) -> dict:
    return {
        "use_cache": use_cache,
        "need_word_time_stamp": config.need_word_time_stamp,
        "language": config.transcribe_language,
        "whisper_model": config.whisper_api_model or "whisper-1",
        "api_key": config.whisper_api_key or "",
        "base_url": config.whisper_api_base or "",
        "prompt": config.whisper_api_prompt or "",
    }


def _build_faster_whisper_kwargs(config: TranscribeConfig, use_cache: bool = True) -> dict:
    return {
        "use_cache": use_cache,
        "need_word_time_stamp": config.need_word_time_stamp,
        "faster_whisper_program": config.faster_whisper_program or "",
        "language": config.transcribe_language,
        "whisper_model": (
            config.faster_whisper_model.value if config.faster_whisper_model else "base"
        ),
        "model_dir": config.faster_whisper_model_dir or "",
        "device": config.faster_whisper_device,
        "vad_filter": config.faster_whisper_vad_filter,
        "vad_threshold": config.faster_whisper_vad_threshold,
        "vad_method": (
            config.faster_whisper_vad_method.value
            if config.faster_whisper_vad_method
            else ""
        ),
        "ff_mdx_kim2": config.faster_whisper_ff_mdx_kim2,
        "one_word": config.faster_whisper_one_word,
        "prompt": config.faster_whisper_prompt,
        "compute_type": getattr(config, "faster_whisper_compute_type", "default"),
    }


def _build_whisperx_kwargs(
    config: TranscribeConfig,
    use_cache: bool = True,
    segment_callback=None,
) -> dict:
    model = (
        config.whisperx_model
        or (
            config.faster_whisper_model.value
            if config.faster_whisper_model
            else ""
        )
    )
    return {
        "use_cache": use_cache,
        "need_word_time_stamp": config.need_word_time_stamp,
        "language": config.transcribe_language,
        "whisper_model": model,
        "model_dir": config.faster_whisper_model_dir or "",
        "device": config.faster_whisper_device,
        "compute_type": getattr(config, "faster_whisper_compute_type", "default"),
        "align_model": getattr(config, "whisperx_align_model", ""),
        "batch_size": getattr(config, "whisperx_batch_size", 4),
        "segment_callback": segment_callback,
    }


def _build_simple_kwargs(config: TranscribeConfig, use_cache: bool = True) -> dict:
    return {
        "use_cache": use_cache,
        "need_word_time_stamp": config.need_word_time_stamp,
    }


# --- ASR instance creation ---

def _create_single_asr(audio_path: str, config: TranscribeConfig):
    """Create a single ASR instance (no chunking) for a short audio segment."""
    model_type = config.transcribe_model

    if model_type == TranscribeModelEnum.WHISPER_CPP:
        return WhisperCppASR(audio_path, **_build_whisper_cpp_kwargs(config, use_vad=False, use_cache=False))
    elif model_type == TranscribeModelEnum.WHISPERX:
        return WhisperXASR(audio_path, **_build_whisperx_kwargs(config, use_cache=False))
    elif model_type == TranscribeModelEnum.WHISPER_API:
        return WhisperAPI(audio_path, **_build_whisper_api_kwargs(config, use_cache=False))
    elif model_type == TranscribeModelEnum.FASTER_WHISPER:
        kwargs = _build_faster_whisper_kwargs(config, use_cache=False)
        kwargs["vad_filter"] = False  # VAD already done
        return FasterWhisperASR(audio_path, **kwargs)
    elif model_type == getattr(TranscribeModelEnum, "JIANYING", None):
        return JianYingASR(audio_path, **_build_simple_kwargs(config, use_cache=False))
    elif model_type == getattr(TranscribeModelEnum, "BIJIAN", None):
        return BcutASR(audio_path, **_build_simple_kwargs(config, use_cache=False))
    else:
        raise ValueError(f"Invalid transcription model: {model_type}")


def _create_asr_instance(audio_path: str, config: TranscribeConfig, on_segment=None) -> ChunkedASR:
    """Create appropriate ASR instance based on configuration."""
    model_type = config.transcribe_model

    if model_type == TranscribeModelEnum.WHISPER_CPP:
        return ChunkedASR(asr_class=WhisperCppASR, audio_path=audio_path,
                          asr_kwargs=_build_whisper_cpp_kwargs(
                              config,
                              use_cache=False,
                              segment_callback=on_segment,
                          ),
                          chunk_concurrency=1, chunk_length=60 * 60)

    elif model_type == TranscribeModelEnum.WHISPERX:
        return ChunkedASR(asr_class=WhisperXASR, audio_path=audio_path,
                          asr_kwargs=_build_whisperx_kwargs(
                              config,
                              use_cache=False,
                              segment_callback=on_segment,
                          ),
                          chunk_concurrency=1, chunk_length=60 * 60)

    elif model_type == TranscribeModelEnum.WHISPER_API:
        return ChunkedASR(asr_class=WhisperAPI, audio_path=audio_path,
                          asr_kwargs=_build_whisper_api_kwargs(config))

    elif model_type == TranscribeModelEnum.FASTER_WHISPER:
        return ChunkedASR(asr_class=FasterWhisperASR, audio_path=audio_path,
                          asr_kwargs=_build_faster_whisper_kwargs(config),
                          chunk_concurrency=1, chunk_length=60 * 20)

    elif model_type == getattr(TranscribeModelEnum, "JIANYING", None):
        return ChunkedASR(asr_class=JianYingASR, audio_path=audio_path,
                          asr_kwargs=_build_simple_kwargs(config))

    elif model_type == getattr(TranscribeModelEnum, "BIJIAN", None):
        return ChunkedASR(asr_class=BcutASR, audio_path=audio_path,
                          asr_kwargs=_build_simple_kwargs(config))

    else:
        raise ValueError(f"Invalid transcription model: {model_type}")
