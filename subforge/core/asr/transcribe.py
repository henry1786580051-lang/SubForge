from subforge.core.asr.asr_data import ASRData
from subforge.core.asr.bcut import BcutASR
from subforge.core.asr.chunked_asr import ChunkedASR
from subforge.core.asr.faster_whisper import FasterWhisperASR
from subforge.core.asr.jianying import JianYingASR
from subforge.core.asr.whisper_api import WhisperAPI
from subforge.core.asr.whisper_cpp import WhisperCppASR
from subforge.core.entities import TranscribeConfig, TranscribeModelEnum


def transcribe(audio_path: str, config: TranscribeConfig, callback=None) -> ASRData:
    """Transcribe audio file using specified configuration.

    Args:
        audio_path: Path to audio file
        config: Transcription configuration
        callback: Progress callback function(progress: int, message: str)

    Returns:
        ASRData: Transcription result data
    """
    import logging
    logger = logging.getLogger(__name__)

    def _default_callback(x, y):
        pass

    if callback is None:
        callback = _default_callback

    if config.transcribe_model is None:
        raise ValueError("Transcription model not set")

    # Enhance audio with DeepFilterNet (speech denoising)
    enhanced_path = None
    try:
        from subforge.core.asr.audio_enhancer import enhance_audio, is_available
        if is_available():
            logger.info("Enhancing audio with DeepFilterNet3...")
            enhanced_path = enhance_audio(audio_path)
            audio_for_asr = enhanced_path
            logger.info(f"Using enhanced audio: {enhanced_path}")
        else:
            logger.info("DeepFilterNet not available, using original audio")
            audio_for_asr = audio_path
    except Exception as e:
        logger.warning(f"Audio enhancement failed, using original: {e}")
        import traceback
        traceback.print_exc()
        audio_for_asr = audio_path

    try:
        # Try Silero VAD preprocessing to skip silence
        asr_data = None
        try:
            from subforge.core.asr.silero_vad import detect_speech_segments
            from subforge.core.asr.silero_vad import is_available as vad_available
            if vad_available():
                speech_segments = detect_speech_segments(audio_for_asr)
                if speech_segments and len(speech_segments) > 1:
                    asr_data = _transcribe_segments(audio_for_asr, speech_segments, config, callback)
        except Exception as e:
            logger.warning(f"Silero VAD preprocessing failed, using full transcription: {e}")
            import traceback
            traceback.print_exc()

        # Fallback: transcribe full audio
        if asr_data is None:
            asr = _create_asr_instance(audio_for_asr, config)
            asr_data = asr.run(callback=callback)

        # Filter hallucinated segments using audio energy analysis
        asr_data.filter_hallucinations(audio_path=audio_for_asr)

        # Optimize subtitle timing if not using word timestamps
        if not config.need_word_time_stamp:
            asr_data.optimize_timing()

        return asr_data
    finally:
        # Clean up enhanced audio
        if enhanced_path:
            try:
                from pathlib import Path
                Path(enhanced_path).unlink(missing_ok=True)
            except Exception:
                pass


def _create_asr_instance(audio_path: str, config: TranscribeConfig) -> ChunkedASR:
    """Create appropriate ASR instance based on configuration.

    Args:
        audio_path: Path to audio file
        config: Transcription configuration

    Returns:
        ChunkedASR: Chunked ASR instance ready to run
    """
    model_type = config.transcribe_model

    if model_type == TranscribeModelEnum.JIANYING:
        return _create_jianying_asr(audio_path, config)

    elif model_type == TranscribeModelEnum.BIJIAN:
        return _create_bijian_asr(audio_path, config)

    elif model_type == TranscribeModelEnum.WHISPER_CPP:
        return _create_whisper_cpp_asr(audio_path, config)

    elif model_type == TranscribeModelEnum.WHISPER_API:
        return _create_whisper_api_asr(audio_path, config)

    elif model_type == TranscribeModelEnum.FASTER_WHISPER:
        return _create_faster_whisper_asr(audio_path, config)

    else:
        raise ValueError(f"Invalid transcription model: {model_type}")


def _transcribe_segments(
    audio_path: str,
    speech_segments: list,
    config: TranscribeConfig,
    callback=None,
) -> ASRData:
    """Transcribe only speech segments detected by VAD.

    Extracts each speech segment as a temporary WAV, transcribes it,
    then adjusts timestamps back to the original audio timeline.
    """
    import logging
    import tempfile
    from pathlib import Path

    from pydub import AudioSegment

    logger = logging.getLogger(__name__)
    audio = AudioSegment.from_file(audio_path)
    all_segments = []
    total = len(speech_segments)

    def _default_callback(x, y):
        pass
    if callback is None:
        callback = _default_callback

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, (start_ms, end_ms) in enumerate(speech_segments):
            chunk_audio = audio[start_ms:end_ms]
            chunk_path = str(Path(tmp_dir) / f"vad_{idx:04d}.wav")
            chunk_audio.export(chunk_path, format="wav")

            logger.info(f"Transcribing segment {idx+1}/{total}: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s")

            # Each segment is short (< 1 min), use ASR directly without ChunkedASR
            asr = _create_single_asr(chunk_path, config)
            chunk_result = asr.run(callback=_default_callback)

            # Shift timestamps back to original timeline
            for seg in chunk_result.segments:
                seg.start_time += start_ms
                seg.end_time += start_ms
            all_segments.extend(chunk_result.segments)

            progress = int((idx + 1) / total * 100)
            callback(progress, f"Segment {idx+1}/{total}")

    callback(100, "All segments transcribed")
    return ASRData(all_segments)


def _create_single_asr(audio_path: str, config: TranscribeConfig):
    """Create a single ASR instance (no chunking) for a short audio segment."""
    from subforge.core.asr.bcut import BcutASR
    from subforge.core.asr.faster_whisper import FasterWhisperASR
    from subforge.core.asr.jianying import JianYingASR
    from subforge.core.asr.whisper_api import WhisperAPI
    from subforge.core.asr.whisper_cpp import WhisperCppASR

    model_type = config.transcribe_model
    kwargs = {
        "use_cache": False,  # Don't cache individual segments
        "need_word_time_stamp": config.need_word_time_stamp,
    }

    if model_type == TranscribeModelEnum.WHISPER_CPP:
        kwargs.update({
            "language": config.transcribe_language,
            "whisper_model": config.whisper_model.value if config.whisper_model else None,
            "n_threads": getattr(config, "whisper_n_threads", 4),
            "use_vad": False,  # VAD already done, don't re-run
        })
        return WhisperCppASR(audio_path, **kwargs)
    elif model_type == TranscribeModelEnum.WHISPER_API:
        kwargs.update({
            "language": config.transcribe_language,
            "whisper_model": config.whisper_api_model or "whisper-1",
            "api_key": config.whisper_api_key or "",
            "base_url": config.whisper_api_base or "",
            "prompt": config.whisper_api_prompt or "",
        })
        return WhisperAPI(audio_path, **kwargs)
    elif model_type == TranscribeModelEnum.FASTER_WHISPER:
        kwargs.update({
            "faster_whisper_program": config.faster_whisper_program or "",
            "language": config.transcribe_language,
            "whisper_model": config.faster_whisper_model.value if config.faster_whisper_model else "base",
            "model_dir": config.faster_whisper_model_dir or "",
            "device": config.faster_whisper_device,
            "vad_filter": False,
            "compute_type": getattr(config, "faster_whisper_compute_type", "default"),
        })
        return FasterWhisperASR(audio_path, **kwargs)
    elif model_type == TranscribeModelEnum.JIANYING:
        return JianYingASR(audio_path, **kwargs)
    elif model_type == TranscribeModelEnum.BIJIAN:
        return BcutASR(audio_path, **kwargs)
    else:
        raise ValueError(f"Invalid transcription model: {model_type}")


def _create_jianying_asr(audio_path: str, config: TranscribeConfig) -> ChunkedASR:
    """Create JianYing ASR instance with chunking support."""
    asr_kwargs = {
        "use_cache": True,
        "need_word_time_stamp": config.need_word_time_stamp,
    }
    return ChunkedASR(
        asr_class=JianYingASR, audio_path=audio_path, asr_kwargs=asr_kwargs
    )


def _create_bijian_asr(audio_path: str, config: TranscribeConfig) -> ChunkedASR:
    """Create Bijian ASR instance with chunking support."""
    asr_kwargs = {
        "use_cache": True,
        "need_word_time_stamp": config.need_word_time_stamp,
    }
    return ChunkedASR(asr_class=BcutASR, audio_path=audio_path, asr_kwargs=asr_kwargs)


def _create_whisper_cpp_asr(audio_path: str, config: TranscribeConfig) -> ChunkedASR:
    """Create WhisperCpp ASR instance with chunking support."""
    asr_kwargs = {
        "use_cache": True,
        "need_word_time_stamp": config.need_word_time_stamp,
        "language": config.transcribe_language,
        "whisper_model": config.whisper_model.value if config.whisper_model else None,
        "n_threads": getattr(config, "whisper_n_threads", 4),
        "use_vad": True,
    }
    return ChunkedASR(
        asr_class=WhisperCppASR,
        audio_path=audio_path,
        asr_kwargs=asr_kwargs,
        chunk_concurrency=1,  # 本地转录使用单线程
        chunk_length=60 * 20,  # 每块20分钟
    )


def _create_whisper_api_asr(audio_path: str, config: TranscribeConfig) -> ChunkedASR:
    """Create Whisper API ASR instance with chunking support."""
    asr_kwargs = {
        "use_cache": True,
        "need_word_time_stamp": config.need_word_time_stamp,
        "language": config.transcribe_language,
        "whisper_model": config.whisper_api_model or "whisper-1",
        "api_key": config.whisper_api_key or "",
        "base_url": config.whisper_api_base or "",
        "prompt": config.whisper_api_prompt or "",
    }
    return ChunkedASR(
        asr_class=WhisperAPI, audio_path=audio_path, asr_kwargs=asr_kwargs
    )


def _create_faster_whisper_asr(audio_path: str, config: TranscribeConfig) -> ChunkedASR:
    """Create FasterWhisper ASR instance with chunking support."""
    asr_kwargs = {
        "use_cache": True,
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
    return ChunkedASR(
        asr_class=FasterWhisperASR,
        audio_path=audio_path,
        asr_kwargs=asr_kwargs,
        chunk_concurrency=1,  # 本地转录使用单线程
        chunk_length=60 * 20,  # 每块20分钟
    )


if __name__ == "__main__":
    # 示例用法
    from subforge.core.entities import WhisperModelEnum

    # 创建配置
    config = TranscribeConfig(
        transcribe_model=TranscribeModelEnum.WHISPER_CPP,
        transcribe_language="zh",
        whisper_model=WhisperModelEnum.MEDIUM,
    )

    # 转录音频
    audio_file = "test.wav"

    def progress_callback(progress: int, message: str):
        print(f"Progress: {progress}%, Message: {message}")

    result = transcribe(audio_file, config, callback=progress_callback)
    print(result)
