"""Audio enhancement using DeepFilterNet3 for speech denoising."""

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy-loaded model
_df_model = None
_df_state = None


def _load_model():
    """Lazy-load DeepFilterNet model."""
    global _df_model, _df_state
    if _df_model is not None:
        return

    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*torchaudio.backend.*")

        from df.enhance import init_df
        logger.info("Loading DeepFilterNet3 model...")
        _df_model, _df_state, _ = init_df()
        logger.info("DeepFilterNet3 model loaded")
    except ImportError:
        logger.warning("DeepFilterNet not installed (pip install deepfilternet)")
        raise
    except Exception as e:
        logger.error(f"Failed to load DeepFilterNet: {e}")
        raise


def enhance_audio(input_path: str, output_path: str = None) -> str:
    """Enhance audio file using DeepFilterNet3.

    Args:
        input_path: Path to input audio file (any format ffmpeg supports)
        output_path: Path for output WAV file (16kHz mono). If None, creates temp file.

    Returns:
        Path to enhanced audio file
    """
    import numpy as np
    import soundfile as sf
    import torch
    from df.enhance import enhance

    _load_model()

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix="_enhanced.wav")
        import os
        os.close(fd)

    # Convert input to 48kHz WAV (DeepFilterNet requires 48kHz)
    fd, temp_48k = tempfile.mkstemp(suffix="_48k.wav")
    import os
    os.close(fd)

    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', input_path, '-ac', '1', '-ar', '48000', temp_48k],
            capture_output=True, check=True
        )

        # Load and enhance
        audio, sr = sf.read(temp_48k)
        logger.info(f"Enhancing audio: {len(audio)/sr:.1f}s")

        # Process in 60-second chunks
        chunk_size = 60 * sr
        enhanced_chunks = []

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            enhanced = enhance(_df_model, _df_state, chunk_tensor)
            enhanced_chunks.append(enhanced.squeeze(0).numpy())
            logger.debug(f"Enhanced {i/sr:.0f}s - {min((i+chunk_size)/sr, len(audio)/sr):.0f}s")

        enhanced_audio = np.concatenate(enhanced_chunks)

        # Save as 48kHz first, then convert to 16kHz
        fd, temp_enhanced_48k = tempfile.mkstemp(suffix="_enhanced_48k.wav")
        os.close(fd)
        sf.write(temp_enhanced_48k, enhanced_audio, sr)

        subprocess.run(
            ['ffmpeg', '-y', '-i', temp_enhanced_48k, '-ac', '1', '-ar', '16000', output_path],
            capture_output=True, check=True
        )

        logger.info(f"Enhanced audio saved: {output_path}")
        return output_path

    finally:
        # Clean up temp files
        for f in [temp_48k, temp_enhanced_48k if 'temp_enhanced_48k' in dir() else None]:
            try:
                if f:
                    Path(f).unlink(missing_ok=True)
            except Exception:
                pass


def is_available() -> bool:
    """Check if DeepFilterNet is available."""
    import importlib.util
    return importlib.util.find_spec("df") is not None
