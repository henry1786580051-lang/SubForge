"""Source-tree entry point for the isolated MLX Whisper worker."""

from subforge.core.asr.whisperx_asr import run_packaged_mlx_whisper_worker

if __name__ == "__main__":
    run_packaged_mlx_whisper_worker()
