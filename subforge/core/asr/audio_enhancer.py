"""Audio enhancement using DeepFilterNet3 for speech denoising."""

import importlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from .worker_runtime import atomic_json_write as _atomic_json_write
from .worker_runtime import stop_process as _stop_process

logger = logging.getLogger(__name__)

# Lazy-loaded model
_df_model = None
_df_state = None
_df_device = None
_df_mps_failed = False
_df_lock = threading.RLock()
_DENOISE_WORKER_FLAG = "SUBFORGE_DENOISE_WORKER"
_DENOISE_WORKER_REQUEST = "SUBFORGE_DENOISE_WORKER_REQUEST"
_DENOISE_WORKER_RESULT = "SUBFORGE_DENOISE_WORKER_RESULT"
_DENOISE_WORKER_HEARTBEAT = "SUBFORGE_DENOISE_WORKER_HEARTBEAT"
_DENOISE_WORKER_IDLE_TIMEOUT = "SUBFORGE_DENOISE_WORKER_IDLE_TIMEOUT"


def _worker_idle_timeout() -> float:
    try:
        return max(30.0, float(os.environ.get(_DENOISE_WORKER_IDLE_TIMEOUT, "300")))
    except (TypeError, ValueError):
        return 300.0


def _device_name(device) -> str:
    return getattr(device, "type", str(device))


def _configure_apple_silicon_cpu(torch_module) -> None:
    """Use the measured optimal CPU parallelism for DeepFilterNet3."""
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        return
    available_threads = os.cpu_count() or 4
    configured = os.environ.get("SUBFORGE_DENOISE_THREADS", "").strip()
    if configured:
        try:
            threads = int(configured)
        except ValueError:
            logger.warning("Invalid SUBFORGE_DENOISE_THREADS=%r; using automatic value", configured)
            threads = min(8, available_threads)
    else:
        threads = min(8, available_threads)
    threads = max(1, min(threads, available_threads))
    if torch_module.get_num_threads() != threads:
        torch_module.set_num_threads(threads)
        logger.info("DeepFilterNet3 CPU inference using %d threads", threads)


def _select_device():
    """Choose the fastest available DeepFilterNet device.

    DeepFilterNet uses PyTorch. On Apple Silicon this means MPS/Metal, not the
    whisper.cpp Metal backend. Keep this isolated so unavailable or incomplete
    MPS support can fall back to CPU without disabling transcription.
    """
    import torch

    requested = os.environ.get("SUBFORGE_DENOISE_DEVICE", "auto").strip().lower()
    if requested in {"cpu", "mps"}:
        if requested == "mps" and not _mps_available(torch):
            logger.warning("Requested DeepFilterNet3 MPS device is unavailable, using CPU")
            _configure_apple_silicon_cpu(torch)
            return torch.device("cpu")
        if requested == "cpu":
            _configure_apple_silicon_cpu(torch)
        return torch.device(requested)

    if _df_mps_failed:
        _configure_apple_silicon_cpu(torch)
        return torch.device("cpu")

    # DeepFilterNet3 repeatedly transfers features and masks between its Rust
    # DSP code and PyTorch. On Apple Silicon, those small transfers make MPS
    # slower than the tuned CPU path.
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        _configure_apple_silicon_cpu(torch)
        return torch.device("cpu")

    if _mps_available(torch):
        return torch.device("mps")

    return torch.device("cpu")


def _mps_available(torch_module) -> bool:
    try:
        return (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_built()
            and torch_module.backends.mps.is_available()
        )
    except Exception:
        return False


def _configure_df_device(device) -> None:
    """Tell DeepFilterNet's internal get_device() helper which device to use."""
    try:
        from df.config import config as df_config

        df_config.set("DEVICE", _device_name(device), str, section="train")
    except Exception as e:
        logger.debug("Could not set DeepFilterNet3 device config: %s", e)


def _move_model_to_device(device) -> None:
    global _df_model, _df_device
    if _df_model is None:
        return
    _configure_df_device(device)
    _df_model = _df_model.to(device)
    _df_device = device
    logger.info("DeepFilterNet3 running on %s", _device_name(device))


def _repair_default_model_cache(enhance_module, *, force: bool = False) -> bool:
    """Discard a broken built-in model cache so DeepFilter can redownload it."""
    default_model = getattr(enhance_module, "DEFAULT_MODEL", "")
    get_model_basedir = getattr(enhance_module, "get_model_basedir", None)
    if not default_model or not callable(get_model_basedir):
        return False
    try:
        resolved_model_dir = get_model_basedir(default_model)
        if not isinstance(resolved_model_dir, (str, os.PathLike)):
            return False
        model_dir = Path(resolved_model_dir).expanduser().resolve()
        if not model_dir.exists() or model_dir.name.casefold() != str(default_model).casefold():
            return False
        config_ready = (model_dir / "config.ini").is_file()
        checkpoints_dir = model_dir / "checkpoints"
        checkpoint_ready = checkpoints_dir.is_dir() and any(
            path.is_file() and path.stat().st_size > 0
            for path in checkpoints_dir.iterdir()
        )
        if not force and config_ready and checkpoint_ready:
            return False
        logger.warning("Removing invalid DeepFilterNet3 cache: %s", model_dir)
        shutil.rmtree(model_dir)
        return True
    except OSError as exc:
        logger.warning("Could not repair invalid DeepFilterNet3 cache: %s", exc)
        return False


def _load_model(device=None):
    """Lazy-load DeepFilterNet model."""
    global _df_model, _df_state, _df_device, _df_mps_failed
    with _df_lock:
        if device is None:
            device = _select_device()

        if _df_model is not None:
            if _df_device is None or _device_name(_df_device) != _device_name(device):
                _move_model_to_device(device)
            return

        try:
            import warnings

            warnings.filterwarnings("ignore", message=".*torchaudio.backend.*")

            df_enhance = importlib.import_module("df.enhance")

            logger.info("Loading DeepFilterNet3 model...")
            _repair_default_model_cache(df_enhance)
            try:
                _df_model, _df_state, _ = df_enhance.init_df(log_file=None)
            except SystemExit:
                if not _repair_default_model_cache(df_enhance, force=True):
                    raise
                logger.warning("Retrying DeepFilterNet3 after refreshing its model cache")
                _df_model, _df_state, _ = df_enhance.init_df(log_file=None)
            try:
                _move_model_to_device(device)
            except Exception as e:
                if _device_name(device) != "mps":
                    raise
                logger.warning(
                    "DeepFilterNet3 MPS model setup failed (%s), using CPU",
                    e,
                )
                logger.debug("DeepFilterNet3 MPS model setup traceback", exc_info=True)
                import torch

                _df_mps_failed = True
                _move_model_to_device(torch.device("cpu"))
            logger.info("DeepFilterNet3 model loaded")
        except SystemExit as exc:
            _df_model = None
            _df_state = None
            _df_device = None
            raise RuntimeError(
                "DeepFilterNet3 model is incomplete or its checkpoint could not be loaded"
            ) from exc
        except ImportError:
            logger.warning("DeepFilterNet not installed (pip install deepfilternet)")
            raise
        except Exception as e:
            logger.error("Failed to load DeepFilterNet: %s", e)
            raise


def _enhance_with_python(
    input_path: str,
    output_path: str,
    atten_lim_db: float | None,
    sample_rate: int,
) -> None:
    import soundfile as sf
    import torch
    from df.enhance import enhance

    global _df_mps_failed

    chunk_size = 60 * sample_rate
    processed_frames = 0
    with _df_lock:
        device = _select_device()
        _load_model(device)
        if _df_model is None or _df_state is None:
            raise RuntimeError("DeepFilterNet3 model is not initialized")

        with sf.SoundFile(
            output_path,
            mode="w",
            samplerate=sample_rate,
            channels=1,
            format="WAV",
            subtype="FLOAT",
        ) as output_file:
            for chunk in sf.blocks(
                input_path,
                blocksize=chunk_size,
                dtype="float32",
                always_2d=False,
            ):
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
                try:
                    enhanced = enhance(
                        _df_model,
                        _df_state,
                        chunk_tensor,
                        atten_lim_db=atten_lim_db,
                    )
                except Exception:
                    if _device_name(_df_device) != "mps":
                        raise
                    logger.warning("DeepFilterNet3 MPS enhancement failed, retrying on CPU")
                    logger.debug("DeepFilterNet3 MPS enhancement traceback", exc_info=True)
                    _df_mps_failed = True
                    _move_model_to_device(torch.device("cpu"))
                    enhanced = enhance(
                        _df_model,
                        _df_state,
                        chunk_tensor,
                        atten_lim_db=atten_lim_db,
                    )
                enhanced_chunk = enhanced.squeeze(0).cpu().numpy()
                output_file.write(enhanced_chunk)
                processed_frames += len(chunk)
                logger.debug(
                    "Enhanced %.0fs - %.0fs",
                    (processed_frames - len(chunk)) / sample_rate,
                    processed_frames / sample_rate,
                )


def _enhance_audio_direct(
    input_path: str,
    output_path: str | None = None,
    *,
    atten_lim_db: float | None = None,
) -> str:
    """Enhance audio file using DeepFilterNet3.

    Args:
        input_path: Path to input audio file (any format ffmpeg supports)
        output_path: Path for output WAV file (16kHz mono). If None, creates temp file.
        atten_lim_db: Maximum noise attenuation in dB. ``None`` keeps the
            historical unrestricted DeepFilterNet behavior.

    Returns:
        Path to enhanced audio file
    """
    import soundfile as sf

    if not input_path or not Path(input_path).is_file():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix="_enhanced.wav")
        os.close(fd)

    # Convert input to 48kHz WAV (DeepFilterNet requires 48kHz)
    fd, temp_48k = tempfile.mkstemp(suffix="_48k.wav")
    os.close(fd)
    temp_enhanced_48k = None

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-ac",
                "1",
                "-ar",
                "48000",
                temp_48k,
            ],
            capture_output=True,
            check=True,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            ),
        )

        info = sf.info(temp_48k)
        if info.frames <= 0:
            raise ValueError("Input audio is empty")
        sr = info.samplerate
        logger.info(
            "Enhancing audio: %.1fs (attenuation limit: %s)",
            info.frames / sr,
            f"{atten_lim_db:g} dB" if atten_lim_db is not None else "unrestricted",
        )

        fd, temp_enhanced_48k = tempfile.mkstemp(suffix="_enhanced_48k.wav")
        os.close(fd)
        _enhance_with_python(temp_48k, temp_enhanced_48k, atten_lim_db, sr)

        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_enhanced_48k, "-ac", "1", "-ar", "16000", output_path],
            capture_output=True,
            check=True,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            ),
        )

        logger.info(f"Enhanced audio saved: {output_path}")
        return output_path

    finally:
        for f in [temp_48k, temp_enhanced_48k]:
            if f:
                Path(f).unlink(missing_ok=True)


def _enhance_in_packaged_worker(
    input_path: str,
    output_path: str | None,
    atten_lim_db: float | None,
    cancel_event=None,
) -> str:
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix="_enhanced.wav")
        os.close(fd)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="subforge-denoise-worker-") as temp_path:
        temp_dir = Path(temp_path)
        request_path = temp_dir / "request.json"
        result_path = temp_dir / "result.json"
        heartbeat_path = temp_dir / "heartbeat.json"
        log_path = temp_dir / "worker.log"
        _atomic_json_write(
            request_path,
            {
                "input_path": str(Path(input_path).resolve()),
                "output_path": str(output.resolve()),
                "atten_lim_db": atten_lim_db,
            },
        )
        env = os.environ.copy()
        env.update(
            {
                _DENOISE_WORKER_FLAG: "1",
                _DENOISE_WORKER_REQUEST: str(request_path),
                _DENOISE_WORKER_RESULT: str(result_path),
                _DENOISE_WORKER_HEARTBEAT: str(heartbeat_path),
            }
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [sys.executable],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            last_heartbeat_at = time.monotonic()
            last_heartbeat_mtime = -1
            try:
                while process.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Audio enhancement was cancelled")
                    try:
                        heartbeat_mtime = heartbeat_path.stat().st_mtime_ns
                        if heartbeat_mtime != last_heartbeat_mtime:
                            last_heartbeat_mtime = heartbeat_mtime
                            last_heartbeat_at = time.monotonic()
                    except OSError:
                        pass
                    if time.monotonic() - last_heartbeat_at > _worker_idle_timeout():
                        raise RuntimeError("DeepFilterNet3 worker stopped responding")
                    time.sleep(0.2)
            finally:
                _stop_process(process)

        payload = None
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        if process.returncode != 0 or not isinstance(payload, dict) or not payload.get("ok"):
            detail = str((payload or {}).get("error") or "").strip()
            if not detail:
                try:
                    detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:].strip()
                except OSError:
                    detail = ""
            output.unlink(missing_ok=True)
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"DeepFilterNet3 worker exited with code {process.returncode}{suffix}"
            )
        if not output.is_file() or output.stat().st_size == 0:
            output.unlink(missing_ok=True)
            raise RuntimeError("DeepFilterNet3 worker did not produce enhanced audio")
        return str(output)


def enhance_audio(
    input_path: str,
    output_path: str | None = None,
    *,
    atten_lim_db: float | None = None,
    cancel_event=None,
) -> str:
    """Enhance audio, isolating the native stack in packaged Windows builds."""
    use_worker = (
        platform.system() == "Windows"
        and getattr(sys, "frozen", False)
        and os.environ.get(_DENOISE_WORKER_FLAG) != "1"
    )
    if use_worker:
        return _enhance_in_packaged_worker(
            input_path,
            output_path,
            atten_lim_db,
            cancel_event,
        )
    return _enhance_audio_direct(input_path, output_path, atten_lim_db=atten_lim_db)


def run_packaged_denoise_worker() -> None:
    """Execute one DeepFilterNet request and contain native/SystemExit failures."""
    request_path = Path(os.environ[_DENOISE_WORKER_REQUEST])
    result_path = Path(os.environ[_DENOISE_WORKER_RESULT])
    heartbeat_path = Path(os.environ[_DENOISE_WORKER_HEARTBEAT])
    stop_heartbeat = threading.Event()

    def publish_heartbeat() -> None:
        while not stop_heartbeat.is_set():
            try:
                _atomic_json_write(heartbeat_path, {"time": time.time()})
            except OSError:
                pass
            stop_heartbeat.wait(2.0)

    heartbeat = threading.Thread(target=publish_heartbeat, daemon=True)
    heartbeat.start()
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        enhanced = _enhance_audio_direct(
            str(request["input_path"]),
            str(request["output_path"]),
            atten_lim_db=request.get("atten_lim_db"),
        )
        _atomic_json_write(result_path, {"ok": True, "path": enhanced})
        exit_code = 0
    except BaseException:
        try:
            _atomic_json_write(result_path, {"ok": False, "error": traceback.format_exc()})
        except OSError:
            pass
        exit_code = 1
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=3)
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream:
                stream.flush()
        except Exception:
            pass
    os._exit(exit_code)


def is_available() -> bool:
    """Check if DeepFilterNet is available."""
    try:
        import importlib

        importlib.import_module("soundfile")
        importlib.import_module("torch")
        enhance_module = importlib.import_module("df.enhance")
        getattr(enhance_module, "enhance")
        getattr(enhance_module, "init_df")
    except Exception as e:
        logger.info("DeepFilterNet3 unavailable: %s", e)
        return False
    return True
