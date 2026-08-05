"""doctor command -- diagnose local dependencies and configuration."""

import json
import shutil
import subprocess
import sys
from argparse import Namespace
from dataclasses import asdict, dataclass
from datetime import date

from subforge.cli import exit_codes as EXIT
from subforge.cli.config import CONFIG_FILE, get


@dataclass
class Check:
    name: str
    status: str
    message: str
    fix: str = ""


def run(args: Namespace, config: dict) -> int:
    checks = _run_checks(config, check_api=bool(getattr(args, "check_api", False)))
    if getattr(args, "json", False):
        print(json.dumps({"checks": [asdict(c) for c in checks]}, ensure_ascii=False, indent=2))
    else:
        _print_checks(checks)
    return EXIT.DEPENDENCY_MISSING if any(c.status == "error" for c in checks) else EXIT.SUCCESS


def _run_checks(config: dict, *, check_api: bool = False) -> list[Check]:
    checks: list[Check] = []
    checks.append(_check_python())
    checks.append(_check_command("ffmpeg", "Required for audio extraction and resampling."))
    checks.append(_check_command("ffprobe", "Required for media duration checks."))
    checks.append(_check_ytdlp())
    checks.append(_check_config_file())
    checks.extend(_check_transcribe(config))
    checks.extend(_check_subtitle(config))
    if check_api:
        checks.extend(_check_api(config))
    return checks


def _check_python() -> Check:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if (3, 10) <= sys.version_info[:2] < (3, 13):
        return Check("python", "ok", f"Python {version}")
    return Check("python", "error", f"Python {version} is unsupported", "Use Python >=3.10,<3.13")


def _check_command(name: str, purpose: str) -> Check:
    path = shutil.which(name)
    if not path:
        return Check(name, "error", f"{name} not found. {purpose}", f"Install {name} and make sure it is on PATH")
    version = _command_version(name)
    return Check(name, "ok", f"{path}" + (f" ({version})" if version else ""))


def _check_ytdlp() -> Check:
    path = shutil.which("yt-dlp")
    if path:
        version = _command_version("yt-dlp")
        if version and _yt_dlp_version_is_old(version):
            return Check("yt-dlp", "warn", f"{path} ({version}) may be old", "Update yt-dlp if online downloads fail")
        return Check("yt-dlp", "ok", f"{path}" + (f" ({version})" if version else ""))
    try:
        import yt_dlp
        import yt_dlp.version

        version = getattr(yt_dlp.version, "__version__", "")
        return Check("yt-dlp", "ok", "embedded yt_dlp module" + (f" ({version})" if version else ""))
    except Exception:
        return Check("yt-dlp", "error", "yt-dlp not found. Required by subforge download.", "Install yt-dlp and make sure it is on PATH")


def _yt_dlp_version_is_old(version: str) -> bool:
    try:
        year, month, _day = [int(part) for part in version.split(".")[:3]]
        release_date = date(year, month, _day)
    except Exception:
        return False
    # Stable yt-dlp versions are date-like.
    return (date.today() - release_date).days > 90


def _command_version(name: str) -> str:
    try:
        result = subprocess.run([name, "-version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout:
            return result.stdout.splitlines()[0][:100]
    except Exception:
        pass
    try:
        result = subprocess.run([name, "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout:
            return result.stdout.splitlines()[0][:100]
    except Exception:
        pass
    return ""


def _check_config_file() -> Check:
    if CONFIG_FILE.exists():
        return Check("config.file", "ok", str(CONFIG_FILE))
    return Check(
        "config.file",
        "warn",
        f"Config file does not exist: {CONFIG_FILE}",
        "Run 'subforge config init' or set values with environment variables",
    )


def _check_transcribe(config: dict) -> list[Check]:
    asr = get(config, "transcribe.asr", "whisper-api")
    checks = [Check("transcribe.asr", "ok", f"default ASR: {asr}")]
    if asr == "whisper-api" and not get(config, "whisper_api.api_key", ""):
        checks.append(Check("whisper_api.api_key", "error", "Whisper API key is missing", "Run 'subforge config set whisper_api.api_key <key>'"))
    if asr == "whisper-cpp" and not any(shutil.which(n) for n in ["whisper-cpp", "whisper", "whisper-cpp-main"]):
        checks.append(Check("whisper-cpp", "error", "whisper.cpp binary not found", "Install whisper.cpp or choose --asr whisper-api"))
    return checks


def _check_subtitle(config: dict) -> list[Check]:
    checks: list[Check] = []
    optimize = bool(get(config, "subtitle.optimize", True))
    split = bool(get(config, "subtitle.split", True))
    translator = get(config, "translate.service", "bing")
    needs_llm = optimize or split or translator == "llm"
    checks.append(Check("subtitle.processing", "ok", f"ai_polish={optimize}, split={split}, translator={translator}"))
    if needs_llm and not get(config, "llm.api_key", ""):
        checks.append(Check("llm.api_key", "warn", "LLM API key is missing; AI polish/split/LLM translation will fail", "Run 'subforge config set llm.api_key <key>' or disable AI polish/split"))
    if needs_llm and not get(config, "llm.model", ""):
        checks.append(Check("llm.model", "error", "LLM model is missing", "Run 'subforge config set llm.model <model>'"))
    if translator == "bing" and not get(config, "translate.azure_key", ""):
        checks.append(
            Check(
                "translate.azure_key",
                "error",
                "Microsoft Azure Translator API key is missing",
                "Run 'subforge config set translate.azure_key <key>'",
            )
        )
    return checks


def _check_api(config: dict) -> list[Check]:
    checks: list[Check] = []
    if not get(config, "llm.api_key", ""):
        return checks
    checks.append(Check("api.llm", "warn", "--check-api is currently limited to configuration validation", "Run a short subtitle task to verify provider access"))
    return checks


def _print_checks(checks: list[Check]) -> None:
    for check in checks:
        prefix = {"ok": "OK", "warn": "WARN", "error": "ERROR"}.get(check.status, check.status.upper())
        print(f"{prefix:5} {check.name}: {check.message}")
        if check.fix:
            print(f"      fix: {check.fix}")
    errors = sum(1 for c in checks if c.status == "error")
    warnings = sum(1 for c in checks if c.status == "warn")
    if errors:
        print(f"ERROR Doctor found {errors} error(s) and {warnings} warning(s)")
    elif warnings:
        print(f"WARN  Doctor found {warnings} warning(s)")
    else:
        print("OK    Doctor found no issues")
