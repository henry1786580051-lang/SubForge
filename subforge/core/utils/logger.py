import logging
import logging.handlers
import tempfile
from pathlib import Path

from ...config import LOG_LEVEL, LOG_PATH

FALLBACK_LOG_PATH = Path(tempfile.gettempdir()) / "SubForge" / "logs"
_active_log_file: Path | None = None
_ROOT_HANDLER_MARKER = "_subforge_root_handler"


def get_active_log_file(default: Path | None = None) -> Path:
    return _active_log_file or default or (LOG_PATH / "app.log")


def _build_file_handler(
    log_file: str,
    level: int,
    formatter: logging.Formatter,
) -> logging.Handler:
    path = Path(log_file)
    candidates = [path]
    if path.parent != FALLBACK_LOG_PATH:
        candidates.append(FALLBACK_LOG_PATH / path.name)

    last_error: OSError | None = None
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                candidate,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            handler.setLevel(level)
            handler.setFormatter(formatter)
            global _active_log_file
            _active_log_file = candidate
            return handler
        except OSError as exc:
            last_error = exc

    raise last_error or OSError(f"Unable to create log file handler: {log_file}")


def setup_logger(
    name: str,
    level: int = LOG_LEVEL,
    info_fmt: str = "%(message)s",  # INFO级别使用简化格式
    default_fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # 其他级别使用详细格式
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    log_file: str = str(LOG_PATH / "app.log"),
    console_output: bool = True,
) -> logging.Logger:
    """
    创建并配置一个日志记录器，INFO级别使用简化格式。

    参数:
    - name: 日志记录器的名称
    - level: 日志级别
    - info_fmt: INFO级别的日志格式字符串
    - default_fmt: 其他级别的日志格式字符串
    - datefmt: 时间格式字符串
    - log_file: 日志文件路径
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        class LevelSpecificFormatter(logging.Formatter):
            """Thread-safe formatter that uses different formats per log level."""
            def format(self, record):
                # Use local variable instead of mutating shared _style._fmt
                fmt = info_fmt if record.levelno == logging.INFO else default_fmt
                formatter = logging.Formatter(fmt, datefmt=datefmt)
                return formatter.format(record)

        level_formatter = LevelSpecificFormatter(default_fmt, datefmt=datefmt)

        # 只在console_output为True时添加控制台处理器
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(level_formatter)
            logger.addHandler(console_handler)

        # 文件处理器
        if log_file:
            try:
                logger.addHandler(_build_file_handler(log_file, level, level_formatter))
            except OSError as exc:
                if console_output:
                    logger.warning("Failed to initialize file logging: %s", exc)

    # 设置特定库的日志级别为ERROR以减少日志噪音
    error_loggers = [
        "urllib3",
        "requests",
        "openai",
        "httpx",
        "httpcore",
        "ssl",
        "certifi",
    ]
    for lib in error_loggers:
        logging.getLogger(lib).setLevel(logging.ERROR)

    return logger


def configure_root_logger(level: int = LOG_LEVEL) -> Path:
    """Persist logs from backend modules that use the standard logging API."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        if getattr(handler, _ROOT_HANDLER_MARKER, False):
            return get_active_log_file()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = _build_file_handler(str(LOG_PATH / "app.log"), level, formatter)
    setattr(handler, _ROOT_HANDLER_MARKER, True)
    root.addHandler(handler)
    return get_active_log_file()
