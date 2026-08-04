import logging

from subforge.core.utils import logger as logger_module


def test_root_logger_persists_standard_backend_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "LOG_PATH", tmp_path)
    monkeypatch.setattr(logger_module, "_active_log_file", None)
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    for handler in previous_handlers:
        root.removeHandler(handler)

    try:
        log_path = logger_module.configure_root_logger()
        logging.getLogger("app.api.transcribe").error("windows backend failure")
        for handler in root.handlers:
            handler.flush()

        assert log_path == tmp_path / "app.log"
        assert "windows backend failure" in log_path.read_text(encoding="utf-8")
    finally:
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)
        for handler in previous_handlers:
            root.addHandler(handler)
