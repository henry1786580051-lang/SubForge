import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.main import app


def test_retired_catalog_has_no_routes_or_importable_modules():
    assert not any(
        getattr(route, "path", "").startswith("/api/free-models")
        for route in app.routes
    )
    assert importlib.util.find_spec("app.api.free_models") is None
    assert importlib.util.find_spec("app.services.free_model_scanner") is None
