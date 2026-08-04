import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_core_does_not_depend_on_backend_or_ui():
    violations = []
    for path in (ROOT / "subforge" / "core").rglob("*.py"):
        for module in _imports(path):
            if module == "app" or module.startswith(("app.", "backend.", "subforge.ui")):
                violations.append(f"{path.relative_to(ROOT)} -> {module}")

    assert violations == []


def test_settings_model_does_not_depend_on_interface_layers():
    violations = []
    for path in (ROOT / "subforge" / "settings").rglob("*.py"):
        for module in _imports(path):
            if module == "app" or module.startswith(
                ("app.", "backend.", "subforge.ui", "subforge.cli")
            ):
                violations.append(f"{path.relative_to(ROOT)} -> {module}")

    assert violations == []


def test_application_layer_does_not_depend_on_interface_layers():
    violations = []
    for path in (ROOT / "subforge" / "application").rglob("*.py"):
        for module in _imports(path):
            if module == "app" or module.startswith(
                ("app.", "backend.", "subforge.ui", "subforge.cli")
            ):
                violations.append(f"{path.relative_to(ROOT)} -> {module}")

    assert violations == []


def test_current_desktop_launcher_does_not_import_legacy_pyqt_ui():
    imports = _imports(ROOT / "launcher.py")

    assert not any(
        module == "subforge.ui" or module.startswith("subforge.ui.")
        for module in imports
    )
