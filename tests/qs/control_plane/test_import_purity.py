"""QS-405 D6: the Control Plane imports only the standard library and itself.

The exceptions are the lazy ``import models`` in ``cli._policy_resolver``
(the model policy a no-model ``tool spawn`` reads) and in
``watchdog._messenger_model`` (QS-406: the watchdog messenger's fast-class
model). Precedent: the AST scan in ``tests/qs/test_mermaid_svg.py``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

CP_DIR = Path(__file__).resolve().parents[3] / "scripts" / "qs" / "control_plane"
LAZY_MODELS = {("cli.py", "_policy_resolver"), ("watchdog.py", "_messenger_model")}


def _foreign_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed_models = {
        id(node)
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and (path.name, fn.name) in LAZY_MODELS
        for node in ast.walk(fn)
        if isinstance(node, ast.Import) and [a.name for a in node.names] == ["models"]
    }
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level >= 1 or node.module == "__future__":
                continue
            names = [node.module or ""]
        elif isinstance(node, ast.Import):
            if id(node) in allowed_models:
                continue
            names = [a.name for a in node.names]
        else:
            continue
        bad += [f"{path.name}:{node.lineno} {n}" for n in names if n.split(".")[0] not in _ALLOWED_TOP]
    return bad


_ALLOWED_TOP = sys.stdlib_module_names | {"control_plane"}  # the standard library and itself


def test_control_plane_imports_only_stdlib_and_itself() -> None:
    files = sorted(CP_DIR.rglob("*.py"))
    assert files, CP_DIR
    bad = [b for f in files for b in _foreign_imports(f)]
    assert bad == []


def test_the_scan_catches_a_foreign_import(tmp_path: Path) -> None:
    f = tmp_path / "cli.py"
    f.write_text(
        "import os\nimport models\nfrom yaml import x\nfrom control_plane import db\n\n"
        "def _policy_resolver():\n    import models\n"
    )
    assert _foreign_imports(f) == ["cli.py:2 models", "cli.py:3 yaml"]
    other = tmp_path / "tools.py"
    other.write_text("def _policy_resolver():\n    import models\n")
    assert _foreign_imports(other) == ["tools.py:2 models"]
    wd = tmp_path / "watchdog.py"
    wd.write_text("import models\n\ndef _messenger_model():\n    import models\n")
    assert _foreign_imports(wd) == ["watchdog.py:1 models"]
