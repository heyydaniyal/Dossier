"""Global rule 2: the agent runtime must never import the evaluation harness.

Scans every .py file in the runtime packages with the AST (static imports) and also flags
string literals naming the eval package (catches importlib/__import__ tricks).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGES = ["src/agents", "src/tools", "src/rag", "src/app"]
FORBIDDEN_PREFIXES = ("src.eval",)


def _violations(py: Path) -> list[str]:
    tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    pkg_parts = py.relative_to(ROOT).with_suffix("").parts[:-1]
    out: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # resolve relative import
                base = list(pkg_parts[: len(pkg_parts) - node.level + 1])
                mod = ".".join(base + ([node.module] if node.module else []))
                names = [mod] + [f"{mod}.{a.name}" for a in node.names]
            else:
                names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(node.value.startswith(p) for p in FORBIDDEN_PREFIXES):
                out.append(f"{py.name}:{node.lineno} string literal '{node.value}'")
        for n in names:
            if n == "src.eval" or n.startswith("src.eval."):
                out.append(f"{py.name}:{node.lineno} imports '{n}'")
    return out


def _runtime_files() -> list[Path]:
    return [p for pkg in RUNTIME_PACKAGES for p in (ROOT / pkg).rglob("*.py")]


def test_runtime_packages_exist():
    for pkg in RUNTIME_PACKAGES:
        assert (ROOT / pkg / "__init__.py").exists(), pkg


@pytest.mark.parametrize("py", _runtime_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_runtime_never_imports_eval(py: Path):
    assert _violations(py) == []


def test_detector_catches_violations(tmp_path: Path):
    """The guard itself must work: plant each violation type and check it is caught."""
    d = ROOT / "src" / "agents" / "_tmp_probe"
    d.mkdir(exist_ok=True)
    try:
        cases = {
            "a.py": "import src.eval\n",
            "b.py": "from src.eval.labels import AlertLabel\n",
            # three dots from src/agents/_tmp_probe resolve to src.eval
            "c.py": "from ...eval import labels\n",
            "d.py": "import importlib\nimportlib.import_module('src.eval.labels')\n",
        }
        for name, code in cases.items():
            f = d / name
            f.write_text(code)
            assert _violations(f), f"not caught: {code!r}"
        ok = d / "ok.py"
        ok.write_text("from src.contracts import Alert\n")
        assert _violations(ok) == []
    finally:
        for f in d.glob("*"):
            f.unlink()
        d.rmdir()
