"""Global rule 2: no code outside the evaluation harness and the offline P2/P1 generators may reach
ground truth -- by import, dynamic import, or by reading an evaluation/devtools store or the raw
label column by path.

Scope (widened in P3 task 0 after the independent P2 review, finding M-1): EVERY .py file under
src/ except src/eval, every .py under scripts/ except scripts/p2 and scripts/audit, and every code
cell of every notebook. The P2 version scanned six named packages only, and missed
src/features, src/models, other scripts, notebooks, string-built imports, exec/runpy/sys.path
tricks and path literals.

Checks per file (AST, docstrings ignored; adjacent string constants are folded, so
"is_" + "laundering" is seen as one string):
  1. imports of src.eval / scripts.p2 / scripts.audit (absolute, relative, or a bare `eval`
     package reached through sys.path);
  2. string literals naming those packages or their paths;
  3. dynamic code loading: __import__, exec, eval, compile, importlib.import_module,
     importlib.util.spec_from_file_location, runpy, and any sys.path change;
  4. ground-truth / evaluation-store literals: the label column, typology/attempt fields, eval and
     devtools store names and paths, the patterns file, and the interim transactions file read
     other than through src.data.transactions.scan_transactions (the label-free loader).
src/contracts/models.py is exempt from check 4 only: it DEFINES the forbidden field list.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGES = ["src/agents", "src/tools", "src/rag", "src/app", "src/alerts", "src/data"]
# Ground-truth readers: the evaluation harness, the offline data audit (P1) and the offline P2
# generators (TRAIN-label calibration, planted KYC explanations, simulated dispositions).
FORBIDDEN_PREFIXES = ("src.eval", "scripts.audit", "scripts.p2")
EXEMPT_DIRS = ("src/eval", "scripts/p2", "scripts/audit")
FORBIDDEN_PATH_TOKENS = ("src/eval", "scripts/p2", "scripts/audit", "src\\eval")
# Substrings that name a ground-truth column, file or store. Matched case-insensitively inside any
# non-docstring string literal (after folding "a" + "b"). Kept to names that ordinary runtime code
# never needs; generic words ("typology", "eval", "label") are NOT here (second-pass review M3:
# they would block the Typology agent and ordinary model code).
TRUTH_TOKENS = (
    "is_laundering",
    "is laundering",
    "alert_labels",
    "laundering_legs",
    "positive_account_days",
    "planting.parquet",
    "agent_split.parquet",
    "agent_split_p2v1",
    "agent_dev_alert_ids",
    "p2_agent_split_v2",
    "/eval/",
    "\\eval\\",
    "devtools",
    "/archive/",
    "_patterns.parquet",
    "patterns.txt",
    "_trans.parquet",
    "trans.csv",
    "interim/",  # the P1 interim files are read only through src/data/transactions.py
    "interim\\",
)
DATA_SUFFIXES = (".parquet", ".csv", ".txt")  # a wildcard over these is a blind data read
# Exact (whole-literal) eval-store column names, compared after lowercasing and mapping spaces and
# hyphens to "_". (Generic contract words such as "label" are left out on purpose.)
TRUTH_EXACT_COLUMNS = {
    "is_true_positive",
    "label_definition_id",
    "pattern_id",
    "pattern_instance_id",
    "typology_truth",
    "gt_typology",
    "ground_truth",
    "attempt_id",
    "attempt_ids",
    "has_unattributed",
    "n_laundering_txns",
    "is_pos",
    "laundering_account",
    "agent_group",
}
# Path segments: flagged only where a path is being built (a / b, Path(...), joinpath, os.path.join)
PATH_SEGMENTS = {"eval", "devtools", "archive"}
PATH_CALLS = {"Path", "PurePath", "PosixPath", "WindowsPath", "joinpath", "join"}
# Listing data directories (glob/rglob/walk/...) is how a store could be found without naming it.
# Banned in packages that never need to list files; src/rag (corpus) and src/data may.
LISTING_CALLS = {"glob", "rglob", "iglob", "walk", "listdir", "scandir", "iterdir"}
NO_LISTING_DIRS = ("src/features", "src/models", "src/tools", "src/agents", "src/app")
CHECK4_EXEMPT = ("src/contracts/models.py",)
# Narrow, reasoned exceptions (file -> exact literals / checks allowed there):
#  - verify_interim checks the sidecar sha256 of all three P1 files; it never opens their content.
#  - the KYC generator (label-free) writes the planting record, whose columns it must name.
ALLOWED_LITERALS = {
    "src/data/transactions.py": {"patterns", "Patterns.txt", "Trans.csv"},
}
# the module that DEFINES interim_path and the label-free loader
LOADER_MODULE = "src/data/transactions.py"
#  - Streamlit puts only the script directory on sys.path; the app adds the repo root so that
#    `src.*` imports resolve on the host. Imports of src.eval stay banned by check 1.
SYS_PATH_EXEMPT = ("src/app/streamlit_app.py",)
SYS_PATH_MUTATORS = {"insert", "append", "extend", "remove", "clear", "pop"}
DYNAMIC_CALLS = {"__import__", "exec", "eval", "compile"}
DYNAMIC_ATTRS = {"import_module", "spec_from_file_location", "run_path", "run_module"}
SAFE_TRANS_LOADERS = {"scan_transactions", "verify_interim"}
DYNAMIC_MODULES = ("runpy", "importlib")


def _docstring_ids(tree: ast.AST) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0].value))
    return out


UNKNOWN = "\x00"  # stands for a non-constant part of a string expression


def _fold(node: ast.AST) -> str | None:
    """Value of a string expression with every non-constant part replaced by UNKNOWN:
    "a" + "b" -> "ab";  P + "/eval/" + x -> "\\x00/eval/\\x00";  f"{p}/eval" -> "\\x00/eval".
    None if the expression has no string constant at all."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _fold(node.left), _fold(node.right)
        if a is None and b is None:
            return None
        return (a if a is not None else UNKNOWN) + (b if b is not None else UNKNOWN)
    if isinstance(node, ast.JoinedStr):
        parts = [v.value if isinstance(v, ast.Constant) else UNKNOWN for v in node.values]
        return "".join(parts) if any(p != UNKNOWN for p in parts) else None
    return None


def _call_name(call: ast.Call) -> str:
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def _norm(x: str) -> str:
    return x.lower().replace(" ", "_").replace("-", "_")


def _is_sys_path(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _safe_interim_calls(tree: ast.AST) -> set[int]:
    """interim_path(...) calls whose result goes (directly, by keyword, or through a simple
    assignment to a name) into the label-free loader scan_transactions / verify_interim."""
    safe: set[int] = set()
    passed_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in SAFE_TRANS_LOADERS:
            vals = [*node.args, *(k.value for k in node.keywords)]
            safe |= {id(v) for v in vals}
            passed_names |= {v.id for v in vals if isinstance(v, ast.Name)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if any(isinstance(t, ast.Name) and t.id in passed_names for t in node.targets):
                safe.add(id(node.value))
    return safe


def _path_segment_ids(tree: ast.AST) -> set[int]:
    """String constants used as a path segment."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            out |= {id(node.left), id(node.right)}
        elif isinstance(node, ast.Call) and _call_name(node) in PATH_CALLS:
            out |= {id(a) for a in node.args}
    return out


def _violations_in_source(src: str, rel: str, pkg_parts: tuple[str, ...] = ()) -> list[str]:
    tree = ast.parse(src, filename=rel)
    docs = _docstring_ids(tree)
    rel = rel.replace("\\", "/")
    check4 = rel not in CHECK4_EXEMPT
    allowed = ALLOWED_LITERALS.get(rel, set())
    safe_interim = _safe_interim_calls(tree)
    segments = _path_segment_ids(tree)
    no_listing = any(rel.startswith(d + "/") for d in NO_LISTING_DIRS)
    out: list[str] = []
    seen_str: set[int] = set()
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", 0)
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
        for n in names:
            if any(n == p or n.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
                out.append(f"{rel}:{ln} imports '{n}'")
            if n == "eval" or n.startswith("eval."):
                out.append(f"{rel}:{ln} imports bare '{n}' (sys.path trick)")
            if any(n == m or n.startswith(m + ".") for m in DYNAMIC_MODULES):
                out.append(f"{rel}:{ln} imports {n} (dynamic loading)")
        # 3. dynamic code loading, sys.path changes, directory listing
        if isinstance(node, ast.Call):
            cn = _call_name(node)
            if (isinstance(node.func, ast.Name) and cn in DYNAMIC_CALLS) or cn in DYNAMIC_ATTRS:
                out.append(f"{rel}:{ln} dynamic code loading '{cn}'")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in SYS_PATH_MUTATORS
                and _is_sys_path(node.func.value)
                and rel not in SYS_PATH_EXEMPT
            ):
                out.append(f"{rel}:{ln} changes sys.path")
            if no_listing and cn in LISTING_CALLS:
                out.append(f"{rel}:{ln} lists a directory ('{cn}')")
            if cn == "interim_path" and check4 and rel != LOADER_MODULE:
                kinds = [_fold(a) for a in [*node.args, *(k.value for k in node.keywords)]]
                if "patterns" in kinds:
                    out.append(f"{rel}:{ln} reads the patterns file")
                elif id(node) not in safe_interim and (
                    "trans" in kinds or len(kinds) < 3 or kinds[-1] is None
                ):
                    out.append(f"{rel}:{ln} interim file not read via scan_transactions")
        if isinstance(node, ast.Assign | ast.AugAssign) and rel not in SYS_PATH_EXEMPT:
            tg = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_is_sys_path(t) for t in tg):
                out.append(f"{rel}:{ln} changes sys.path")
        # 2 + 4. string literals (folded; docstrings ignored)
        if isinstance(node, ast.Constant | ast.BinOp | ast.JoinedStr) and id(node) not in docs:
            if id(node) in seen_str:
                continue
            s = _fold(node)
            if s is None:
                continue
            for child in ast.walk(node):  # report the outermost folded expression only
                seen_str.add(id(child))
            if s in allowed:
                continue
            low = s.lower()
            if any(s.startswith(p) for p in FORBIDDEN_PREFIXES) or any(
                t in s for t in FORBIDDEN_PATH_TOKENS
            ):
                out.append(f"{rel}:{ln} string names an eval/generator module '{s[:60]}'")
            elif check4 and (
                any(t in low for t in TRUTH_TOKENS)
                or _norm(s) in TRUTH_EXACT_COLUMNS
                or (low in PATH_SEGMENTS and id(node) in segments)
                or any(f"/{g}/" in low or f"{UNKNOWN}{g}{UNKNOWN}" in low for g in PATH_SEGMENTS)
                or ("*" in low and any(x in low for x in DATA_SUFFIXES))
            ):
                out.append(f"{rel}:{ln} ground-truth / eval-store literal '{s[:60]}'")
    return out


def _violations(py: Path) -> list[str]:
    rel = str(py.relative_to(ROOT)) if py.is_relative_to(ROOT) else py.name
    pkg_parts = py.relative_to(ROOT).with_suffix("").parts[:-1] if py.is_relative_to(ROOT) else ()
    return _violations_in_source(py.read_text(encoding="utf-8"), rel, pkg_parts)


def _notebook_source(nb: Path) -> str:
    """Code cells as one Python source. IPython magics and shell escapes (%run, !cat ...) are not
    Python: each becomes a string literal holding the line, so its paths are still scanned."""
    cells = json.loads(nb.read_text(encoding="utf-8")).get("cells", [])
    lines = []
    for c in cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        for ln in src.splitlines():
            if ln.lstrip().startswith(("%", "!")):
                indent = ln[: len(ln) - len(ln.lstrip())]
                lines.append(f"{indent}_magic = {ln.strip()!r}")
            else:
                lines.append(ln)
        lines.append("")
    return "\n".join(lines)


def _exempt(p: Path) -> bool:
    rel = p.relative_to(ROOT).as_posix()
    return any(rel == d or rel.startswith(d + "/") for d in EXEMPT_DIRS)


def _scanned_py() -> list[Path]:
    files = [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]
    return sorted(p for p in files if not _exempt(p) and "__pycache__" not in p.parts)


def _notebooks() -> list[Path]:
    return sorted(
        p for p in (ROOT / "notebooks").rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts
    )


def test_runtime_packages_exist():
    for pkg in RUNTIME_PACKAGES:
        assert (ROOT / pkg / "__init__.py").exists(), pkg


def test_scan_covers_every_non_exempt_package():
    """New packages (src/features, src/models, ...) are covered automatically, not by a list."""
    scanned = {p.relative_to(ROOT).parts[:2] for p in _scanned_py()}
    for d in (ROOT / "src").iterdir():
        if d.is_dir() and (d / "__init__.py").exists() and not _exempt(d):
            assert ("src", d.name) in scanned, d.name
    assert ("scripts", "cost_model.py") in scanned


@pytest.mark.parametrize("py", _scanned_py(), ids=lambda p: str(p.relative_to(ROOT)))
def test_non_eval_code_cannot_reach_ground_truth(py: Path):
    assert _violations(py) == []


@pytest.mark.parametrize("nb", _notebooks() or [None], ids=lambda p: str(p) if p else "none")
def test_notebooks_cannot_reach_ground_truth(nb):
    if nb is None:
        pytest.skip("no notebooks yet")
    src = _notebook_source(nb)
    try:
        v = _violations_in_source(src, nb.relative_to(ROOT).as_posix())
    except SyntaxError as e:  # an unscannable notebook is a failure, not a pass
        pytest.fail(f"{nb.name}: cannot be scanned ({e})")
    assert v == []


PROBES = {
    "import src.eval\n": "imports",
    "from src.eval.labels import AlertLabel\n": "imports",
    "from ...eval import labels\n": "imports",  # from src/features/_probe -> src.eval
    "from scripts.audit import discover\n": "imports",
    "import scripts.p2.dispositions\n": "imports",
    "import importlib\nimportlib.import_module('src.eval.labels')\n": "dynamic",
    "import importlib\nm = importlib.import_module('src.' + 'eval' + '.labels')\n": "dynamic",
    "pkg = 'x'\nm = __import__(f'src.{pkg}.alert_labels')\n": "dynamic",
    "import runpy\nrunpy.run_path('src/eval/group_split.py')\n": "runpy",
    "import sys\nsys.path.insert(0, 'src')\n": "sys.path",
    "import eval.labels\n": "bare",
    "exec('from src.ev' + 'al import identity_guard')\n": "dynamic",
    "import polars as pl\n"
    "y = pl.read_parquet('data/p2/HI-Medium/eval/alert_labels.parquet')\n": "lit",
    "import polars as pl\nP = 'x'\ny = pl.read_parquet(P + '/' + 'eval' + '/a.parquet')\n": "lit",
    "c = 'is_' + 'laundering'\n": "lit",
    "c = 'Is Laundering'\n": "lit",
    "from pathlib import Path\np = Path('data') / 'devtools' / 'x.parquet'\n": "lit",
    "from pathlib import Path\np = Path('data') / 'eval'\n": "lit",
    "from src.data.transactions import interim_path\n"
    "p = interim_path('i', 'v', 'patterns')\n": "pat",
    "import polars as pl\nfrom src.data.transactions import interim_path\n"
    "t = pl.read_parquet(interim_path('i', 'v', 'trans'))\n": "trans",
    "cols = ['attempt_id']\n": "lit",
    "import polars as pl\nt = pl.scan_parquet('data/interim/*_trans.parquet')\n": "lit",
    "import polars as pl\nt = pl.scan_parquet('data/interim/*.parquet')\n": "glob read",
    "import polars as pl\nt = pl.scan_parquet(base + '/*.parquet')\n": "glob read",
    "t = pl.read_parquet('data\\\\interim\\\\HI-Medium_x.parquet')\n": "win path",
    "y = pl.read_parquet('data\\\\p2\\\\HI-Medium\\\\eval\\\\a.parquet')\n": "win",
    "import polars as pl\nt = pl.read_csv('data/raw/HI-Medium_Trans.csv')\n": "lit",
    "from pathlib import Path\nfs = list(Path('data/p2/HI-Medium').rglob('*.parquet'))\n": "list",
    "from pathlib import Path\np = Path('data') / 'archive' / 'x.parquet'\n": "archive",
    "import yaml\nc = yaml.safe_load(open('configs/p2_agent_split_v2.yaml'))\n": "cfg",
    "from src.data.transactions import interim_path\np = interim_path(i, v, k)\n": "var kind",
    "import importlib\nm = getattr(importlib, 'import_' + 'module')('x')\n": "importlib",
}
CLEAN = (
    '"""docstring may mention src/eval and is_laundering"""\n'
    "from src.contracts import Alert\n"
    "from src.data.transactions import interim_path, scan_transactions\n"
    "from src.data.stores import scan_runtime\n"
    "t = scan_transactions(interim_path('i', 'v', 'trans'))\n"
    "p = interim_path('i', 'v', 'trans')\n"
    "t2 = scan_transactions(p)\n"
    "t3 = scan_transactions(path=interim_path('i', 'v', 'trans'))\n"
    "a = scan_runtime('alerts.parquet')\n"
    "n = cfg['agent_split']['fraction_agent_test']\n"
    "out = {'typology': 'FAN-IN', 'typologies': ['CYCLE'], 'label': 'x'}\n"
    "def lookup_typology_definition(name): return name\n"
    "evals = [(dv, 'eval')]\n"
    "mode = 'patterns'\n"
    "import sys\n"
    "here = sys.path[0]\n"
    "from src.data.periods import ROOT\n"
    "interim = ROOT / 'data' / 'interim'\n"
)


def test_detector_catches_every_probe():
    """The guard itself must work: each known bypass (incl. the reviewer's probe) is caught,
    and ordinary runtime code is not flagged."""
    pkg = ("src", "features", "_probe")
    for code, kind in PROBES.items():
        assert _violations_in_source(code, "src/features/_probe/x.py", pkg), (kind, code)
    assert _violations_in_source(CLEAN, "src/features/_probe/ok.py", pkg) == []


def test_notebook_cells_are_scanned(tmp_path: Path):
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["src.eval is fine in markdown"]},
            {"cell_type": "code", "source": ["%matplotlib inline\n", "import src.eval.labels\n"]},
        ]
    }
    src = _notebook_source(_write(tmp_path / "n.ipynb", json.dumps(nb)))
    assert _violations_in_source(src, "notebooks/n.ipynb")
    magic = {"cells": [{"cell_type": "code", "source": ["%run src/eval/group_split.py\n"]}]}
    src = _notebook_source(_write(tmp_path / "m.ipynb", json.dumps(magic)))
    assert _violations_in_source(src, "notebooks/m.ipynb")
    ok = {"cells": [{"cell_type": "code", "source": ["%matplotlib inline\n", "x = 1\n"]}]}
    src = _notebook_source(_write(tmp_path / "ok.ipynb", json.dumps(ok)))
    assert _violations_in_source(src, "notebooks/ok.ipynb") == []


def _write(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------- runtime store access layer


def test_store_layer_refuses_eval_and_devtools(tmp_path: Path):
    from datetime import UTC, datetime

    import polars as pl

    from src.data import stores

    for bad in (
        "alert_labels.parquet",
        "agent_dev_alert_ids.parquet",
        "../eval/alert_labels.parquet",
        "../devtools/agent_dev_alert_ids.parquet",
    ):
        with pytest.raises(stores.StoreAccessError):
            stores.runtime_path(bad, "V", tmp_path)
    rt = tmp_path / "V" / "runtime"
    rt.mkdir(parents=True)
    t = lambda d, h=0: datetime(2022, 9, d, h, tzinfo=UTC)  # noqa: E731
    pl.DataFrame(
        {
            "alert_id": ["a", "b", "c"],
            "account_key": ["x", "y", "z"],
            "disposition": ["closed_legitimate"] * 3,
            "closed_at": [t(12, 23), t(13), t(14)],
        }
    ).write_parquet(rt / "dispositions.parquet")
    vis = stores.dispositions_visible(t(13), "V", tmp_path).collect()
    assert vis["alert_id"].to_list() == ["a"]  # closed_at == as_of is NOT visible (tie rule)
    with pytest.raises(ValueError):
        stores.dispositions_visible(datetime(2022, 9, 13), "V", tmp_path)
