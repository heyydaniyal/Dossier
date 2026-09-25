"""Global rule 13: secrets never enter the repository."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_PATTERNS = [re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), re.compile(r"sk-[A-Za-z0-9]{20,}")]
SKIP = {".venv", ".git", "__pycache__", ".ruff_cache", ".pytest_cache"}


def test_env_is_gitignored():
    lines = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in lines and "!.env.example" in lines


def test_no_api_keys_in_files():
    hits = []
    for p in ROOT.rglob("*"):
        if p.is_file() and not SKIP.intersection(p.parts) and p.stat().st_size < 2_000_000:
            text = p.read_text(errors="ignore")
            hits += [str(p) for rx in KEY_PATTERNS if rx.search(text)]
    assert hits == []


def test_dotenv_not_tracked():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout
    except FileNotFoundError:
        return
    assert not any(Path(f).name == ".env" for f in out.splitlines())
