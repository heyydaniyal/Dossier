"""Save and load P4 models BY HASH (P4 task 11; the Scoring tool loads models by hash only).

One JSON file per model: models/p4/model_<first 16 hex of sha256>.json. The hash is the sha256 of
the file bytes (= ScoreResult.model_version_hash). Loading recomputes it and refuses any mismatch,
so an edited or swapped file can never be served under an old hash. Runtime-safe: no labels.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.data.periods import ROOT
from src.models import families

MODEL_DIR = ROOT / "models" / "p4"
_HEX = re.compile(r"^[0-9a-f]{64}$")


class ModelHashError(ValueError):
    """The requested hash is malformed, missing, or does not match the file's bytes."""


def serialise(model, meta: dict) -> bytes:
    doc = {"format": "dossier-model-v1", "meta": meta, "model": model.to_dict()}
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8")


def save(model, meta: dict, directory: Path = MODEL_DIR, max_bytes: int | None = None) -> str:
    b = serialise(model, meta)
    if max_bytes is not None and len(b) > max_bytes:
        raise ModelHashError(f"model file {len(b)} bytes exceeds {max_bytes}")
    h = hashlib.sha256(b).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"model_{h[:16]}.json").write_bytes(b)
    return h


def load(sha256: str, directory: Path = MODEL_DIR):
    """(model, meta) for a full 64-hex sha256; refuses prefixes, missing files and mismatches."""
    if not _HEX.match(sha256 or ""):
        raise ModelHashError("a full lowercase 64-hex sha256 is required")
    p = Path(directory) / f"model_{sha256[:16]}.json"
    if not p.is_file():
        raise ModelHashError(f"no model file for {sha256[:16]}")
    b = p.read_bytes()
    if hashlib.sha256(b).hexdigest() != sha256:
        raise ModelHashError(f"{p.name}: bytes do not match the requested hash")
    doc = json.loads(b.decode("utf-8"))
    if doc.get("format") != "dossier-model-v1":
        raise ModelHashError(f"{p.name}: unknown format")
    return families.from_dict(doc["model"]), doc["meta"]
