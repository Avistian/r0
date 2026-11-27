"""Portable source and input identity checks for the independent root review."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SOURCE = "bc96787f5a423179b0b5b413f1d096ce81ef1a0f"
BASE = "8f3366a80bf7abb80f0d76e45e92031ea774234e"
BUNDLE = Path(__file__).resolve().parent
INPUTS = Path(os.environ.get("R0_REVIEW_INPUTS", BUNDLE / "inputs")).resolve()
HASHES = {
    "_points.py": "04dcbd7664004f6acc8453ca866caba0879d87d4ecbcdad08dd79f9ca2ff7a04",
    "_base.py": "065ed93d700fb125222ee60b68656eb419a39c630645d1b8f44fe9d6baf718d7",
    "I2func.py": "ff2939197b353f6d11a3fbaf0e27586998bdf93527dad11f24698ff296ce2f6c",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def prepare(priors):
    imported = Path(priors.__file__).resolve()
    root = imported.parents[4]
    assert imported == root / "src/r0/modules/priors/__init__.py"
    assert git(root, "rev-parse", "HEAD").decode().strip() == SOURCE
    output = Path(os.environ["R0_REVIEW_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = git(root, "ls-tree", "-r", "--name-only", SOURCE, "src/r0").decode().splitlines()
    paths += [
        "AGENTS.md",
        "CODING_STANDARDS.md",
        "docs/requirements.md",
        "docs/specs/0028-causal-network-generators.md",
        "docs/research/0028-joint-root-function-plans.md",
        "scripts/run_check.py",
    ]
    sources = {}
    for name in paths:
        actual = digest((root / name).read_bytes())
        assert actual == digest(git(root, "show", f"{SOURCE}:{name}")), name
        sources[name] = actual
    for name, expected in HASHES.items():
        assert digest((INPUTS / name).read_bytes()) == expected, name
    imports = {}
    for name, module in tuple(sys.modules.items()):
        if name == "r0" or name.startswith("r0."):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                assert path.is_relative_to(root / "src"), (name, path)
                imports[name] = {"path": str(path), "sha256": digest(path.read_bytes())}
    identity = {
        "source": SOURCE,
        "base": BASE,
        "checkout": str(root),
        "cwd": str(Path.cwd()),
        "status": git(root, "status", "--short").decode(),
        "sources": sources,
        "imports": imports,
        "inputs": HASHES,
        "input_directory": str(INPUTS),
        "upstream_refresh": "NOT_RUN",
        "python": sys.executable,
        "python_version": sys.version,
        "argv": sys.argv,
        "programs": {p.name: digest(p.read_bytes()) for p in BUNDLE.glob("*.py")},
        "environment": {
            k: os.environ.get(k)
            for k in (
                "R0_REVIEW_INPUTS",
                "R0_REVIEW_OUTPUT",
                "PYTHONPATH",
                "UV_CACHE_DIR",
                "UV_PROJECT_ENVIRONMENT",
            )
        },
    }
    (output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return output
