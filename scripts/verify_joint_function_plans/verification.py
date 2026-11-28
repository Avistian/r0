"""Pin review inputs and verify every priors source byte in the current checkout."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SOURCE = "a6ab8c70902d9d474cea12e7cab2aadf67ca35f5"
BASE = "731f6a44b309a75fc5c5f756168550eac48b372a"
BUNDLE = Path(__file__).resolve().parent
INPUTS = Path(os.environ.get("R0_REVIEW_INPUTS", str(BUNDLE / "inputs"))).resolve()
BASE_INPUT_SHA = "533b367b3b5d82abbeb8f445afe9e29d0940f9044c88b7f3f5cd23af7a8963c2"
INPUT_HASHES = {
    "I2func.py": "ff2939197b353f6d11a3fbaf0e27586998bdf93527dad11f24698ff296ce2f6c",
    "_base.py": "065ed93d700fb125222ee60b68656eb419a39c630645d1b8f44fe9d6baf718d7",
    "base_parameter_preferences.txt": BASE_INPUT_SHA,
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def prepare(script, priors):
    root = Path.cwd().resolve()
    if Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise RuntimeError("Run from the reconstructed repository root")
    imported = Path(priors.__file__).resolve()
    if imported != root / "src/r0/modules/priors/__init__.py":
        raise RuntimeError("Imported priors is not the current checkout src package")
    output = Path(os.environ["R0_REVIEW_OUTPUT"]).resolve()
    # Refuse to mix new proof with an earlier execution.
    output.mkdir(parents=True, exist_ok=False)
    names = (
        git(root, "ls-tree", "-r", "--name-only", SOURCE, "src/r0/modules/priors")
        .decode()
        .splitlines()
    )
    names += [
        "AGENTS.md",
        "CODING_STANDARDS.md",
        "docs/requirements.md",
        "docs/specs/0028-causal-network-generators.md",
        "docs/research/0028-joint-root-function-plans.md",
        "scripts/run_check.py",
        "tests/priors/test_joint_function_plans.py",
        "tests/priors/test_source_integer_law.py",
    ]
    source_files = {}
    for name in names:
        expected = digest(git(root, "show", SOURCE + ":" + name))
        actual = digest((root / name).read_bytes())
        if actual != expected:
            raise RuntimeError("Checkout bytes differ from pinned SOURCE: " + name)
        source_files[name] = actual
    for name, expected in INPUT_HASHES.items():
        if digest((INPUTS / name).read_bytes()) != expected:
            raise RuntimeError("Retained input bytes differ: " + name)
    metadata = {
        "source": SOURCE,
        "base": BASE,
        "cwd": str(root),
        "observed_head": git(root, "rev-parse", "HEAD").decode().strip(),
        "git_status": git(root, "status", "--short").decode(),
        "public_import": str(imported),
        "source_files": source_files,
        "cached_inputs": INPUT_HASHES,
        "input_directory": str(INPUTS),
        "upstream_refresh": "NOT_RUN",
        "python": sys.executable,
        "python_version": sys.version,
        "argv": sys.argv,
        "output": str(output),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "UV_CACHE_DIR",
                "UV_PROJECT_ENVIRONMENT",
                "PYTHONPATH",
                "R0_REVIEW_OUTPUT",
                "R0_REVIEW_INPUTS",
            )
        },
        "programs": {
            Path(name).name: digest(Path(name).read_bytes()) for name in (script, __file__)
        },
    }
    (output / "identity.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("IDENTITY", json.dumps(metadata))
    return output, metadata
