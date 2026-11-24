"""Identity-bound external predictor commands; no installation or asset discovery."""

import os
import subprocess
from pathlib import Path
from typing import cast

from .execution import THREAD_ENV, run_process
from .storage import Document, file_hash


def external_recipe(descriptor: Document) -> Document:
    required = {"name", "command", "identity", "files"}
    if set(descriptor) != required or descriptor["name"] not in ("learner", "tabpfn"):
        raise ValueError("invalid external predictor descriptor")
    identity = descriptor["identity"]
    if set(identity) != {"implementation", "environment", "settings", "checkpoint", "evidence"}:
        raise ValueError("incomplete adapter identity")
    if not all(identity.values()) or identity["evidence"] != "FIXTURE_ONLY":
        raise ValueError("real external execution requires the guarded admitted lane")
    command = descriptor["command"]
    if not command or not all(isinstance(arg, str) for arg in command):
        raise ValueError("external command must be an argument vector")
    executable = Path(command[0])
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("explicit provisioned interpreter required")
    if not descriptor["files"]:
        raise ValueError("external implementation/asset files must be hash-bound")
    # Only a file entrypoint or our identity-bound TabPFN worker is supported.
    # Arbitrary -c/-m code could hide an unbound implementation behind a label.
    if command[1:4] == ["-m", "r0.modules.comparison", "tabpfn-worker"]:
        arguments = command[4:]
        if (
            len(arguments) != 2
            or arguments[0] != "--config"
            or arguments[1] not in descriptor["files"]
        ):
            raise ValueError("TabPFN worker requires a hash-bound config")
    else:
        if len(command) < 2 or command[1] not in descriptor["files"]:
            raise ValueError("external entrypoint must be hash-bound")
        arguments = command[1:]
    for argument in arguments:
        value = argument.split("=", 1)[-1] if argument.startswith("--") else argument
        path = Path(value)
        if path.is_dir():
            raise ValueError("command directories are not a hash-bound file contract")
        if path.is_file() and (not path.is_absolute() or value not in descriptor["files"]):
            raise ValueError("every command file must be absolute and hash-bound")
    for name, expected in descriptor["files"].items():
        path = Path(name)
        if not path.is_absolute() or not path.is_file() or file_hash(path) != expected:
            raise ValueError("external file hash mismatch")
    return {**descriptor, "executable_sha256": file_hash(executable)}


def execute_external(
    descriptor: Document,
    input_path: Path,
    output: Path,
    *,
    diagnostic: bool,
    cgroup_root: Path | None,
) -> Document:
    external_recipe(descriptor)
    command = [
        *descriptor["command"],
        "--input",
        str(input_path.resolve()),
        "--out",
        str(output.resolve()),
    ]
    if not diagnostic:
        if cgroup_root is None:
            raise ValueError("enforced attempt requires delegated cgroup_root")
        return run_process(command, cwd=input_path.parent.resolve(), cgroup_root=cgroup_root)
    env = {
        key: value
        for key, value in os.environ.items()
        if key in ("PATH", "PYTHONPATH", "LANG", "LC_ALL")
    }
    env.update({key: "1" for key in THREAD_ENV})
    env.update(TABPFN_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        command,
        cwd=input_path.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return {
        "status": "COMPLETED" if result.returncode == 0 else "FAILED",
        "enforcement": "NOT_ENFORCED",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def validate_external_output(output: Document, view: Document, identity: Document) -> None:
    raw = output.get("query_ids")
    if not isinstance(raw, list):
        raise ValueError("external query IDs missing or malformed")
    if not all(isinstance(i, str) for i in cast(list[object], raw)):
        raise ValueError("external query IDs missing or malformed")
    ids = cast(list[str], raw)
    if len(ids) != len(set(ids)) or set(ids) != set(view["query_ids"]):
        raise ValueError("external query IDs differ from prepared input")
    if not isinstance(output.get("evidence"), dict):
        raise ValueError("external fit evidence missing or malformed")
    if output.get("version") != 1:
        raise ValueError("unsupported external output version")
    if output.get("backend_identity") != identity:
        raise ValueError("external backend identity mismatch")
    if output.get("environment") != identity["environment"]:
        raise ValueError("external runtime differs from frozen environment")
    if output.get("invariance") != "PASS":
        raise ValueError("external invariance not verified")
    if output.get("evidence", {}).get("fit_ids") != view["context_ids"]:
        raise ValueError("external fit IDs differ from context")
