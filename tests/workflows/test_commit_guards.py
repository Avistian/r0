"""Guard behavior against real Git configuration and staged blobs."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_hook_setup_and_local_preflight(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    shutil.copytree(ROOT / ".githooks", tmp_path / ".githooks")
    command = [sys.executable, str(ROOT / "scripts/check_hooks.py")]
    environment = {**os.environ, "CI": ""}
    missing = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True)
    assert missing.returncode == 1
    assert b"make hooks" in missing.stdout
    subprocess.run(["make", "-f", str(ROOT / "Makefile"), "hooks"], cwd=tmp_path, check=True)
    active = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True)
    assert active.returncode == 0, active.stdout + active.stderr
    (tmp_path / ".githooks/pre-commit").chmod(0o644)
    disabled = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True)
    assert disabled.returncode == 1
    subprocess.run(["make", "-f", str(ROOT / "Makefile"), "hooks"], cwd=tmp_path, check=True)
    repaired = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True)
    assert repaired.returncode == 0
    ci = subprocess.run(
        command, cwd=tmp_path, env={**environment, "CI": "true"}, capture_output=True
    )
    assert ci.returncode == 0
    assert b"CI" in ci.stdout


def test_size_guard_checks_index_not_working_tree(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    artifact = tmp_path / "large artifact.bin"
    artifact.write_bytes(b"x" * (1048576 + 1))
    subprocess.run(["git", "add", artifact.name], cwd=tmp_path, check=True)
    artifact.write_bytes(b"small unstaged replacement")
    command = [sys.executable, str(ROOT / "scripts/check_artifact_sizes.py")]
    large = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert large.returncode == 1
    assert artifact.name in large.stdout
    assert "1048577" in large.stdout
    artifact.unlink()
    deleted = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert deleted.returncode == 1
    artifact.write_bytes(b"x" * 1048576)
    subprocess.run(["git", "add", artifact.name], cwd=tmp_path, check=True)
    allowed = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
