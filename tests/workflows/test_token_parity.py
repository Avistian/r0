"""The Make/CI and commit-hook paths must reject the same indexed artifacts."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_json_token_limit_matches_make_and_hook(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in ("src", "tests", "scripts"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("Makefile", "pyproject.toml"):
        shutil.copy(ROOT / name, tmp_path / name)
    (tmp_path / ".venv").symlink_to(sys.prefix, target_is_directory=True)
    artifact = tmp_path / "large evidence.json"
    artifact.write_text('"' + "x" * 20000 + '"\n')
    subprocess.run(["git", "add", artifact.name], cwd=tmp_path, check=True)
    environment = {
        **os.environ,
        "UV_OFFLINE": "1",
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / ".venv"),
    }
    for command in (["make", "tokens"], ["sh", "scripts/pre-commit.sh"]):
        result = subprocess.run(
            command, cwd=tmp_path, env=environment, capture_output=True, text=True
        )
        assert result.returncode != 0, result.stdout + result.stderr
        assert artifact.name in result.stdout
        assert "5000 tokens" in result.stdout


def test_token_guard_reads_index_and_preserves_existing_exemptions(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    artifact = tmp_path / "evidence with spaces.json"
    artifact.write_text("x" * 20001)
    subprocess.run(["git", "add", artifact.name], cwd=tmp_path, check=True)
    command = [sys.executable, str(ROOT / "scripts/check_file_tokens.py")]
    artifact.write_text("{}")
    hidden = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert hidden.returncode == 1
    assert artifact.name in hidden.stdout
    artifact.unlink()
    deleted = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert deleted.returncode == 1
    artifact.write_text("x" * 20000)
    subprocess.run(["git", "add", artifact.name], cwd=tmp_path, check=True)
    for name, content in (("binary.bin", b"\0" + b"x" * 20000), ("uv.lock", b"x" * 20001)):
        (tmp_path / name).write_bytes(content)
        subprocess.run(["git", "add", name], cwd=tmp_path, check=True)
    allowed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
