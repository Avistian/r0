"""Exercise repository guards in subprocesses with deliberately invalid state."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_checkout_rejects_foreign_package(tmp_path: Path) -> None:
    package = tmp_path / "r0"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for source, expected_code in ((tmp_path, 1), (ROOT / "src", 0)):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_checkout.py")],
            env={**os.environ, "PYTHONPATH": str(source)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == expected_code, result.stdout + result.stderr
        assert str(source / "r0" / "__init__.py") in result.stdout


def test_bytecode_guard_reads_index_including_deleted_working_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("*.pyc\n")
    bytecode = tmp_path / "bad file.pyc"
    bytecode.write_bytes(b"generated")
    subprocess.run(["git", "add", "-f", bytecode.name], cwd=tmp_path, check=True)
    bytecode.unlink()
    command = [sys.executable, str(ROOT / "scripts/check_tracked_files.py")]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert bytecode.name in result.stdout
    subprocess.run(["git", "rm", "--cached", bytecode.name], cwd=tmp_path, check=True)
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0
