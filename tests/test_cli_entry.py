import subprocess
import sys
import tomllib
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from r0.cli import main


def test_no_arguments_preserves_greeting() -> None:
    stdout, stderr = StringIO(), StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        main([])
    assert stdout.getvalue() == "r0\n"
    assert stderr.getvalue() == ""


def test_version_matches_pyproject() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as source:
        expected = tomllib.load(source)["project"]["version"]

    result = subprocess.run(
        [str(Path(sys.executable).with_name("r0")), "--version"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == f"{expected}\n"
    assert result.stderr == ""


def test_quiet_console_entry() -> None:
    result = subprocess.run(
        [str(Path(sys.executable).with_name("r0")), "--quiet"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
