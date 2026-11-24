"""Exercise verification through Make with a real borrowed uv environment."""

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def project(path: Path) -> None:
    (path / "src/r0").mkdir(parents=True)
    (path / "src/r0/__init__.py").write_text("")
    (path / "pyproject.toml").write_text("""[project]
name = "r0"
version = "0.0.1"
requires-python = ">=3.12"
[build-system]
requires = ["uv_build>=0.9.21,<0.10.0"]
build-backend = "uv_build"
""")


def editable_files(environment: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(environment)): p.read_bytes()
        for pattern in ("**/*.pth", "**/*.dist-info/direct_url.json")
        for p in environment.glob(pattern)
    }


def test_make_preflight_preserves_borrowed_editable_install(tmp_path: Path) -> None:
    original, checkout, shared = (tmp_path / n for n in ("original", "checkout", "shared"))
    project(original)
    project(checkout)
    venv.EnvBuilder(with_pip=False).create(shared)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--offline",
            "--no-deps",
            "--python",
            str(shared / "bin/python"),
            "--editable",
            str(original),
        ],
        check=True,
    )
    shutil.copy(ROOT / "Makefile", checkout / "Makefile")
    shutil.copytree(ROOT / "scripts", checkout / "scripts")
    before = editable_files(shared)
    assert before
    result = subprocess.run(
        ["make", "preflight"],
        cwd=checkout,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CI": "true",
            "UV_OFFLINE": "1",
            "UV_PROJECT_ENVIRONMENT": str(shared),
            "PYTHONPATH": str(checkout / "src"),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(checkout / "src/r0/__init__.py") in result.stdout
    assert editable_files(shared) == before


def test_verification_detects_editable_metadata_mutation(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    venv.EnvBuilder(with_pip=False).create(shared)
    metadata = next(shared.glob("lib/python*/site-packages")) / "fixture.pth"
    metadata.write_text("# original\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_check.py"),
            "python",
            "-c",
            'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("# changed\\n")',
            str(metadata),
        ],
        cwd=ROOT,
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(shared)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "verification changed editable-install metadata" in result.stdout
    assert "fixture.pth" in result.stdout


def test_preflight_requires_local_environment_unless_explicitly_reused(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    project(checkout)
    shutil.copy(ROOT / "Makefile", checkout / "Makefile")
    shutil.copytree(ROOT / "scripts", checkout / "scripts")
    environment = dict(os.environ)
    environment.pop("UV_PROJECT_ENVIRONMENT", None)
    environment["VIRTUAL_ENV"] = sys.prefix
    result = subprocess.run(
        ["make", "preflight"], cwd=checkout, env=environment, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "existing virtualenv required" in result.stdout
    assert str(checkout / ".venv") in result.stdout
    assert not (checkout / ".venv").exists()
