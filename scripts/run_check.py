"""Run verification without syncing or changing editable installations."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path


def editable_snapshot(environment: Path) -> dict[str, str]:
    """Cover uv path files, setuptools finders and PEP 610 editable provenance."""
    roots = list(environment.glob("lib*/python*/site-packages"))
    roots.append(environment / "Lib/site-packages")
    return {
        str(path.relative_to(environment)): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots
        for pattern in ("*.pth", "__editable__*.py", "*.dist-info/direct_url.json")
        for path in root.glob(pattern)
    }


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print("Usage: python3 scripts/run_check.py COMMAND [ARG ...]", file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parents[1]
    environment = Path(os.environ.get("UV_PROJECT_ENVIRONMENT", str(root / ".venv")))
    if not environment.is_absolute():
        environment = root / environment
    environment = environment.resolve()
    if not (environment / "pyvenv.cfg").is_file():
        print(f"FAIL: existing virtualenv required at {environment}; run make setup first.")
        return 1
    before = editable_snapshot(environment)
    process_environment = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(environment),
        "UV_NO_SYNC": "1",
        "PYTHONPATH": str(root / "src"),
    }
    result = subprocess.run(
        ["uv", "run", "--offline", "--no-sync", "--no-env-file", "--", *sys.argv[1:]],
        cwd=root,
        env=process_environment,
        check=False,
    )
    after = editable_snapshot(environment)
    if after != before:
        changed = sorted(
            key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
        )
        print("FAIL: verification changed editable-install metadata; inspect the environment:")
        print("\n".join(changed))
        return 1
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
