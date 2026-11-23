"""Require the repository's executable commit hook during local preflight."""

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if os.environ.get("CI") == "true":
        print("CI runs repository checks directly; local hook activation is not required.")
        return 0
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    active = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-path", "hooks/pre-commit"],
            cwd=root,
            text=True,
        ).strip()
    )
    expected = root / ".githooks/pre-commit"
    if (
        active.resolve() != expected.resolve()
        or not expected.is_file()
        or not os.access(expected, os.X_OK)
    ):
        print("FAIL: repository commit hook is inactive or not executable. Run make hooks.")
        print(f"Resolved hook: {active}")
        return 1
    print(f"Active commit hook: {active}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
