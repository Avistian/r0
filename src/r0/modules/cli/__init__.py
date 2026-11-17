"""Parse CLI requests and apply the console output policy."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version
from typing import TextIO


def run(argv: list[str] | None, *, stdout: TextIO, stderr: TextIO) -> None:
    """Run with supplied streams; quiet preserves help, version, and errors."""
    with redirect_stdout(stdout), redirect_stderr(stderr):
        parser = argparse.ArgumentParser(prog="r0")
        parser.add_argument("--version", action="version", version=version("r0"))
        parser.add_argument(
            "--quiet", action="store_true", help="suppress normal output (except help and version)"
        )
        args = parser.parse_args(argv)
        if not args.quiet:
            print("r0", file=stdout)
