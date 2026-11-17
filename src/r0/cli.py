"""Console entry point for r0."""

import sys

from r0.modules.cli import run


def main(argv: list[str] | None = None) -> None:
    """Run the CLI with explicit arguments or the process arguments."""
    run(argv, stdout=sys.stdout, stderr=sys.stderr)
