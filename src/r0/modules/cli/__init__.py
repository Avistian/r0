"""Command-line interface for r0."""

import argparse
from importlib.metadata import version


def main(argv: list[str] | None = None) -> None:
    """Run the CLI with explicit arguments or the process arguments."""
    parser = argparse.ArgumentParser(prog="r0")
    parser.add_argument("--version", action="version", version=version("r0"))
    parser.parse_args(argv)
    print("r0")
