"""Parse CLI requests and apply the console output policy."""

import argparse
from importlib.metadata import version
from typing import TYPE_CHECKING, NoReturn, TextIO

if TYPE_CHECKING:
    from _typeshed import SupportsWrite


class _Parser(argparse.ArgumentParser):
    def __init__(self, stdout: TextIO, stderr: TextIO) -> None:
        super().__init__(prog="r0", add_help=False)
        self.stdout = stdout
        self.stderr = stderr
        self.quiet = False

    def _print_message(self, message: str, file: "SupportsWrite[str] | None" = None) -> None:
        if message and (file is not None or not self.quiet):
            (self.stdout if file is None else file).write(message)

    def print_help(self, file: "SupportsWrite[str] | None" = None) -> None:
        self._print_message(self.format_help(), file)

    def print_usage(self, file: "SupportsWrite[str] | None" = None) -> None:
        self._print_message(self.format_usage(), file)

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        if message:
            self.stderr.write(message)
        raise SystemExit(status)

    def error(self, message: str) -> NoReturn:
        self.print_usage(file=self.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def run(argv: list[str], *, stdout: TextIO, stderr: TextIO) -> None:
    """Run with supplied arguments and streams; quiet preserves only errors."""
    parser = _Parser(stdout, stderr)
    parser.add_argument("--quiet", action="store_true", help="suppress all output except errors")
    # Resolve quiet before argparse's help/version actions can print and exit.
    options, _ = parser.parse_known_args(argv)
    parser.quiet = options.quiet
    parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    parser.add_argument(
        "--version", action="store_true", help="show program's version number and exit"
    )
    args = parser.parse_args(argv)
    if not args.quiet:
        print(version("r0") if args.version else "r0", file=stdout)
    if args.version:
        parser.exit()
