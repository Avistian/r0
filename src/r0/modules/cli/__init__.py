"""Parse CLI requests and apply the console output policy."""

import argparse
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, TextIO, cast

from r0.modules.replay import write_oracle_report, write_report

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
    parser.add_argument("--hello", action="store_true", help="print hello")
    parser.add_argument("command", nargs="?", choices=("replay", "oracle"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--expected", type=Path, help="independent numeric reference card for oracle"
    )
    args = parser.parse_args(argv)
    if args.command in ("replay", "oracle"):
        if args.version or args.hello:
            parser.error(f"{args.command} cannot be combined with --version or --hello")
        if args.config is None or args.out is None:
            parser.error(f"{args.command} requires --config and --out")
        if args.command == "replay" and args.expected is not None:
            parser.error("--expected requires oracle")
        try:
            if args.command == "oracle":
                report = write_oracle_report(args.config, args.out, args.expected)
                verification = report["verification"]
                if (
                    isinstance(verification, dict)
                    and cast(dict[str, object], verification).get("status") == "FAIL"
                ):
                    parser.error(f"reference comparison failed; retained report at {args.out}")
            else:
                write_report(args.config, args.out)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if not args.quiet:
            print(f"Wrote {args.command} report to {args.out}", file=stdout)
        return
    if args.expected is not None:
        parser.error("--expected requires oracle")
    if args.config is not None or args.out is not None:
        parser.error("--config and --out require replay or oracle")
    if not args.quiet:
        greeting = "hello" if args.hello else "r0"
        print(version("r0") if args.version else greeting, file=stdout)
    if args.version:
        parser.exit()
