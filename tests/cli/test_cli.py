from importlib.metadata import version
from io import StringIO

from r0.modules.cli import run


def invoke(argv: list[str]) -> tuple[int | str | None, str, str]:
    stdout, stderr = StringIO(), StringIO()
    code = None
    try:
        run(argv, stdout=stdout, stderr=stderr)
    except SystemExit as error:
        code = error.code
    return code, stdout.getvalue(), stderr.getvalue()


def test_normal_output() -> None:
    assert invoke([]) == (None, "r0\n", "")
    assert invoke(["--quiet"]) == (None, "", "")
    assert invoke(["--quiet", "--quiet"]) == (None, "", "")


def test_hello_output() -> None:
    assert invoke(["--hello"]) == (None, "hello\n", "")


def test_quiet_suppresses_hello() -> None:
    for argv in [["--quiet", "--hello"], ["--hello", "--quiet"]]:
        assert invoke(argv) == (None, "", "")


def test_explicit_information_prints() -> None:
    for flag in ["--help", "-h", "--version"]:
        code, stdout, stderr = invoke([flag])
        assert code == 0
        if flag == "--version":
            assert stdout == f"{version('r0')}\n"
        else:
            assert "usage: r0" in stdout
            assert "--quiet" in stdout
        assert stderr == ""


def test_quiet_preserves_errors() -> None:
    for argv in [["--quiet", "--invalid"], ["--invalid", "--quiet"]]:
        code, stdout, stderr = invoke(argv)
        assert code == 2
        assert stdout == ""
        assert "unrecognized arguments: --invalid" in stderr


def test_quiet_suppresses_information() -> None:
    for flag in ["--help", "-h", "--version"]:
        for argv in [["--quiet", flag], [flag, "--quiet"]]:
            assert invoke(argv) == (0, "", "")


def test_streams_are_not_replaced() -> None:
    import sys

    original_stdout, original_stderr = sys.stdout, sys.stderr

    class ObservedStream(StringIO):
        def write(self, text: str) -> int:
            assert sys.stdout is original_stdout
            assert sys.stderr is original_stderr
            return super().write(text)

    run([], stdout=ObservedStream(), stderr=ObservedStream())
