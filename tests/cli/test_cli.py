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


def test_explicit_information_prints() -> None:
    for flag in ["--help", "-h", "--version"]:
        for argv in [["--quiet", flag], [flag, "--quiet"]]:
            code, stdout, stderr = invoke(argv)
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
