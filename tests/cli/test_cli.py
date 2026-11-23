from importlib.metadata import version
from io import StringIO
from pathlib import Path

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


def test_replay_command_preserves_quiet_and_errors(tmp_path: "Path") -> None:
    config = tmp_path / "fixture.toml"
    config.write_text('version = 1\nfixture = "binary"\n')
    output = tmp_path / "report.json"
    code, stdout, stderr = invoke(
        [
            "--quiet",
            "replay",
            "--config",
            str(config),
            "--out",
            str(output),
        ]
    )
    assert code is None
    assert stdout == stderr == ""
    assert output.is_file()
    config.write_text('version = 1\nfixture = "binary"\nunknown = 1')
    code, stdout, stderr = invoke(
        [
            "replay",
            "--config",
            str(config),
            "--out",
            str(output),
            "--quiet",
        ]
    )
    assert code == 2
    assert stdout == ""
    assert "unknown configuration keys" in stderr


def test_oracle_command_creates_prior_report_and_preserves_existing_output(
    tmp_path: "Path",
) -> None:
    config = tmp_path / "oracle.toml"
    config.write_text(
        'version=1\nname="prior"\nmode="prior"\n'
        '[reference]\nfamily="beta_bernoulli"\nclasses=["H","T"]\n'
        'concentration=[1.0,1.0]\npositive_class="H"\n'
        "[observations]\nquery_x=[0.0]\n"
    )
    output = tmp_path / "report.json"
    args = ["oracle", "--config", str(config), "--out", str(output), "--quiet"]
    assert invoke(args) == (None, "", "")
    original = output.read_bytes()
    config.write_text(config.read_text().replace("[1.0,1.0]", "[2.0,1.0]"))
    code, stdout, stderr = invoke(args)
    assert code == 2
    assert stdout == ""
    assert "different run record" in stderr
    assert output.read_bytes() == original


def test_oracle_cli_retains_failed_reference_comparison(tmp_path: "Path") -> None:
    config = Path(__file__).resolve().parents[2] / "configs/oracles/coin-four.toml"
    expected = tmp_path / "wrong.json"
    expected.write_text('{"prediction.probabilities.0.0": 0.75}')
    output = tmp_path / "rejected.json"
    code, stdout, stderr = invoke(
        [
            "oracle",
            "--config",
            str(config),
            "--out",
            str(output),
            "--expected",
            str(expected),
            "--quiet",
        ]
    )
    assert code == 2
    assert stdout == ""
    assert "reference comparison failed" in stderr
    assert '"status": "FAIL"' in output.read_text()


def test_diagnostic_command_and_immutable_output(tmp_path: Path) -> None:
    config = tmp_path / "diagnostic.toml"
    config.write_text('version=1\nfamilies=["linear"]\nseeds=[11]\ndraws=1\n')
    output = tmp_path / "run"
    args = ["diagnose-priors", "--config", str(config), "--out", str(output), "--quiet"]
    assert invoke(args) == (None, "", "")
    assert (output / "report.json").is_file()
    code, stdout, stderr = invoke(args)
    assert code == 2
    assert stdout == ""
    assert "immutable" in stderr


def test_challenges_reports_pending_custody_and_respects_quiet(tmp_path: Path) -> None:
    import json
    from io import StringIO

    from r0.modules.cli import run

    output = tmp_path / "challenges.json"
    stdout, stderr = StringIO(), StringIO()
    run(["challenges", "--out", str(output), "--quiet"], stdout=stdout, stderr=stderr)
    assert stdout.getvalue() == stderr.getvalue() == ""
    report = json.loads(output.read_text())
    assert report["public_status"] == "PASS"
    assert report["sealed_status"] == "PENDING_OWNER_VERIFICATION"
