"""Headless generation must persist real, reloadable datasets and resume them."""

import json
from dataclasses import asdict
from io import StringIO
from pathlib import Path

from r0.modules.cli import run
from r0.modules.generation import load_group, load_request, verify_group


def invoke(argv: list[str]) -> tuple[int | str | None, str, str]:
    stdout, stderr = StringIO(), StringIO()
    code = None
    try:
        run(argv, stdout=stdout, stderr=stderr)
    except SystemExit as error:
        code = error.code
    return code, stdout.getvalue(), stderr.getvalue()


def test_generate_save_reload_resume(tmp_path: Path) -> None:
    out = tmp_path / "datasets"
    argv = ["generate", "--config", "configs/generation/polynomial-cpu.toml", "--out", str(out)]
    code, stdout, stderr = invoke(argv)
    assert code is None, stderr
    report = json.loads(stdout)
    assert report["completed"] == report["generated"] == 8
    archives = sorted(out.glob("groups/**/*.zip"))
    assert sum(len(verify_group(path).task_ids) for path in archives) == 8
    assert sum(len(list(load_group(path))) for path in archives) == 8
    original = {path: path.read_bytes() for path in archives}
    config = tmp_path / "request.json"
    config.write_text(json.dumps(asdict(load_request(Path(argv[2])))))
    argv[2] = str(config)
    code, stdout, stderr = invoke(argv)
    assert code is None, stderr
    assert json.loads(stdout)["generated"] == 0
    assert original == {path: path.read_bytes() for path in archives}


def test_generate_invalid_config_and_timeout(tmp_path: Path) -> None:
    config = tmp_path / "request.json"
    config.write_text(json.dumps({"max_seconds": 0.000001}))
    args = ["generate", "--config", str(config), "--out", str(tmp_path / "out")]
    code, stdout, stderr = invoke([*args, "--quiet"])
    assert code == 1
    assert stdout == ""
    assert "timed out" in stderr
    config.write_text("[]")
    code, _, stderr = invoke(args)
    assert code == 2
    assert "object" in stderr
