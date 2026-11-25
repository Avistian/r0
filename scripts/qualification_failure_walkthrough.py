"""Authored interruption receipts and a real killed worker check reporting boundaries."""

import argparse
from pathlib import Path

from qualify_generators import execute, summarize, write


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    directory = out / "damaged-timeout"
    directory.mkdir()
    (directory / "worker.json").write_text('{"incomplete":')
    timeout = execute(directory, 1, 128 * 1024**2, "timeout")
    assert timeout["status"] == "TIMEOUT"
    assert "worker" in timeout["receipt_errors"]
    assert (directory / "result.json").exists()
    fixture = out / "authored-counts"
    fixture.mkdir()
    cases = [
        {
            "id": name,
            "family": "linear",
            "size": "small",
            "task_type": "regression",
            "request": {"recipe": {"rows": 64, "features": 8, "active_features": 4, "classes": 0}},
        }
        for name in ("never-launched", "unknown-launch", "confirmed-launch")
    ]
    write(
        fixture / "manifest.json",
        {
            "source": "AUTHORED_REPORTING_FIXTURE",
            "suite": "not-a-generation-measurement",
            "cases": cases,
        },
    )
    for name, record in (
        ("unknown-launch", "launch-request"),
        ("confirmed-launch", "worker-started"),
    ):
        location = fixture / name
        location.mkdir()
        write(location / f"{record}.json", {"kind": "authored fixture; no process launched"})
    report = summarize(fixture)
    case = report["cases"][0]
    assert case["planned"] == 3
    assert case["confirmed_launched"] == 1
    assert case["launch_unknown"] == 1
    assert case["outcomes"]["NOT_RUN"] == 1
    assert case["outcomes"]["INCOMPLETE"] == 1
    assert case["outcomes"]["LAUNCH_UNKNOWN"] == 1
    assert case["qualification"] == "NOT_QUALIFIED"
    assert case["extrapolated_100m_generation_days"] is None
    write(out / "report.json", {"status": "PASS", "timeout": timeout, "authored_counts": report})
    print("PASS: damaged timeout receipt and distinct planned/unknown/confirmed counts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
