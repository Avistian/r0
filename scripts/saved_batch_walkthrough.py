"""Build a tiny saved group, then consume it with regeneration forbidden."""

import argparse
import json
import runpy
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from r0.modules.generation import save_group
from r0.modules.priors import LinearRecipe, generate_task


def forbidden(*args, **kwargs):
    raise AssertionError("saved consumption called generate_task")


def main(out: Path):
    out.mkdir(parents=True, exist_ok=False)

    def tasks():
        for task_id in range(7):
            kind, classes = (("regression", 0), ("binary", 2), ("multiclass", 100))[task_id % 3]
            yield (
                task_id,
                generate_task(
                    LinearRecipe(rows=32, context_rows=20, task_type=kind, classes=classes),
                    seed=91,
                    task_id=task_id,
                ),
            )

    archive = out / "source-group.zip"
    receipt = save_group(archive, tasks(), max_bytes=1024 * 1024)
    (out / "saved-source.json").write_text(json.dumps(asdict(receipt), indent=2) + "\n")
    sys.argv = [
        "consume_generated_batches.py",
        "--archive",
        str(archive),
        "--count",
        "30",
        "--batch-size",
        "4",
        "--out",
        str(out / "consumption"),
    ]
    with patch("r0.modules.priors.generate_task", side_effect=forbidden):
        runpy.run_path(
            str(Path(__file__).with_name("consume_generated_batches.py")), run_name="__main__"
        )
    summary = json.loads((out / "consumption" / "summary.json").read_text())
    assert summary["consumed"] == 7
    assert summary["batches"] == 2
    (out / "no-regeneration.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "generate_task_disabled_during_consumption": True,
                "consumed": 7,
                "batches": 2,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
