"""Run three tasks, retain an authored failure, resume, then explicitly retry it."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from r0.modules.generation import RunRequest, run_generation
from r0.modules.priors import LinearRecipe, TrainingTask, generate_linear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a fresh output directory; evidence is never overwritten")
    recipe = LinearRecipe()
    request = RunRequest(recipe, seed=4, count=3, group_size=2, max_seconds=20, max_bytes=2_000_000)

    def fail_one(task_id: int) -> TrainingTask:
        if task_id == 1:
            raise ValueError("authored walkthrough failure")
        return generate_linear(recipe, seed=4, task_id=task_id)

    first = run_generation(request, args.out, produce=fail_one)
    resumed = run_generation(request, args.out, produce=fail_one)
    retried = run_generation(request, args.out, retry_failed=True)
    assert (first.attempted, first.completed, first.failed) == (3, 2, 1)
    assert (resumed.attempted, resumed.completed, resumed.failed) == (0, 2, 1)
    assert (retried.attempted, retried.completed, retried.failed) == (1, 3, 0)
    assert "authored walkthrough failure" in (args.out / "groups/0/attempt-0.json").read_text()
    result = {
        name: asdict(value)
        for name, value in (("first", first), ("resume", resumed), ("explicit_retry", retried))
    }
    result["execution"] = {
        "device": "cpu",
        "hosted": "NOT_RUN",
        "hard_timeout": "NOT_IMPLEMENTED_IN_THIS_SLICE",
    }
    (args.out / "walkthrough.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
