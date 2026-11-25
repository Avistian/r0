"""Retain ordinary tasks before the correction, then check unchanged replay afterward."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from r0.modules.priors import (
    FourierRecipe,
    LinearRecipe,
    PolynomialRecipe,
    generate_task,
    recipe_from_dict,
)


def main(mode: str, out: Path) -> None:
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src"], check=False).returncode:
        raise ValueError("commit source changes before retaining revision-labelled evidence")
    if mode == "baseline":
        out.mkdir(parents=True, exist_ok=False)
        cases = []
        for family in ("rff", "periodic", "polynomial"):
            for kind, classes in (("regression", 0), ("binary", 2), ("multiclass", 3)):
                for seed in (3, 27):
                    controls = LinearRecipe(task_type=kind, classes=classes)
                    recipe = (
                        PolynomialRecipe(task=controls, calibration_rows=256)
                        if family == "polynomial"
                        else FourierRecipe(
                            task=controls, family=family, terms=8, calibration_rows=256
                        )
                    )
                    task = generate_task(recipe, seed=seed, task_id=7)
                    name = f"{family}-{kind}-{seed}"
                    (out / f"{name}.json").write_text(
                        json.dumps(
                            {
                                "recipe": asdict(recipe),
                                "seed": seed,
                                "task_id": 7,
                                "mechanism_json": task.mechanism_json,
                                "source_task_id": task.source_task_id,
                                "episode_id": task.episode_id,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    np.savez(
                        out / f"{name}.npz",
                        inputs=task.inputs.numpy(),
                        scores=task.scores.numpy(),
                        targets=task.targets.numpy(),
                        probabilities=np.array([])
                        if task.probabilities is None
                        else task.probabilities.numpy(),
                    )
                    cases.append(name)
        (out / "baseline.json").write_text(
            json.dumps(
                {
                    "source": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "cases": cases,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Retained {len(cases)} baseline tasks")
        return
    baseline = json.loads((out / "baseline.json").read_text())
    for name in baseline["cases"]:
        record = json.loads((out / f"{name}.json").read_text())
        task = generate_task(
            recipe_from_dict(record["recipe"]), seed=record["seed"], task_id=record["task_id"]
        )
        assert task.mechanism_json == record["mechanism_json"]
        assert task.source_task_id == record["source_task_id"]
        assert task.episode_id == record["episode_id"]
        with np.load(out / f"{name}.npz", allow_pickle=False) as arrays:
            for key in ("inputs", "scores", "targets"):
                actual = getattr(task, key).numpy()
                expected = arrays[key]
                assert actual.dtype == expected.dtype
                assert actual.shape == expected.shape
                assert actual.tobytes() == expected.tobytes(), (name, key)
            if task.probabilities is not None:
                actual = task.probabilities.numpy()
                expected = arrays["probabilities"]
                assert actual.dtype == expected.dtype
                assert actual.shape == expected.shape
                assert actual.tobytes() == expected.tobytes()
    result = {
        "baseline_source": baseline["source"],
        "verification_source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": len(baseline["cases"]),
        "bitwise_arrays_and_exact_cards": "PASS",
    }
    with (out / "verified.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "verify"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.mode, args.out)
