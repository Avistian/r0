"""Inspect new finite edge cases and keep invalid arithmetic failures visible."""

import argparse
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from r0.modules.priors import (
    Combination,
    CompositeRecipe,
    FourierRecipe,
    FunctionComponent,
    InputDistribution,
    LinearRecipe,
    PolynomialRecipe,
    explain_generation,
    generate_task,
    recipe_from_dict,
)


def run(out: Path) -> None:
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src"], check=False).returncode:
        raise ValueError("commit source changes before retaining revision-labelled evidence")
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for kind, classes in (("regression", 0), ("binary", 2), ("multiclass", 3)):
        task = LinearRecipe(task_type=kind, classes=classes)
        for family in ("rff", "periodic"):
            for length in (1e13, 1e200):
                recipe = FourierRecipe(
                    task=task, family=family, terms=8, length_scale=length, calibration_rows=256
                )
                cases.append((f"{family}-{kind}-{length}", recipe, 27))
        recipe = PolynomialRecipe(
            task=replace(task, features=128, signal_features=128, active_features=1),
            inputs=InputDistribution(scale=5e18),
            degree=16,
            calibration_rows=256,
        )
        cases.append((f"polynomial-{kind}", recipe, 1))
        recipe = CompositeRecipe(
            task=task,
            components=(
                FunctionComponent(
                    FourierRecipe(task=task, length_scale=1e200, terms=8, calibration_rows=256)
                ),
                FunctionComponent(PolynomialRecipe(task=task, calibration_rows=256)),
            ),
            expression=Combination("sum", (0, 1)),
        )
        cases.append((f"composition-{kind}", recipe, 27))
    results = []
    for name, recipe, seed in cases:
        task = generate_task(recipe, seed=seed, task_id=0)
        card = json.loads(task.mechanism_json)
        graph = explain_generation(task, [0, 1, 3])
        assert graph["signal_max_error"] < 1e-10
        assert graph["target_max_error"] < 1e-10
        replay = generate_task(
            recipe_from_dict(json.loads(task.recipe_json)["recipe"]), seed=seed, task_id=0
        )
        assert task.episode() == replay.episode()
        assert task.mechanism_json == replay.mechanism_json
        errors = []
        if not isinstance(recipe, CompositeRecipe):
            x = task.inputs[0].tolist()
            for channel in range(task.scores.shape[1]):
                if isinstance(recipe, FourierRecipe):
                    raw = math.fsum(
                        -2
                        * math.sin(
                            math.fsum(
                                a * b for a, b in zip(x[: len(frequency)], frequency, strict=True)
                            )
                            / 2
                        )
                        * math.sin(
                            phase
                            + math.fsum(
                                a * b for a, b in zip(x[: len(frequency)], frequency, strict=True)
                            )
                            / 2
                        )
                        * coefficient[channel]
                        for frequency, phase, coefficient in zip(
                            card["frequencies"], card["phases"], card["coefficients"], strict=True
                        )
                    )
                else:
                    raw = math.fsum(
                        x[i] ** card["powers"][i] * card["weights"][i][channel]
                        for i in card["active"]
                    )
                expected = (raw - card["center"][channel]) / card["scale"][channel]
                errors.append(abs(expected - float(task.scores[0, channel])))
            assert max(errors) < 1e-10
        else:
            child = explain_generation(task, [0, 1, 3], component=0)
            json.dumps(child, allow_nan=False)
        (out / f"{name}.json").write_text(
            json.dumps(
                {"recipe": json.loads(task.recipe_json), "mechanism": card, "graph": graph},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        np.savez(
            out / f"{name}.npz",
            inputs=task.inputs.numpy(),
            scores=task.scores.numpy(),
            targets=task.targets.numpy(),
        )
        results.append(
            {
                "name": name,
                "episode_id": task.episode_id,
                "source_task_id": task.source_task_id,
                "row_check_max_error": None if not errors else max(errors),
                "graph_signal_max_error": graph["signal_max_error"],
                "graph_target_max_error": graph["target_max_error"],
                "replay": "PASS",
                "finite": bool(torch.isfinite(task.scores).all()),
            }
        )
    failures = []
    for recipe in [
        FourierRecipe(length_scale=1e-320, terms=8, calibration_rows=256),
        PolynomialRecipe(inputs=InputDistribution(scale=1e200), degree=16, calibration_rows=256),
    ]:
        with np.errstate(over="ignore", invalid="ignore"):
            try:
                generate_task(recipe, seed=27, task_id=0)
            except ValueError as error:
                failures.append({"family": recipe.version, "error": str(error)})
            else:
                raise AssertionError("invalid arithmetic unexpectedly accepted")
    report = {
        "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": "cpu",
        "cases": results,
        "expected_failures": failures,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": len(results), "expected_failures": failures}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
