"""Persist composed tasks and independently reconstruct a saved scalar target."""

import argparse
import io
import json
import math
import subprocess
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from r0.modules.generation import RunRequest, run_guarded
from r0.modules.priors import (
    Combination,
    CompositeRecipe,
    FourierRecipe,
    FunctionComponent,
    GroupedLinearRecipe,
    InputDistribution,
    LinearRecipe,
    PolynomialRecipe,
    QuadraticRecipe,
    SmoothRecipe,
)


def scalar_from_card(card: dict[str, Any], row: list[float], channel: int = 0) -> float:
    values = []
    for component in card["components"]:
        m = component["mechanism"]
        x = [0.0] * component["feature_count"]
        for child, parent in zip(m["active"], component["input_columns"], strict=True):
            x[child] = row[parent]
        if m["version"] in ("linear-v1", "grouped-linear-v1", "polynomial-v1"):
            powers = m.get("powers", [1] * len(m["weights"]))
            raw = math.fsum(
                x[i] ** p * w[channel]
                for i, (p, w) in enumerate(zip(powers, m["weights"], strict=True))
            )
        elif m["version"] == "quadratic-v1":
            directions, weights = m["directions"], m["weights"]
            terms = []
            for i in range(len(directions[0])):
                for j in range(len(directions[0])):
                    a = math.fsum(
                        d[i] * d[j] * w[channel] for d, w in zip(directions, weights, strict=True)
                    )
                    terms.append(x[i] * x[j] * a * (1 if i == j else m["interaction_strength"]))
            raw = math.fsum(terms) + m["linear_strength"] * math.fsum(
                x[i] * w[channel] for i, w in enumerate(m["linear_weights"])
            )
        elif m["version"] == "fourier-v1":
            raw = math.fsum(
                math.cos(math.fsum(v * f for v, f in zip(x, frequency, strict=True)) + phase)
                * coefficient[channel]
                for frequency, phase, coefficient in zip(
                    m["frequencies"], m["phases"], m["coefficients"], strict=True
                )
            )
        elif m["version"] == "smooth-v1":
            terms = []
            for d, center, width, weight in zip(
                m["directions"], m["centers"], m["widths"], m["weights"], strict=True
            ):
                z = (math.fsum(v * a for v, a in zip(x, d, strict=True)) - center) / width
                terms.append(
                    (math.sin(z) if m["family"].endswith("sinusoidal") else math.exp(-z * z / 2))
                    * weight[channel]
                )
            raw = math.fsum(terms)
        else:
            raise ValueError(m["version"])
        if "center" in m:
            raw = (raw - m["center"][channel]) / m["scale"][channel]
        values.append(raw * component["strength"])

    def expression(node: Any) -> float:
        if type(node) is int:
            return values[node]
        children = [expression(child) for child in node["children"]]
        return node["strength"] * (
            math.fsum(children) if node["operation"] == "sum" else math.prod(children)
        )

    return (expression(card["expression"]) - card["center"][channel]) / card["scale"][channel]


def cases() -> list[CompositeRecipe]:
    small = LinearRecipe(rows=16, context_rows=8)
    all_families = (
        small,
        GroupedLinearRecipe(task=small, calibration_rows=256),
        PolynomialRecipe(task=small, calibration_rows=256),
        QuadraticRecipe(task=small, rank=4, calibration_rows=256),
        FourierRecipe(task=small, terms=8, calibration_rows=256),
        FourierRecipe(task=small, family="periodic", terms=8, calibration_rows=256),
        SmoothRecipe(task=small, family="sinusoidal", terms=4, calibration_rows=256),
        SmoothRecipe(task=small, projection="ridge", terms=8, calibration_rows=256),
    )
    result = [
        CompositeRecipe(
            task=small,
            components=tuple(
                FunctionComponent(r, strength=(-1) ** i / 8) for i, r in enumerate(all_families)
            ),
            expression=Combination("sum", tuple(range(8))),
            calibration_rows=256,
        )
    ]
    binary = replace(small, task_type="binary", classes=2, active_features=2)
    inputs = InputDistribution(distribution="uniform")
    result.append(
        CompositeRecipe(
            task=replace(
                binary,
                rows=256,
                context_rows=128,
                active_features=4,
                separation=0.1,
                class_offsets=(-3,),
            ),
            inputs=inputs,
            components=(
                FunctionComponent(
                    GroupedLinearRecipe(task=binary, inputs=inputs, calibration_rows=256),
                    axes=(0, 1),
                ),
                FunctionComponent(
                    QuadraticRecipe(task=binary, inputs=inputs, rank=4, calibration_rows=256),
                    axes=(2, 3),
                ),
            ),
            expression=Combination("product", (0, 1)),
            calibration_rows=512,
        )
    )
    multiclass = replace(
        small, task_type="multiclass", classes=3, features=1, signal_features=1, active_features=1
    )
    bounded = InputDistribution(distribution="student_t", df=1, bound=3)
    tree: int | Combination = 0
    for _ in range(8):
        tree = Combination("product", (tree, 0))
    result.append(
        CompositeRecipe(
            task=multiclass,
            inputs=bounded,
            components=(
                FunctionComponent(
                    PolynomialRecipe(
                        task=multiclass, inputs=bounded, degree=1, calibration_rows=256
                    )
                ),
            ),
            expression=tree,
            calibration_rows=256,
        )
    )
    result.append(
        CompositeRecipe(
            task=small,
            components=(
                FunctionComponent(PolynomialRecipe(task=small, degree=1, calibration_rows=256)),
            )
            * 64,
            expression=Combination("sum", tuple(range(64))),
            calibration_rows=256,
        )
    )
    large = replace(
        small,
        features=256,
        signal_features=256,
        active_features=256,
        task_type="multiclass",
        classes=100,
    )
    result.append(
        CompositeRecipe(
            task=replace(large, rows=20000, context_rows=10000),
            components=(
                FunctionComponent(QuadraticRecipe(task=large, rank=8, calibration_rows=256)),
                FunctionComponent(
                    SmoothRecipe(task=large, projection="ridge", terms=32, calibration_rows=256)
                ),
            ),
            expression=Combination("sum", (0, 1)),
            calibration_rows=16384,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for i, recipe in enumerate(cases()):
        request = RunRequest(
            recipe, seed=27, count=1, group_size=1, max_seconds=120, max_bytes=128 * 1024**2
        )
        result = run_guarded(request, args.out / str(i))
        assert result.returncode == 0, result.stderr
        assert result.summary is not None
        assert result.summary["completed"] == 1
        archive = next((args.out / str(i)).glob("groups/*/*/data-*.zip"))
        with zipfile.ZipFile(archive) as saved:
            card = json.loads(saved.read("0/card.json"))["mechanism"]
            row = json.loads(saved.read("0/predictor.json"))["context_rows"][0]
            with np.load(io.BytesIO(saved.read("0/predictor.npz")), allow_pickle=False) as data:
                x = data["context_x"][0].tolist()
            with np.load(io.BytesIO(saved.read("0/truth.npz")), allow_pickle=False) as data:
                actual = float(data["scores"][row, 0])
        expected = scalar_from_card(card, x)
        assert math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9), (i, expected, actual)
        resumed = run_guarded(request, args.out / str(i))
        assert resumed.returncode == 0
        assert resumed.summary is not None
        assert resumed.summary["generated"] == 0
        records.append(
            {
                "request": asdict(request),
                "row_index": row,
                "scalar_expected": expected,
                "scalar_actual": actual,
                "receipt": json.loads(archive.with_suffix(".json").read_text()),
                "result": asdict(result),
                "resume": asdict(resumed),
            }
        )
    bad = CompositeRecipe(
        components=(FunctionComponent(PolynomialRecipe(calibration_rows=256), strength=5e-324),),
        calibration_rows=256,
    )
    failed_request = RunRequest(
        bad, seed=27, count=1, group_size=1, max_seconds=30, max_bytes=1048576
    )
    failed = run_guarded(failed_request, args.out / "failure")
    assert failed.summary is not None
    assert failed.summary["failed"] == 1
    assert failed.summary["completed"] == 0
    failures = [
        json.loads(line)
        for p in (args.out / "failure").rglob("failures-*.jsonl")
        for line in p.read_text().splitlines()
    ]
    print(
        json.dumps(
            {
                "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "cases": records,
                "failure": {
                    "request": asdict(failed_request),
                    "result": asdict(failed),
                    "records": failures,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
