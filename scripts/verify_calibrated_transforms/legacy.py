"""Public generation compatibility capture; every proposal is retained once."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from custody import record_custody

from r0.modules import priors


def encode(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    record_custody(Path(os.environ["LEGACY_OUTPUT"]).with_suffix(".custody.json"))
    profiles = [
        f"causal-{name}-v1"
        for name in (
            "affine",
            "activated",
            "preferences",
            "exogenous",
            "factorized",
            "importance",
            "matrices",
            "gaussian-matrices",
        )
    ]
    profiles += [
        f"causal-{mode}-matrices-v1" for mode in ("shared", "refreshed", "local", "global")
    ]
    profiles += [
        f"causal-source-{selector}{suffix}{matrix}-v1"
        for matrix in ("", "-matrices")
        for selector in ("standard", "plain")
        for suffix in ("", "-refreshed", "-local", "-global")
    ]
    records = []
    for profile in profiles:
        for family in ("forward", "layered", "redirection"):
            for task_id in (0, 10):
                row = dict(profile=profile, family=family, seed=28, task_id=task_id)
                try:
                    recipe = priors.sample_recipe(
                        family,
                        profile=profile,
                        seed=28,
                        task_id=task_id,
                        task=priors.LinearRecipe(rows=16, context_rows=8),
                    )
                    row["recipe"] = asdict(recipe)
                    task = priors.generate_task(recipe, seed=28, task_id=task_id)
                    row["task"] = asdict(task)
                    row["trace"] = priors.explain_generation(task, [0, 7, 15])
                    row["status"] = "PASS"
                except (ValueError, RuntimeError) as error:
                    row["status"] = type(error).__name__
                    row["error"] = str(error)
                records.append(row)
    out = Path(os.environ["LEGACY_OUTPUT"])
    out.write_text(json.dumps(records, default=encode, sort_keys=True) + "\n")
    print(
        json.dumps(
            dict(
                profiles=len(profiles),
                proposals=len(records),
                passed=sum(row["status"] == "PASS" for row in records),
                sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
            )
        )
    )


if __name__ == "__main__":
    main()
