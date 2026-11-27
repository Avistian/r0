"""Independent admission accounting, custody validation and frozen record invariance."""

import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from custody import record_custody

from r0.modules import priors as p
from r0.modules.priors.lib.node_pipeline import PipelineEvaluation

OUT = Path(__file__).parent
SOURCE = Path(p.__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    record = json.loads((OUT / "numerical-readout-only.json").read_text())["recipe"]
    recipe = p.recipe_from_dict(record)
    # Base equations: root 3*(2+0), child 3*(2+3) = 21.
    # Matrix metadata: Gaussian exogenous 4, response 3, L/D/R = 9+3+9 = 21.
    # Pipelines: root (width/importance 2, warp converter 8, identity converter 1,
    # factors 3, L2 3) = 17; child (width/importance 2, warp converters 16, factors 3)=21.
    parameters = 21 + (4 + 3 + 21) + 17 + 21
    # Retained stages/buffers: root (8+2)*3+16 = 46; child (8+1)*3+16 = 43.
    # Matrix stochastic scratch 8*3 = 24. Sum retained pipeline nodes, not max.
    working = 8 * 16 * (6 * 6 + 2 * 3 + 2 * 3 + 4 * 1 + 4 + 24 + 46 + 43)
    working += 128 * parameters
    actual = json.loads((OUT / "resource-observed.json").read_text())
    assert parameters == 87 == actual["parameters"]
    assert working == 32768 == actual["estimated_working_bytes"]
    before = asdict(recipe)["pipelines"]
    original = p.generate_task(recipe, seed=125, task_id=9)
    changed = replace(recipe, calibration_rows=4096, task=replace(recipe.task, context_rows=3))
    repeated = p.generate_task(changed, seed=125, task_id=9)
    assert asdict(changed)["pipelines"] == before
    np.testing.assert_array_equal(original.inputs, repeated.inputs)
    np.testing.assert_array_equal(original.scores, repeated.scores)
    # Alias independence through subsequent state mutation; exact captured tensor values.
    evaluation = PipelineEvaluation(recipe.pipelines[0], 0, None)
    input_value = original.inputs.clone()
    state = evaluation.before_noise(input_value)
    captured = {k: v.clone() for k, v in evaluation.captured.items()}
    state.fill_(900.0)
    input_value.fill_(-300.0)
    for channel, expected in captured.items():
        np.testing.assert_array_equal(evaluation.captured[channel], expected)
    custody = []
    for name, commit, source_root in (
        ("legacy-baseline", "f5efdc34ad75012e9d24c7b69a221fa86cf396ee", OUT / "baseline/src"),
        ("legacy-candidate", "ae9b3b8b64c226d1532e7cb7939ab99b907fc2ad", SOURCE / "src"),
        ("verify-start", "ae9b3b8b64c226d1532e7cb7939ab99b907fc2ad", SOURCE / "src"),
        ("verify-end", "ae9b3b8b64c226d1532e7cb7939ab99b907fc2ad", SOURCE / "src"),
    ):
        receipt = json.loads((OUT / f"{name}.custody.json").read_text())
        for module, item in receipt["imports"].items():
            path = Path(item["path"])
            relative = path.relative_to(source_root)
            committed = subprocess.check_output(
                ["git", "show", f"{commit}:src/{relative}"], cwd=SOURCE
            )
            assert hashlib.sha256(committed).hexdigest() == item["sha256"], module
            assert digest(path) == item["sha256"], module
        for script, sha in receipt["scripts"].items():
            assert digest(OUT / script) == sha, (name, script)
        custody.append(dict(run=name, commit=commit, imported_modules=len(receipt["imports"])))
    assert (OUT / "legacy-baseline.json").read_bytes() == (
        OUT / "legacy-candidate.json"
    ).read_bytes()
    result = dict(
        status="PASS",
        parameters=parameters,
        estimated_working_bytes=working,
        frozen_records="unchanged with context and calibration row count",
        alias_independence="PASS",
        custody=custody,
        legacy_sha256=digest(OUT / "legacy-candidate.json"),
    )
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    record_custody(OUT / "audit.custody.json")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
