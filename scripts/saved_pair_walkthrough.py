"""Create ten saved pairs, then verify them in a separate process without a worker."""

import argparse
import base64
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r0.modules.explorer import ExplorerSession
from r0.modules.generation import load_group
from r0.modules.priors import generate_task, recipe_from_dict


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def create(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    session = ExplorerSession(root / "datasets")
    examples = session.request("examples", {})["examples"]
    changes = [
        ("regression", "sparsity", 2),
        ("binary", "irrelevant", 12),
        ("regression", "noise", 0.8),
        ("multiclass", "sparsity", 2),
        ("binary", "separation", 0),
        ("multiclass", "separation", 0),
        ("regression", "irrelevant", 12),
        ("regression", "noise", 0.8),
        ("multiclass", "irrelevant", 12),
        ("regression", "sparsity", 2),
    ]
    cases = []
    for i, ((family, recipe), (kind, intervention, value)) in enumerate(
        zip(examples.items(), changes, strict=True)
    ):
        task = recipe.get("task", recipe)
        task.update(
            rows=32,
            context_rows=20,
            task_type=kind,
            classes=0 if kind == "regression" else 2 if kind == "binary" else 3,
            noise_ratio=0.2 if kind == "regression" else 0,
        )
        if family == "Exact Gaussian process":
            recipe["calibration_rows"] = 32
        session.request("generate", {"recipe": recipe, "seed": str(2**63 - 1), "task_id": i})
        result = session.request(
            "pair",
            {
                "intervention": intervention,
                "value": value,
                "allow_estimated_variance": intervention == "sparsity" and family != "Linear",
            },
        )
        pair = result["pair"]
        exported = session.request("pair_export", {"expected_pair_id": pair["card"]["pair_id"]})
        payload = base64.b64decode(exported["data"])
        filename = f"pair-{i}.zip"
        (root / filename).write_bytes(payload)
        write(root / f"card-{i}.json", json.loads(pair["card_json"]))
        cases.append(
            {
                "family": family,
                "task_type": kind,
                "intervention": intervention,
                "path": pair["path"],
                "pair_id": pair["card"]["pair_id"],
                "export": filename,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        print(f"created {family}: {intervention}", flush=True)
    write(
        root / "created.json",
        {
            "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "cases": cases,
        },
    )


def verify(root: Path) -> None:
    records = json.loads((root / "created.json").read_text())
    verification_source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if verification_source != records["source"]:
        raise ValueError("verify must use the same source revision as create")
    output = root / "datasets"
    attempts = sorted(str(path) for path in output.glob(".attempts/*/*.json"))
    cases = []
    for case in records["cases"]:
        session = ExplorerSession(output)
        result = session.request("open_pair", {"path": case["path"]})
        pair = result["pair"]
        assert pair["card"]["pair_id"] == case["pair_id"]
        card = json.loads(pair["card_json"])
        payload = (root / case["export"]).read_bytes()
        assert len(payload) == case["bytes"]
        assert hashlib.sha256(payload).hexdigest() == case["sha256"]
        with zipfile.ZipFile(io.BytesIO(payload)) as exported:
            assert json.loads(exported.read("card.json")) == card
            for side, view in (("before", result), ("after", pair["after"])):
                saved = next(load_group(output / view["archive"]))
                record = json.loads(saved.recipe_json)
                replay = generate_task(
                    recipe_from_dict(record["recipe"]),
                    seed=record["seed"],
                    task_id=record["task_id"],
                )
                assert saved.mechanism_json == replay.mechanism_json
                for key in ("inputs", "targets", "scores", "context_indices", "query_indices"):
                    assert torch.equal(getattr(saved, key), getattr(replay, key))
                if saved.probabilities is not None:
                    assert torch.equal(saved.probabilities, replay.probabilities)
                with zipfile.ZipFile(io.BytesIO(exported.read(f"{side}.zip"))) as bundle:
                    assert json.loads(bundle.read("recipe.json")) == record
                    assert len(bundle.read("predictor.csv").decode().splitlines()) == 33
                    assert len(bundle.read("truth.csv").decode().splitlines()) == 33
                    with np.load(
                        io.BytesIO(bundle.read("predictor.npz")), allow_pickle=False
                    ) as arrays:
                        assert np.array_equal(
                            arrays["query_x"], saved.inputs[saved.query_indices].numpy()
                        )
                        assert np.array_equal(
                            arrays["context_y"], saved.targets[saved.context_indices].numpy()
                        )
                        assert "query_y" not in arrays
                    with np.load(
                        io.BytesIO(bundle.read("truth.npz")), allow_pickle=False
                    ) as arrays:
                        assert np.array_equal(arrays["scores"], saved.scores.numpy())
                        assert np.array_equal(arrays["targets"], saved.targets.numpy())
        assert all(card["required_equalities"].values())
        cases.append(
            {
                **case,
                "replay": "PASS",
                "export_arrays": "PASS",
                "reopen": "PASS",
                "variance_control": card["variance_control"],
            }
        )
        print(f"verified {case['family']}: {case['intervention']}", flush=True)
    assert sorted(str(path) for path in output.glob(".attempts/*/*.json")) == attempts
    write(
        root / "verified.json",
        {"source": records["source"], "worker_launches_during_verify": 0, "cases": cases},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (create if args.mode == "create" else verify)(args.out)
