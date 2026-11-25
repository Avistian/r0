"""Independently inspect the ZIP actually downloaded by Chromium."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np

root = Path(__file__).parent
with zipfile.ZipFile(root / "comparison.zip") as pair:
    card = json.loads(pair.read("card.json"))
    assert card["seed"] == 2**63 - 1
    assert card == json.loads((root / "comparison-card.json").read_text())
    datasets = []
    for side in ["before", "after"]:
        with zipfile.ZipFile(io.BytesIO(pair.read(side + ".zip"))) as data:
            with np.load(io.BytesIO(data.read("predictor.npz")), allow_pickle=False) as arrays:
                predictor = dict(arrays)
            with np.load(io.BytesIO(data.read("truth.npz")), allow_pickle=False) as arrays:
                truth = dict(arrays)
            assert len(data.read("truth.csv").decode().splitlines()) == 129
            assert "query_y" not in predictor
            datasets.append((predictor, truth))
    before, after = datasets
    assert np.array_equal(before[0]["query_x"], after[0]["query_x"])
    assert np.array_equal(before[0]["context_x"], after[0]["context_x"])
    assert np.array_equal(before[1]["scores"], after[1]["scores"])
    residuals = [truth["targets"] - truth["scores"][:, 0] for _, truth in datasets]
    error = float(np.abs(residuals[1] - 4 * residuals[0]).max())
    assert error < 1e-14
    print(
        json.dumps(
            {
                "rows": 128,
                "seed": card["seed"],
                "pair_id": card["pair_id"],
                "identical_inputs_and_scores": True,
                "residual_multiplier": 4,
                "max_residual_scaling_error": error,
                "bytes": (root / "comparison.zip").stat().st_size,
            },
            indent=2,
        )
    )
