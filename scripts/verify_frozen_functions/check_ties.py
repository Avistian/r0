"""Independent exact integer-distance oracle for discretization ties."""

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import r0
from r0.modules import priors

ROOT = Path(__file__).resolve().parent
assert Path(r0.__file__).resolve().is_relative_to(ROOT / "source")
torch.set_num_threads(1)
print(json.dumps({"python": sys.version, "torch": torch.__version__, "r0": r0.__file__}))
for total in range(3, 101):
    centers = tuple((float(total - j), float(j)) for j in range(total + 1))
    record = priors.FrozenDiscretizationFunction(
        centers=centers,
        linear=priors.FrozenLinearFunction(((1.0, 0.0),)),
        p=1.0,
    )
    # Exact integer sums prove every center has the same L1 distance.
    exact_distances = [sum(abs(int(v)) for v in center) for center in centers]
    assert all(d == total for d in exact_distances)
    expected = float(total)  # The first center wins every exact tie.
    observed = priors.evaluate_frozen_function(
        record, torch.zeros((1, 2), dtype=torch.float64)
    ).item()
    print(json.dumps({"total": total, "expected": expected, "observed": observed}), flush=True)
    if observed != expected:
        failure = {
            "status": "REJECT",
            "record": asdict(record),
            "inputs": [[0.0, 0.0]],
            "exact_distances": exact_distances,
            "expected": expected,
            "observed": observed,
            "source_cdist": torch.cdist(
                torch.zeros((1, 2), dtype=torch.float64),
                torch.tensor(centers, dtype=torch.float64),
                p=1.0,
            ).tolist(),
            "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (ROOT / "first-failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise AssertionError(f"first-minimum tie broken at total={total}: {observed} != {expected}")
print("PASS 98 exact integer-distance tie cases")
