"""Independent exact-square oracle for a multidimensional quasi-distance tie."""

import contextlib
import hashlib
import io
import json
import math
from pathlib import Path

import torch
from evidence_common import finish, p, record_event

ROOT = Path(__file__).resolve().parent
f = p.FrozenDiscretizationFunction(
    ((64.0, 0.0), (9.0, 25.0)), p.FrozenLinearFunction(((1.0, 0.0),)), 0.5
)
# Integer perfect squares give exact radicals, independent of any float norm.
radicals = [[math.isqrt(v) for v in center] for center in ((64, 0), (9, 25))]
assert radicals == [[8, 0], [3, 5]]
distances = [sum(row) ** 2 for row in radicals]
assert distances == [64, 64]
inputs = torch.zeros((1, 2), dtype=torch.float64)
observed = p.evaluate_frozen_function(f, inputs).item()
source_distances = torch.cdist(inputs, torch.tensor(f.centers, dtype=torch.float64), p=0.5)
assert source_distances.tolist() == [[64.0, 64.0]]
event = {
    "name": "exact-two-axis-quasi-tie",
    "status": "REJECT" if observed != 64 else "PASS",
    "centers": f.centers,
    "linear": f.linear.matrix,
    "p": f.p,
    "inputs": inputs.tolist(),
    "exact_integer_radicals": radicals,
    "exact_distances": distances,
    "expected": 64.0,
    "observed": observed,
    "cached_source_expression_distances": source_distances.tolist(),
    "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(ROOT / "quasi-tie.json").write_text(json.dumps(event, indent=2) + "\n")
record_event(event)
with contextlib.redirect_stdout(io.StringIO()):
    finish()
assert observed == 64, "first exact quasi-distance minimum must win"
