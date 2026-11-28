"""Check serialization closure of a publicly accepted frozen MLP record."""

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import r0
from r0.modules import priors

ROOT = Path(__file__).resolve().parent
assert Path(r0.__file__).resolve().is_relative_to(ROOT / "source")
torch.set_num_threads(1)
activation = priors.CalibratedActivation(
    kind="identity",
    parameters=(("input_scale", 2.0), ("output_shift", 1.0)),
    input_mean=(0.0,),
    input_inverse_std=(1.0,),
    shift=(0.0,),
    scale=1.0,
    output_mean=(0.0,),
    output_inverse_std=(1.0,),
    shift_rows=(0,),
    calibration_rows=1,
    calibration_id="independent-serialization-fixture",
    calibration_digest="0" * 64,
)
record = priors.FrozenMLPFunction(
    linears=(priors.FrozenLinearFunction(((1.0,),)),),
    activations=(activation,),
    end_activation=True,
)
inputs = torch.tensor([[2.0]], dtype=torch.float64)
# All wrapper constants are identities: 1 * (2 * 2 + 1) = 5.
expected = 5.0
observed = priors.evaluate_frozen_function(record, inputs).item()
assert observed == expected
payload = json.loads(json.dumps(asdict(record)))
error = None
try:
    restored = priors.frozen_function_from_dict(payload)
except (ValueError, TypeError, KeyError) as exc:
    error = {"type": type(exc).__name__, "message": str(exc)}
else:
    assert restored == record
imports = {}
for name, module in sorted(sys.modules.items()):
    if name == "r0" or name.startswith("r0."):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(ROOT / "source")
        imports[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
receipt = {
    "verdict": "REJECT" if error else "PASS",
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "threads": torch.get_num_threads(),
    "device": "cpu",
    "dtype": "float64",
    "record_accepted_by_public_constructor": True,
    "fixture": payload,
    "inputs": inputs.tolist(),
    "evaluation_expected": expected,
    "evaluation_observed": observed,
    "json_roundtrip_error": error,
    "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "imports": imports,
}
(ROOT / "roundtrip.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "imports"}, indent=2), flush=True)
assert error is None, "publicly accepted frozen MLP record cannot reconstruct from its JSON"
