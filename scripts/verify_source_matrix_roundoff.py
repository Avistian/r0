import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("evidence_root")
parser.add_argument("outputdir")
args = parser.parse_args()
r = pathlib.Path(args.evidence_root)
out = pathlib.Path(args.outputdir)
out.mkdir(parents=True, exist_ok=True)
(out / "execution.json").write_text(
    json.dumps(
        {
            "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            "python": sys.executable,
            "implementation_imports": [],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        indent=2,
    )
)
d = json.loads((r / "numerical/12-importance-output-0.0-local-shared-recipe.json").read_text())
raw = np.load(r / "numerical/stopped-raw.npz")
vals = []
findings = []


def rng(*keys):
    b = json.dumps(
        ("causal-v1", 28, 0, keys), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return np.random.default_rng(int(hashlib.sha256(b).hexdigest(), 16))


for i, n in enumerate(d["nodes"]):
    eps = torch.tensor(
        np.column_stack(
            [
                rng("source-activation-calibration-v1", "innovation", i, j).normal(size=32)
                for j in range(len(n["bias"]))
            ]
        )
    ) * torch.tensor(n["noise_scale"], dtype=torch.float64)
    x = torch.zeros_like(eps) + torch.tensor(n["bias"], dtype=torch.float64)
    if n["parents"]:
        x = x + torch.cat([vals[j] for j in n["parents"]], dim=1) @ torch.tensor(
            n["weights"], dtype=torch.float64
        )
    x = x + eps
    a = n["output_activation"]
    z = a["scale"] * (
        (x - torch.tensor(a["input_mean"], dtype=torch.float64))
        * torch.tensor(a["input_inverse_std"], dtype=torch.float64)
        + torch.tensor(a["shift"], dtype=torch.float64)
    )
    if a["kind"] == "sigmoid":
        y = torch.sigmoid(z)
    elif a["kind"] == "signed_power":
        y = torch.sign(z) * torch.abs(z).pow(dict(a["parameters"])["exponent"])
    elif a["kind"] == "gaussian_bump":
        y = torch.exp(-z * z)
    v = (y - torch.tensor(a["output_mean"], dtype=torch.float64)) * torch.tensor(
        a["output_inverse_std"], dtype=torch.float64
    )
    vals.append(v)
    data = x.numpy()
    digest = hashlib.sha256(
        f"{data.shape[0]},{data.shape[1]}:".encode() + data.astype("<f8").tobytes()
    ).hexdigest()
    entry = {
        "node": i,
        "sample_digest_exact": digest == a["calibration_digest"],
        "sample_mean": data.mean(axis=0).tolist(),
        "stored_mean": a["input_mean"],
    }
    if f"{i}-values" in raw:
        err = raw[f"{i}-values"] - v.numpy()
        entry["max_output_difference"] = float(np.abs(err).max())
        entry["worst_row_channel"] = np.unravel_index(np.abs(err).argmax(), err.shape)
    if i == 2:
        nx = raw["2-output-sample"]
        nz = a["scale"] * ((nx - np.array(a["input_mean"])) * a["input_inverse_std"] + a["shift"])
        entry["sample_max_difference"] = float(np.abs(nx - data).max())
        entry["shift_arguments_numpy"] = [nz[row, j] for j, row in enumerate(a["shift_rows"])]
        entry["shift_arguments_torch"] = [z[row, j].item() for j, row in enumerate(a["shift_rows"])]
    findings.append(entry)
(out / "roundoff-diagnosis.json").write_text(
    json.dumps(
        findings, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else list(x)
    )
)
print(
    json.dumps(
        findings, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else list(x)
    )
)
