"""Identical public plans/ledger/next draws in isolated baseline and candidate processes."""

import hashlib
import json
from dataclasses import asdict

import numpy as np
from common import save, start

from r0.modules import priors


def main():
    output = start()
    cases = []
    for engine in (
        np.random.PCG64,
        np.random.PCG64DXSM,
        np.random.MT19937,
        np.random.Philox,
        np.random.SFC64,
    ):
        for mode in ("local", "global", "meta"):
            rng = np.random.Generator(engine(781))
            context = priors.JointSamplingContext(rng, mode=mode)
            with context.transaction("graph") as graph:
                weights = priors.sample_joint_weights(
                    graph, "weights", width=3, family="polynomial"
                )
                root = priors.sample_joint_root(
                    graph,
                    "root",
                    width=2,
                    calibration_rows=4,
                    prior=priors.JointRootPrior(
                        families=("gaussian",), function=priors.JointFunctionPrior(functions="tree")
                    ),
                )
                graph.categorical("choice", priors.CategoricalParameter("category", ("a", "b")))
            data = {
                "weights": asdict(weights),
                "root": asdict(root),
                "ledger": asdict(context.freeze()),
                "next_named": rng.random(16).tolist(),
            }
            raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            name = f"{engine.__name__}-{mode}"
            for offset in range(0, len(raw), 16000):
                (output / f"{name}-{offset // 16000:02}.txt").write_bytes(
                    raw[offset : offset + 16000]
                )
            cases.append(
                {"case": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            )
    save(output, "result", {"cases": cases})
    print(f"Captured {len(cases)} isolated public-plan/ledger/seed cases")


if __name__ == "__main__":
    main()
