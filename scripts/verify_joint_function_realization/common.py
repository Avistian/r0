"""Pinned local-only evidence helpers."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from r0.modules import priors

ROOT = Path(os.environ.get("R0_REVIEW_ROOT", Path.cwd())).resolve()
INPUTS = Path(os.environ["R0_REVIEW_INPUTS"]).resolve()
OUTPUT = Path(os.environ["R0_REVIEW_OUTPUT"]).resolve()
HEAD = "c1ee89e766194993c1a79fcc7400f7b2b48fefbe"


def pin():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
    if head.strip() != HEAD:
        raise RuntimeError("checkout HEAD differs from reviewed source")
    pin_files = {
        "source-pins-1.json": "89c5cb98752c5f61cd30ec579d07482c4d1e1ba49413692215d7a9c62e19fbcd",
        "source-pins-2.json": "ae0d60c838c43051c4c9256fa6f5950ba034e3cfd5ef275e17246d0b3b395874",
    }
    pins = {}
    for filename, expected in pin_files.items():
        contents = (INPUTS / filename).read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected:
            raise RuntimeError("source pin inventory identity differs")
        pins.update(json.loads(contents))
    for path, digest in pins.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"source bytes differ: {path}")
    expected_import = ROOT / "src/r0/modules/priors/__init__.py"
    if Path(priors.__file__).resolve() != expected_import:
        raise RuntimeError("imported package differs from supplied checkout")
    source = hashlib.sha256((INPUTS / "I2func.py").read_bytes()).hexdigest()
    if source != "ff2939197b353f6d11a3fbaf0e27586998bdf93527dad11f24698ff296ce2f6c":
        raise RuntimeError("cached primary source identity differs")
    print(
        json.dumps(
            {
                "head": HEAD,
                "python": sys.executable,
                "version": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "platform": platform.platform(),
                "source_sha256": source,
                "imports": priors.__file__,
                "root": str(ROOT),
                "inputs": str(INPUTS),
                "output": str(OUTPUT),
            }
        ),
        flush=True,
    )
    torch.set_num_threads(1)


def plan_for(family, seed=0, width=2, rows=4, centers=None):
    context = priors.JointSamplingContext(np.random.default_rng(seed), mode="local")
    prior = priors.JointFunctionPrior(
        functions=family,
        frequencies=4,
        discretization_centers=centers,
        matrix=priors.JointMatrixPrior(families=("gaussian",), vector_family="uniform"),
    )
    plan = priors.sample_joint_function(
        context, "probe", input_width=width, output_width=2, calibration_rows=rows, prior=prior
    )
    return plan, context.freeze()


def build(plan, decisions, sample, raw=None, fit=None):
    return priors.materialize_joint_function(
        plan,
        decisions,
        rng=np.random.default_rng(303) if raw is None else raw,
        fit_rng=np.random.default_rng(404) if fit is None else fit,
        calibration=sample,
        calibration_id="independent-review-fixture",
    )
