"""Public standalone root materialization/emission compared with old validation."""

import dataclasses
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

from r0.modules import priors as p
from r0.modules.priors.lib.joint_root_base import JointRootBase

OUT = Path(os.environ["R0_VERIFY_OUT"])
source = subprocess.check_output(
    [
        "git",
        "show",
        "3446858399f32789017266700b5e4150fff71fa9:src/r0/modules/priors/lib/joint_root_base.py",
    ],
    text=True,
)
m = types.ModuleType("independent_materialization_baseline")
sys.modules[m.__name__] = m
exec(compile(source, "pinned_root_baseline", "exec"), m.__dict__)
for family in ("uniform", "gaussian", "ball", "covariance"):
    for vector in ("uniform", "polynomial"):
        context = p.JointSamplingContext(np.random.default_rng(814), mode="local")
        plan = p.sample_joint_root(
            context,
            "root",
            width=2,
            calibration_rows=4,
            prior=p.JointRootPrior(
                families=(family,),
                function=p.JointFunctionPrior(
                    functions="lin",
                    matrix=p.JointMatrixPrior(families=("gaussian",), vector_family=vector),
                ),
            ),
        )
        decisions = context.freeze()

        def run(plan=plan, decisions=decisions):
            root = p.materialize_joint_root(
                plan,
                decisions,
                rng=np.random.default_rng(815),
                calibration_rng=np.random.default_rng(816),
                fit_rng=np.random.default_rng(817),
                calibration_id="independent-compatibility",
            )
            emitted = p.sample_joint_root_values(
                root,
                decisions,
                rows=7,
                rng=np.random.default_rng(818),
                sample_id="independent-fresh",
            )
            return root, emitted

        with patch.object(JointRootBase, "__post_init__", m.JointRootBase.__post_init__):
            old, old_emitted = run()
        new, emitted = run()
        assert dataclasses.asdict(old) == dataclasses.asdict(new)
        assert dataclasses.asdict(old_emitted) == dataclasses.asdict(emitted)
        # Independent literal frozen-linear equation on retained transformed base.
        expected = np.asarray(emitted.sample.values) @ np.asarray(new.function.function.matrix).T
        np.testing.assert_array_equal(expected, np.asarray(emitted.outputs))
        restored = p.joint_root_emission_from_dict(
            json.loads(json.dumps(dataclasses.asdict(emitted))), decisions=decisions
        )
        assert restored == emitted
        print("PASS standalone root materialize/emit/reconstruct", family, vector, flush=True)
print("PASS: 8 standalone root paths preserve pinned records and literal linear outputs.")
