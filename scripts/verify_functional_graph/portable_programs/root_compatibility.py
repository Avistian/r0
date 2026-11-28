"""Pinned covariance validation and root emission compatibility."""

import contextlib
import dataclasses
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from r0.modules import priors as p
from r0.modules.priors.lib.joint_root_base import JointRootBase, draw_root_base

OUT = Path(os.environ["R0_VERIFY_OUT"])
oldsource = subprocess.check_output(
    [
        "git",
        "show",
        "3446858399f32789017266700b5e4150fff71fa9:src/r0/modules/priors/lib/joint_root_base.py",
    ],
    text=True,
)
(OUT / "baseline_joint_root_base.py").write_text(oldsource)
m = types.ModuleType("independent_pinned_root_baseline")
sys.modules[m.__name__] = m
exec(compile(oldsource, str(OUT / "baseline_joint_root_base.py"), "exec"), m.__dict__)
weights = [
    p.UniformWeightVector((0.0, 0.5, float(np.nextafter(1.0, 0.0)))),
    p.PolynomialWeightVector(
        (0.0, 1.0, -1.0), decay=2.0, log_noise_scale=1.0, permutation=(2, 0, 1)
    ),
    p.PolynomialWeightVector(
        (1e308, 1e308, 1e308), decay=1.0, log_noise_scale=1.0, permutation=(1, 2, 0)
    ),
]
values = [
    0.0,
    -0.0,
    5e-324,
    -5e-324,
    1e-300,
    -1e-300,
    0.1,
    1.0,
    -3.0,
    float(2**53 + 1),
    1e308,
    -1e308,
]
results = []
for law in weights:
    for value in values:
        gaussian = ((value, 1.0, -value), (1e-300, value, 5e-324), (-3.0, -0.0, value))
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            expected = (
                np.asarray(gaussian, dtype=np.float64)
                * np.asarray(law.values(), dtype=np.float64)[None, :]
            )
        kw = dict(
            family="covariance",
            width=3,
            gaussian_factor=gaussian,
            weights=law,
            factor=tuple(tuple(map(float, row)) for row in expected),
            covariance_base="gaussian",
        )
        outcomes = []
        for constructor in (m.JointRootBase, JointRootBase):
            try:
                outcomes.append(("ACCEPT", constructor(**kw)))
            except ValueError as e:
                outcomes.append(("REJECT", str(e)))
        assert outcomes[0][0] == outcomes[1][0], (kw, outcomes)
        if outcomes[0][0] == "ACCEPT":
            actual = np.asarray(outcomes[1][1].factor, dtype=np.float64)
            assert actual.tobytes() == expected.tobytes()
            corrupt = expected.copy()
            corrupt[0, 1] = np.nextafter(corrupt[0, 1], np.inf)
            bad = dict(kw, factor=tuple(tuple(map(float, row)) for row in corrupt))
            try:
                JointRootBase(**bad)
            except ValueError:
                pass
            else:
                raise AssertionError("REJECT one-ULP factor corruption accepted")
        results.append({"law": law.version, "value": value, "outcome": outcomes[1][0]})
# Draw/emission equivalence for non-overflowing uniform/polynomial covariance laws.
for family in ("uniform", "gaussian", "ball", "covariance"):
    for base_law in ("uniform", "gaussian") if family == "covariance" else (None,):
        for law in weights[:2] if family == "covariance" else (None,):
            kw = {"family": family, "width": 3}
            if family == "covariance":
                g = np.array([[1.0, 2.0, 0.0], [-1.0, 3.0, 0.5], [0.0, 0.0, 1.0]])
                factor = g * np.asarray(law.values())[None, :]
                kw.update(
                    gaussian_factor=tuple(map(tuple, g)),
                    weights=law,
                    factor=tuple(map(tuple, factor)),
                    covariance_base=base_law,
                )
                # Public descriptors require built-in finite floats.
                kw["gaussian_factor"] = tuple(tuple(map(float, row)) for row in g)
                kw["factor"] = tuple(tuple(map(float, row)) for row in factor)
            old = m.draw_root_base(m.JointRootBase(**kw), 16, np.random.default_rng(321))
            new = draw_root_base(JointRootBase(**kw), 16, np.random.default_rng(321))
            assert dataclasses.asdict(old) == dataclasses.asdict(new)
            assert np.asarray(old.values).tobytes() == np.asarray(new.values).tobytes()
            print(
                "PASS pinned draw/emission bytes",
                family,
                base_law,
                law.version if law else "-",
                flush=True,
            )
# Broader transparent allocation observers; enforce caller and payload aggregate caps.
payload = json.loads((OUT / "over_channel_covariance_payload.json").read_text())
observed = []


def observe(owner, name):
    original = getattr(owner, name)

    def wrapper(*args, **kw):
        result = original(*args, **kw)
        observed.append(name)
        return result

    return wrapper


with contextlib.ExitStack() as stack:
    for owner, names in [
        (np, ("asarray", "array", "empty", "zeros", "ones", "full", "fromiter", "frombuffer")),
        (torch, ("tensor", "as_tensor", "empty", "zeros", "ones", "full")),
    ]:
        for name in names:
            stack.enter_context(patch.object(owner, name, observe(owner, name)))
    for cap in ({"node_channels": 1}, {"total_channels": 1}, {"working_bytes": 8192}):
        for ownership in ("caller", "payload"):
            data = json.loads(json.dumps(payload))
            kwargs = {}
            if ownership == "caller":
                kwargs["limits"] = p.CausalLimits(**cap)
            else:
                data["limits"].update(cap)
            try:
                p.functional_recipe_from_dict(data, **kwargs)
            except ValueError as e:
                if "limit" not in str(e):
                    raise AssertionError(str(e)) from e
            else:
                raise AssertionError("REJECT aggregate cap bypass " + repr((cap, ownership)))
            assert not observed, observed
            print("PASS aggregate preallocation", ownership, cap, flush=True)
    p.functional_recipe_from_dict(payload)
    assert not observed, observed
(OUT / "root_compatibility_results.json").write_text(json.dumps(results, indent=2) + "\n")
print(
    "PASS:",
    len(results),
    (
        "pinned binary64 validator cases; 7 draw/emission combinations; 6 "
        "rejection allocation paths and valid decoding."
    ),
)
