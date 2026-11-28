"""The independent calibration forest stream is keyed by its complete request and identity."""

from dataclasses import asdict
from typing import Any

import numpy as np

from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.representatives import encode, record_fingerprint


def validate_calibration_identity(fit: dict[str, Any]) -> None:
    """Called only after both populations and complete retained storage were admitted."""
    seed = int(
        record_fingerprint(
            (
                "recursive-population-dag-construction-v1",
                fit["request"],
                fit["seed"],
                fit["task_id"],
            )
        ),
        16,
    )
    expected = snapshot_generator(
        np.random.default_rng(seed), byte_limit=fit["request"]["raw_limits"]["state_bytes"]
    )
    if encode(asdict(expected)) != encode(fit["bindings"][0]["sample"]["initial"]):
        raise ValueError("calibration forest stream differs from its independent keyed identity")
