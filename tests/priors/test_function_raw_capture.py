"""All root/function realizations capture raw, calibration and fitting draws together."""

import copy
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
@pytest.mark.parametrize("family", ["uniform", "ball", "gaussian", "covariance"])
@pytest.mark.parametrize("function", ["lin", "quad", "gp", "tree", "disc", "em", "mlp", "prod"])
def test_all_joint_roots_capture_complete_three_stream_history(
    family: str, function: str, engine: str
) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(114), mode="local")
    plan = priors.sample_joint_root(
        context,
        "root",
        width=2,
        calibration_rows=4,
        prior=priors.JointRootPrior(
            families=(family,),
            function=priors.JointFunctionPrior(
                functions=function,
                frequencies=4,
                matrix=priors.JointMatrixPrior(families=("gaussian",)),
            ),
        ),
    )
    generator = getattr(np.random, engine)
    streams = {
        "raw": np.random.Generator(generator(115)),
        "calibration": np.random.Generator(generator(116)),
        "fit": np.random.Generator(generator(117)),
    }
    initial = copy.deepcopy(streams)
    control = copy.deepcopy(streams)
    replay = copy.deepcopy(streams)
    expected = priors.materialize_joint_root(
        plan,
        context.freeze(),
        rng=control["raw"],
        calibration_rng=control["calibration"],
        fit_rng=control["fit"],
        calibration_id="sampled-root-fit",
    )
    capture = priors.RawDrawLedger()
    actual = priors.materialize_joint_root(
        plan,
        context.freeze(),
        rng=streams["raw"],
        calibration_rng=streams["calibration"],
        fit_rng=streams["fit"],
        calibration_id="sampled-root-fit",
        capture=capture,
    )
    assert asdict(actual) == asdict(expected)
    records = capture.freeze()
    assert any(record.stream == "calibration" for record in records)
    for record in records:
        np.testing.assert_array_equal(record.replay(replay[record.stream]), record.array())
    for name in streams:
        expected_next = control[name].random(8)
        np.testing.assert_array_equal(streams[name].random(8), expected_next)
        np.testing.assert_array_equal(replay[name].random(8), expected_next)

    # Let every draw except the final one succeed before the budget failure.
    # A previously committed ledger record must survive that later failure.
    limited = priors.RawDrawLedger(limits=priors.RawDrawLimits(draws=len(records)))
    with limited.transaction(np.random.default_rng(118)) as (raw, ledger):
        ledger.draw(raw, owner=("earlier",), role="weights", distribution="normal", shape=(1,))
    earlier = limited.freeze()
    failed = copy.deepcopy(initial)
    with pytest.raises(ValueError, match="raw draw.*limit"):
        priors.materialize_joint_root(
            plan,
            context.freeze(),
            rng=failed["raw"],
            calibration_rng=failed["calibration"],
            fit_rng=failed["fit"],
            calibration_id="sampled-root-fit",
            capture=limited,
        )
    assert limited.freeze() == earlier
    for name in failed:
        np.testing.assert_array_equal(failed[name].random(8), initial[name].random(8))
