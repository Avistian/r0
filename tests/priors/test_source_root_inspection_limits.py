"""Larger admitted source-root tasks must remain inspectable with the same limits."""

import numpy as np

from r0.modules import priors as p


def test_native_vector_source_root_inspection_preserves_larger_limits():
    width, rows = 64, 15000
    limits = p.CausalLimits(parameters=2_000_000, working_bytes=512 * 1024**2)
    law = p.SourceRootLaw((p.SourceRootCoordinate("gaussian"),) * width)
    calibration = p.SourceRootCalibration(
        law, ("identity",) * width, 32, (0.0,) * width, (1.0,) * width
    )
    network = p.FrozenNetworkFunction(
        width,
        (
            p.NetworkBlock(
                (0,), (p.NetworkAffine(((1 / width,) * width, (-1 / width,) * width), (0.0, 0.0)),)
            ),
        ),
    )
    node = p.NativeVectorNode(
        p.FunctionalCausalNode(
            (),
            network,
            p.SourceRootBase(law, calibration),
            version="functional-causal-node-v2",
        ),
        (p.VariableDomain(), p.VariableDomain("binary", ("no", "yes"), "threshold", (0.0,))),
    )
    recipe = p.NativeVectorRecipe(
        p.LinearRecipe(
            rows=rows,
            context_rows=100,
            features=1,
            signal_features=1,
            active_features=1,
            task_type="binary",
            classes=2,
        ),
        (node,),
        ((0, 0),),
        (0, 1),
        limits=limits,
    )
    task = p.generate_task(recipe, seed=41, task_id=9)
    indices = [rows - 1, 0, rows - 1]
    trace = p.explain_generation(task, indices, limits=limits)
    root = next(n for n in trace["nodes"] if n["id"] == "root-0")
    raw = np.asarray(root["values"])
    # Inspection keeps original row selection order, including duplicates.
    np.testing.assert_array_equal(raw[0], raw[2])
    assert trace["rows"] == indices
