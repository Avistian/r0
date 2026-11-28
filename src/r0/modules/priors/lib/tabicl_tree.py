"""Pinned noncausal tree architecture with independent frozen XGBoost fits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.linear import LinearRecipe

if TYPE_CHECKING:
    from r0.modules.priors.lib.sampled_xgboost_graph import SampledXGBoostGraph

_DEFAULT = CausalLimits()


def sample_tabicl_tree_graph(
    task: LinearRecipe,
    *,
    architecture_rng: np.random.Generator,
    parameter_rng: np.random.Generator,
    input_rng: np.random.Generator,
    target_rng: np.random.Generator,
    backend_rng: np.random.Generator,
    fit_rows: int = 128,
    noise_std: float = 0.01,
    pre_sample_noise_std: bool = False,
    limits: CausalLimits = _DEFAULT,
) -> SampledXGBoostGraph:
    """Sample source geometry; fit each tree on an independent Gaussian table.

    This is the named independent-fit adaptation, not upstream per-batch fitting
    on actual layer inputs. Features are roots and targets are the final layer.
    """
    from r0.modules.priors.lib.activation_sampling import finite_number
    from r0.modules.priors.lib.frozen_function_primitives import FrozenLinearFunction
    from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
    from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
    from r0.modules.priors.lib.joint_root_base import JointRootBase
    from r0.modules.priors.lib.raw_draw_capture import raw_transactions
    from r0.modules.priors.lib.sampled_xgboost_graph import sample_parameterized_xgboost_graph
    from r0.modules.priors.lib.xgboost_fit_plan import XGBoostFitPlan
    from r0.modules.priors.lib.xgboost_parameter_sampling import XGBoostParameterPrior

    if (
        type(task) is not LinearRecipe
        or type(limits) is not CausalLimits
        or not finite_number(noise_std)
        or noise_std < 0
        or type(pre_sample_noise_std) is not bool
    ):
        raise ValueError("invalid TabICL tree task, noise or limits")
    outputs = task.classes if task.task_type == "multiclass" else 1
    # Validate fit geometry before copying caller streams.
    XGBoostFitPlan(input_width=task.features, output_width=outputs, rows=fit_rows)
    with raw_transactions(
        dict(
            architecture=architecture_rng,
            parameters=parameter_rng,
            inputs=input_rng,
            targets=target_rng,
            backend=backend_rng,
        ),
        None,
    ) as (streams, _):
        active = streams["architecture"]
        hidden_layers = int(active.integers(1, 3))
        hidden_width = int(active.integers(3, 10))
        widths = (task.features,) + (hidden_width,) * hidden_layers + (outputs,)
        limits.admit_dimensions(
            widths,
            ((),) + tuple((i,) for i in range(len(widths) - 1)),
            rows=task.rows,
            calibration_rows=fit_rows,
            features=task.features,
            outputs=outputs,
            extra_parameters=task.features**2
            + sum(a * b for a, b in zip(widths, widths[1:], strict=False)),
        )
        root_matrix = tuple(
            tuple(float(i == j) for j in range(task.features)) for i in range(task.features)
        )
        nodes: list[FunctionalCausalNode] = [
            FunctionalCausalNode(
                (), FrozenLinearFunction(root_matrix), JointRootBase("gaussian", task.features)
            )
        ]
        priors: list[tuple[int, XGBoostParameterPrior]] = []
        for index, width in enumerate(widths[1:], 1):
            noise = (
                (0.0,) * width
                if index == 1
                else (
                    tuple(float(v) for v in np.abs(active.normal(0, noise_std, width)))
                    if pre_sample_noise_std
                    else (noise_std,) * width
                )
            )
            matrix = tuple((0.0,) * widths[index - 1] for _ in range(width))
            nodes.append(
                FunctionalCausalNode((index - 1,), FrozenLinearFunction(matrix), noise_scale=noise)
            )
            priors.append(
                (
                    index,
                    XGBoostParameterPrior(
                        XGBoostFitPlan(
                            input_width=widths[index - 1],
                            output_width=width,
                            rows=fit_rows,
                        )
                    ),
                )
            )
        base = FunctionalCausalRecipe(
            task,
            tuple(nodes),
            tuple((0, j) for j in range(task.features)),
            tuple((len(widths) - 1, j) for j in range(outputs)),
            limits,
        )
        return sample_parameterized_xgboost_graph(
            base,
            tuple(priors),
            parameter_rng=streams["parameters"],
            input_rng=streams["inputs"],
            target_rng=streams["targets"],
            backend_rng=streams["backend"],
        )
