"""Independent ancestral fitting for sliced children and transformed-member reducers."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_composites import FrozenMLPFunction
from r0.modules.priors.lib.frozen_function_evaluation import (
    FrozenFunction,
    evaluate_admitted_function,
)
from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_function_realization import materialize_joint_function
from r0.modules.priors.lib.joint_root_realization import RootFunctionRealization
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.parent_composition_plans import ParentFunctionPlan
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, recorded_draw


def _needs_fit(plan: JointFunctionPlan) -> bool:
    return plan.family in ("tree", "disc", "em", "mlp") or any(_needs_fit(c) for c in plan.children)


@dataclass(frozen=True, slots=True)
class ParentFunctionRealization:
    plan: ParentFunctionPlan
    children: tuple[RootFunctionRealization, ...]
    joint: RootFunctionRealization | None
    reducer: RootFunctionRealization | None
    function: FrozenFunction
    version: str = "parent-function-realization-v1"


def materialize_parent_function(
    plan: ParentFunctionPlan,
    decisions: JointSamplingRecord,
    *,
    streams: dict[str, np.random.Generator],
    capture: RawDrawLedger,
    population: Callable[[str], NDArray[np.float64]],
    limits: CausalLimits,
) -> ParentFunctionRealization:
    def fit(
        part: JointFunctionPlan, values: NDArray[np.float64] | None, identity: str
    ) -> RootFunctionRealization:
        return materialize_joint_function(
            part,
            decisions,
            rng=streams["raw"],
            fit_rng=streams["fit"],
            calibration=values,
            calibration_id=identity,
            limits=limits,
            capture=capture,
        )

    joint = None
    if plan.joint:
        identity = "/".join((*plan.operation, "joint-fit"))
        values = population("joint-fit") if _needs_fit(plan.joint) else None
        joint = fit(plan.joint, values, identity)
    children: list[RootFunctionRealization] = []
    offset = 0
    for i, child in enumerate(plan.children):
        role = f"parent:{i}-fit"
        values = (
            population(role)[:, offset : offset + child.input_width] if _needs_fit(child) else None
        )
        children.append(fit(child, values, "/".join((*plan.operation, role))))
        offset += child.input_width
    reducer = None
    if plan.reducer:
        # Fresh finalized-ancestor population, then the frozen member transforms.
        # It is independent of all member fits and every emitted/query task row.
        values = torch.tensor(population("reducer-fit"), dtype=torch.float64)
        members = [evaluate_admitted_function(joint.function, values)] if joint else []
        offset = 0
        for child in children:
            width = child.function.input_width
            members.append(
                evaluate_admitted_function(child.function, values[:, offset : offset + width])
            )
            offset += width
        reducer = fit(
            plan.reducer,
            torch.cat(members, dim=1).numpy(),
            "/".join((*plan.operation, "reducer-fit")),
        )
    weights = ()
    if plan.aggregation == "weighted":
        logits = recorded_draw(
            streams["raw"],
            capture,
            owner=plan.operation,
            role="r0-softmax-gaussian-logits-v1",
            distribution="normal",
            shape=(len(children) + int(joint is not None),),
        )
        shifted = np.exp(logits - logits.max())
        weights = tuple(float(v) for v in shifted / shifted.sum())
    reducer_function = reducer.function if reducer else None
    if reducer_function is not None and not isinstance(reducer_function, FrozenMLPFunction):
        raise ValueError("neural reduction requires a frozen MLP")
    function = (
        children[0].function
        if plan.mode == "concat"
        else FrozenParentFunction(
            tuple(c.function for c in children),
            plan.aggregation,
            joint.function if joint else None,
            weights,
            reducer_function,
        )
    )
    return ParentFunctionRealization(plan, tuple(children), joint, reducer, function)
