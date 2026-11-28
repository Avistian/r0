"""Sample actual XGBoost fits while retaining parameter priors in graph archives."""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array, fitted_object
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.raw_draw_capture import raw_transactions
from r0.modules.priors.lib.xgboost_fit_graph import (
    XGBoostFittedGraph,
    XGBoostNodeFit,
    with_xgboost_fits,
)
from r0.modules.priors.lib.xgboost_fitting import fit_xgboost
from r0.modules.priors.lib.xgboost_graph_admission import (
    reserve_fit_workspace,
    xgboost_raw_record_cost,
)
from r0.modules.priors.lib.xgboost_parameter_decoding import xgboost_parameter_prior_from_dict
from r0.modules.priors.lib.xgboost_parameter_sampling import (
    SampledXGBoostPlan,
    XGBoostParameterPrior,
    sample_xgboost_plan,
)


def _prior_workspace(prior: XGBoostParameterPrior, limits: CausalLimits) -> int:
    prior.admit(limits)
    return (
        replace(prior.base, depth=prior.depth_cap, estimators=prior.count_cap).admit(
            limits, prior.raw_limits
        )
        + 8 * prior.raw_limits.state_bytes
        + 4096
    )


def raw_parameter_reservation(value: Any, limits: CausalLimits) -> CausalLimits:
    pairs = fitted_array(value, limits.nodes)
    if not pairs:
        raise ValueError("sampled XGBoost graph requires parameter custody")
    replay = 0
    for pair in pairs:
        entry = fitted_array(pair, 2)
        if len(entry) != 2 or type(entry[0]) is not int:
            raise ValueError("invalid sampled XGBoost node/parameter pair")
        record = fitted_object(entry[1], SampledXGBoostPlan)
        prior = xgboost_parameter_prior_from_dict(record["prior"], limits=limits)
        replay += _prior_workspace(prior, limits)
    numbers, storage = xgboost_raw_record_cost(pairs)
    return reserve_fit_workspace(limits, numbers, storage, replay)


@dataclass(frozen=True, slots=True)
class SampledXGBoostGraph(XGBoostFittedGraph):
    parameter_samples: tuple[tuple[int, SampledXGBoostPlan], ...] = ()
    version: str = "sampled-xgboost-functional-causal-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "sampled-xgboost-functional-causal-v1"
            or type(self.parameter_samples) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not int
                or type(item[1]) is not SampledXGBoostPlan
                for item in self.parameter_samples
            )
            or tuple(i for i, _ in self.parameter_samples) != tuple(f.node for f in self.fits)
        ):
            raise ValueError("sampled XGBoost graph requires ordered parameter/fit bindings")
        for (_, sample), fit in zip(self.parameter_samples, self.fits, strict=True):
            if sample.plan != fit.realization.plan:
                raise ValueError("sampled XGBoost plan differs from actual fit binding")
        for (_, left), (_, right) in zip(
            self.parameter_samples, self.parameter_samples[1:], strict=False
        ):
            if left.final_rng != right.initial_rng:
                raise ValueError("sampled XGBoost parameter stream chain differs")
        XGBoostFittedGraph.__post_init__(self)

    def admit(self, *, evaluation_rows: int | None = None) -> dict[str, int]:
        report = XGBoostFittedGraph.admit(self, evaluation_rows=evaluation_rows)
        numbers, storage = function_record_cost(self.parameter_samples, self.limits)
        replay = sum(_prior_workspace(s.prior, self.limits) for _, s in self.parameter_samples)
        report["parameters"] += numbers
        report["estimated_working_bytes"] += 2 * storage + replay
        if (
            report["parameters"] > self.limits.parameters
            or report["estimated_working_bytes"] > self.limits.working_bytes
        ):
            raise ValueError("sampled XGBoost aggregate parameter/fit working limit exceeded")
        return report


def sample_parameterized_xgboost_graph(
    recipe: FunctionalCausalRecipe,
    priors: tuple[tuple[int, XGBoostParameterPrior], ...],
    *,
    parameter_rng: np.random.Generator,
    input_rng: np.random.Generator,
    target_rng: np.random.Generator,
    backend_rng: np.random.Generator,
) -> SampledXGBoostGraph:
    if type(recipe) is not FunctionalCausalRecipe or type(priors) is not tuple or not priors:
        raise ValueError("sampled XGBoost fitting requires an authored graph and frozen priors")
    if any(
        type(pair) is not tuple
        or len(pair) != 2
        or type(pair[0]) is not int
        or type(pair[1]) is not XGBoostParameterPrior
        for pair in priors
    ):
        raise ValueError("invalid XGBoost node/prior pairs")
    if tuple(i for i, _ in priors) != tuple(sorted({i for i, _ in priors})):
        raise ValueError("XGBoost priors must select ordered unique nodes")
    report = recipe.admit()
    working = report["estimated_working_bytes"]
    parameters = report["parameters"] + function_record_cost(priors, recipe.limits)[0]
    for index, prior in priors:
        if not 0 <= index < len(recipe.nodes):
            raise ValueError("XGBoost prior node exceeds graph")
        node = recipe.nodes[index]
        if node.pipeline is not None or (prior.base.input_width, prior.base.output_width) != (
            node.function.input_width,
            node.function.output_width,
        ):
            raise ValueError("XGBoost prior geometry or pre-fitted pipeline differs")
        working += _prior_workspace(prior, recipe.limits)
        plan = prior.base
        tree_nodes = prior.count_cap * (2 ** (prior.depth_cap + 1) - 1)
        parameters += tree_nodes * (8 + plan.output_width) + 8 * plan.rows * (
            plan.input_width + plan.target_width
        )
    if parameters > recipe.limits.parameters:
        raise ValueError("sampled XGBoost construction parameter limit exceeded")
    if working > recipe.limits.working_bytes:
        raise ValueError("sampled XGBoost construction working limit exceeded")
    bindings = dict(
        parameters=parameter_rng, inputs=input_rng, targets=target_rng, backend=backend_rng
    )
    with raw_transactions(bindings, None) as (streams, _):
        samples = tuple(
            (i, sample_xgboost_plan(prior, rng=streams["parameters"], limits=recipe.limits))
            for i, prior in priors
        )
        fits = tuple(
            XGBoostNodeFit(
                i,
                fit_xgboost(
                    sample.plan,
                    input_rng=streams["inputs"],
                    target_rng=streams["targets"],
                    backend_rng=streams["backend"],
                    limits=recipe.limits,
                    raw_limits=sample.prior.raw_limits,
                ),
            )
            for i, sample in samples
        )
        base = with_xgboost_fits(recipe, fits)
        return SampledXGBoostGraph(
            base.task,
            base.nodes,
            base.observed,
            base.target,
            base.limits,
            fits=base.fits,
            parameter_samples=samples,
        )
