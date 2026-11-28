"""Actual synthetic parent fitting across four authored graph families."""

from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_evaluation import FrozenFunction
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.observation_envelope_admission import observation_metadata_cost
from r0.modules.priors.lib.raw_draw_capture import raw_transactions
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.recursive_population_ancestors import (
    ANCESTOR_TYPES,
    PopulationAncestors,
    population_ancestor_values,
)
from r0.modules.priors.lib.recursive_population_graph_plan import population_equations
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.xgboost_population_fit import (
    PopulationXGBoostFit,
    fit_population_xgboost,
)
from r0.modules.priors.lib.xgboost_population_plan import PopulationXGBoostPlan

_DEFAULT_RAW_LIMITS = RawDrawLimits()


def replace_population_function(base: PopulationAncestors, node: int, function: FrozenFunction):
    nodes = list(base.nodes)
    original = nodes[node]
    nodes[node] = (
        replace(original, function=function)
        if isinstance(original, FunctionalCausalNode)
        else replace(original, equation=replace(original.equation, function=function))
    )
    return replace(base, nodes=tuple(nodes))


def fit_population_base(base: PopulationAncestors, population: str, rows: int):
    if population != "independent-ancestors":
        return base
    return replace(
        base, task=replace(base.task, rows=rows, context_rows=min(base.task.context_rows, rows - 1))
    )


def population_fit_row_ids(
    base: PopulationAncestors, population: str, rows: int, seed: int, task_id: int
) -> tuple[int, ...]:
    if population != "support-only":
        return tuple(range(rows))
    from r0.modules.priors.lib.converted_evaluation import converted_rng
    from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
    from r0.modules.priors.lib.native_evaluation import native_rng
    from r0.modules.priors.lib.native_vector_evaluation import native_vector_rng

    rngs = dict(
        zip(
            ANCESTOR_TYPES,
            (functional_rng, native_rng, native_vector_rng, converted_rng),
            strict=True,
        )
    )
    order = rngs[type(base)](seed, task_id, "split", base.task.split).permutation(base.task.rows)
    return tuple(map(int, order[: base.task.context_rows]))


def admit_xgboost_population_graph(
    base: PopulationAncestors,
    plans: tuple[tuple[int, PopulationXGBoostPlan], ...],
    population: str,
    seed: int,
    task_id: int,
    limits: CausalLimits,
    raw_limits: RawDrawLimits,
    retained: object = None,
) -> None:
    if (
        type(base) not in ANCESTOR_TYPES
        or population not in ("support-only", "batch-refit-reference", "independent-ancestors")
        or any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id))
    ):
        raise ValueError(
            "population XGBoost requires authored graph, lifecycle and exact fit identity"
        )
    if (
        type(limits) is not CausalLimits
        or type(raw_limits) is not RawDrawLimits
        or type(plans) is not tuple
        or not plans
    ):
        raise ValueError("population XGBoost needs complete plans and limits")
    equations = population_equations(base)
    seen: set[int] = set()
    cost = 0
    for pair in plans:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("population XGBoost needs node/plan pairs")
        node, plan = pair
        if (
            type(node) is not int
            or not 0 < node < len(equations)
            or node in seen
            or type(plan) is not PopulationXGBoostPlan
        ):
            raise ValueError("invalid population XGBoost node/plan")
        equation = equations[node]
        seen.add(node)
        if not equation.parents or (plan.input_width, plan.output_width) != (
            equation.function.input_width,
            equation.function.output_width,
        ):
            raise ValueError("population XGBoost parent/output geometry differs")
        rows = base.task.context_rows if population == "support-only" else base.task.rows
        if population != "independent-ancestors" and plan.rows != rows:
            raise ValueError("population XGBoost rows differ from declared lifecycle")
        if population == "independent-ancestors" and plan.rows < 16:
            raise ValueError("independent ancestor population needs at least16 rows")
        cost += (
            plan.admit(limits, raw_limits)
            + 12 * plan.model_byte_bound
            + 256 * plan.rows * (plan.input_width + plan.target_width)
        )
    if len({p.rows for _, p in plans}) != 1:
        raise ValueError("population XGBoost sequential fits need one population geometry")
    report = replace(base, limits=limits).admit()
    ancestor_work = report["estimated_working_bytes"]
    if population == "independent-ancestors":
        fit_base = fit_population_base(base, population, plans[0][1].rows)
        ancestor_work += replace(fit_base, limits=limits).admit()["estimated_working_bytes"]
    numbers, storage = observation_metadata_cost((base, plans, retained), limits)
    if (
        ancestor_work + cost + 4 * storage > limits.working_bytes
        or report["parameters"] + numbers + sum(p.retained_number_bound for _, p in plans)
        > limits.parameters
    ):
        raise ValueError("population XGBoost complete graph/fit aggregate exceeds limits")


@dataclass(frozen=True, slots=True)
class PopulationXGBoostNodeFit:
    node: int
    prefix_digest: str
    row_ids: tuple[int, ...]
    realization: PopulationXGBoostFit
    version: str = "population-xgboost-node-fit-v1"


@dataclass(frozen=True, slots=True)
class XGBoostPopulationGraph:
    base: PopulationAncestors
    fits: tuple[PopulationXGBoostNodeFit, ...]
    population: str
    fit_seed: int
    fit_task_id: int
    limits: CausalLimits
    raw_limits: RawDrawLimits
    version: str = "population-xgboost-graph-v1"

    @property
    def task(self):
        return self.base.task

    @property
    def runtime(self):
        result = self.base
        for fit in self.fits:
            result = replace_population_function(result, fit.node, fit.realization.function)
        return result

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if (
            self.version != "population-xgboost-graph-v1"
            or type(self.fits) is not tuple
            or not self.fits
            or any(
                type(f) is not PopulationXGBoostNodeFit
                or type(f.realization) is not PopulationXGBoostFit
                for f in self.fits
            )
        ):
            raise ValueError("population XGBoost graph requires complete fits")
        plans = tuple((f.node, f.realization.plan) for f in self.fits)
        admit_xgboost_population_graph(
            self.base,
            plans,
            self.population,
            self.fit_seed,
            self.fit_task_id,
            self.limits,
            self.raw_limits,
            self.fits,
        )
        if tuple(f.node for f in self.fits) != tuple(sorted({f.node for f in self.fits})):
            raise ValueError("population XGBoost fit nodes must be ordered unique")
        from r0.modules.priors.lib.xgboost_population_graph_decoding import (
            preflight_xgboost_population_graph,
        )

        preflight_xgboost_population_graph(asdict(self), limits=self.limits)
        runtime = self.base
        for index, fit in enumerate(self.fits):
            if (
                fit.version != "population-xgboost-node-fit-v1"
                or fit.realization.population != self.population
                or fit.realization.limits != self.limits
                or fit.realization.raw_limits != self.raw_limits
            ):
                raise ValueError("population XGBoost fit lifecycle/limits differ")
            if index and self.fits[index - 1].realization.final_rng != fit.realization.initial_rng:
                raise ValueError("population XGBoost stream chain differs")
            digest = record_fingerprint([asdict(n) for n in runtime.nodes[: fit.node]])
            if fit.prefix_digest != digest:
                raise ValueError("population XGBoost prefix custody differs")
            fit.realization.validate(limits=self.limits)
            rows = population_fit_row_ids(
                runtime, self.population, fit.realization.plan.rows, self.fit_seed, self.fit_task_id
            )
            if fit.row_ids != rows:
                raise ValueError("population XGBoost fit row identities differ")
            values = population_ancestor_values(
                fit_population_base(runtime, self.population, fit.realization.plan.rows),
                seed=self.fit_seed,
                task_id=self.fit_task_id,
            )
            parents = population_equations(runtime)[fit.node].parents
            inputs = torch.cat(tuple(values.states[p] for p in parents), dim=1)[list(rows)].numpy()
            if frozen_matrix(inputs) != fit.realization.inputs:
                raise ValueError(
                    "population XGBoost retained inputs differ from actual synthetic parents"
                )
            runtime = replace_population_function(runtime, fit.node, fit.realization.function)


def fit_xgboost_population_graph(
    base: PopulationAncestors,
    plans: tuple[tuple[int, PopulationXGBoostPlan], ...],
    *,
    population: str,
    seed: int,
    task_id: int,
    target_rng: np.random.Generator,
    backend_rng: np.random.Generator,
    limits: CausalLimits | None = None,
    raw_limits: RawDrawLimits = _DEFAULT_RAW_LIMITS,
) -> XGBoostPopulationGraph:
    cap = base.limits if limits is None else limits
    admit_xgboost_population_graph(base, plans, population, seed, task_id, cap, raw_limits)
    with raw_transactions({"target": target_rng, "backend": backend_rng}, None) as (streams, _):
        runtime = base
        fits: list[PopulationXGBoostNodeFit] = []
        for node, plan in sorted(plans):
            rows = population_fit_row_ids(runtime, population, plan.rows, seed, task_id)
            values = population_ancestor_values(
                fit_population_base(runtime, population, plan.rows), seed=seed, task_id=task_id
            )
            inputs = torch.cat(
                tuple(values.states[p] for p in population_equations(runtime)[node].parents), dim=1
            )[list(rows)].numpy()
            fit = fit_population_xgboost(
                plan,
                inputs=inputs,
                population=population,
                target_rng=streams["target"],
                backend_rng=streams["backend"],
                limits=cap,
                raw_limits=raw_limits,
            )
            fits.append(
                PopulationXGBoostNodeFit(
                    node, record_fingerprint([asdict(n) for n in runtime.nodes[:node]]), rows, fit
                )
            )
            runtime = replace_population_function(runtime, node, fit.function)
        return XGBoostPopulationGraph(base, tuple(fits), population, seed, task_id, cap, raw_limits)
