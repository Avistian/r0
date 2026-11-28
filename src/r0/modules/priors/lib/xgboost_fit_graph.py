"""Versioned DAG binding of independent XGBoost fits to structural equations."""

from dataclasses import asdict, dataclass, replace

from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_graph_admission import admit_functional_graph
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.xgboost_fitting import XGBoostFit, canonical_xgboost_record
from r0.modules.priors.lib.xgboost_graph_admission import reserve_fit_workspace


@dataclass(frozen=True, slots=True)
class XGBoostNodeFit:
    node: int
    realization: XGBoostFit
    version: str = "xgboost-node-fit-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "xgboost-node-fit-v1"
            or type(self.node) is not int
            or self.node < 0
            or type(self.realization) is not XGBoostFit
        ):
            raise ValueError("XGBoost node fit requires frozen construction custody")


@dataclass(frozen=True, slots=True)
class XGBoostFittedGraph(FunctionalCausalRecipe):
    fits: tuple[XGBoostNodeFit, ...] = ()
    version: str = "xgboost-fitted-functional-causal-v1"

    def __post_init__(self) -> None:
        if (
            self.version
            not in ("xgboost-fitted-functional-causal-v1", "sampled-xgboost-functional-causal-v1")
            or self.version == "sampled-xgboost-functional-causal-v1"
            and not getattr(self, "parameter_samples", ())
            or type(self.fits) is not tuple
            or not self.fits
            or any(type(f) is not XGBoostNodeFit for f in self.fits)
            or tuple(f.node for f in self.fits) != tuple(sorted({f.node for f in self.fits}))
        ):
            raise ValueError("XGBoost graph requires ordered unique fit records")
        FunctionalCausalRecipe.__post_init__(self)
        self.validate_custody()

    def admit(self, *, evaluation_rows: int | None = None) -> dict[str, int]:
        if evaluation_rows is not None and (
            type(evaluation_rows) is not int or not 1 <= evaluation_rows < 2**63
        ):
            raise ValueError("XGBoost graph evaluation rows must be positive bounded integers")
        numbers, storage = function_record_cost(self.fits, self.limits)
        replay = sum(
            f.realization.plan.admit(self.limits, f.realization.raw_limits)
            + 12 * len(f.realization.model_json)
            for f in self.fits
        )
        residual = reserve_fit_workspace(self.limits, numbers, storage, replay)
        report = admit_functional_graph(
            self.nodes,
            residual,
            rows=max(self.task.rows, evaluation_rows or 0),
            metadata=(self.task, self.nodes, self.observed, self.target, self.limits),
            features=self.task.features,
            outputs=len(self.target),
        )
        report["parameters"] += numbers
        report["estimated_working_bytes"] += 2 * storage + replay
        return report

    def validate_custody(self) -> None:
        self.admit()
        for fit in self.fits:
            if fit.node >= len(self.nodes) or canonical_xgboost_record(
                asdict(fit.realization.function)
            ) != canonical_xgboost_record(asdict(self.nodes[fit.node].function)):
                raise ValueError("XGBoost structural equation and fit binding differ")
            fit.realization.validate(limits=self.limits)


def with_xgboost_fits(
    recipe: FunctionalCausalRecipe, fits: tuple[XGBoostNodeFit, ...]
) -> XGBoostFittedGraph:
    """Attach already frozen independent fits; no fitting or RNG work occurs here.

    Root and nonroot equations are both supported, with unchanged dimensions.
    This does not assert that the independent normal fit rows are ancestor samples.
    """
    if type(recipe) is not FunctionalCausalRecipe or type(fits) is not tuple or not fits:
        raise ValueError("XGBoost attachment requires an authored skeleton and frozen fits")
    if any(type(f) is not XGBoostNodeFit for f in fits):
        raise ValueError("XGBoost attachment requires frozen node fit records")
    if tuple(f.node for f in fits) != tuple(sorted({f.node for f in fits})):
        raise ValueError("XGBoost attachment requires unique ordered node indices")
    nodes = list(recipe.nodes)
    for fit in fits:
        if fit.node >= len(nodes):
            raise ValueError("XGBoost fit node index exceeds graph")
        before = nodes[fit.node]
        after = fit.realization.function
        if (before.function.input_width, before.function.output_width) != (
            after.input_width,
            after.output_width,
        ):
            raise ValueError("XGBoost fit geometry differs from structural equation")
        if before.pipeline is not None:
            raise ValueError("XGBoost replacement requires a node without a pre-fitted pipeline")
        nodes[fit.node] = replace(before, function=after)
    return XGBoostFittedGraph(
        recipe.task, tuple(nodes), recipe.observed, recipe.target, recipe.limits, fits=fits
    )
