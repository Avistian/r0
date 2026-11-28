"""Exact retained score/probability geometry from effective target emission semantics."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_intervention_records import (
    CausalInterventionRecipe,
    HardIntervention,
)
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.recursive_population_ancestors import PopulationAncestors


@dataclass(frozen=True, slots=True)
class PopulationReferenceTarget:
    score_width: int
    vocabulary: tuple[str, ...]

    @property
    def probability_width(self) -> int:
        return len(self.vocabulary)


def population_reference_target(
    runtime: PopulationAncestors, change: CausalInterventionRecipe
) -> PopulationReferenceTarget:
    if isinstance(runtime, FunctionalCausalRecipe):
        # Calibration and task adapters preserve latent score columns. Binary
        # adapters expose one latent score and two class probabilities.
        return PopulationReferenceTarget(
            len(runtime.target), tuple(f"class-{i}" for i in range(runtime.task.classes))
        )
    if isinstance(runtime, NativeCausalRecipe):
        node, _ = runtime.target
        target = runtime.nodes[node]
        domain = target.domain
        if domain.kind == "continuous":
            return PopulationReferenceTarget(1, ())
        forced = any(
            isinstance(cut, HardIntervention) and cut.node == node and cut.category is not None
            for cut in change.interventions
        )
        # A scalar native category cut bypasses threshold/logit evaluation and
        # records the complete category indicator, including for threshold nodes.
        width = len(domain.vocabulary) if forced else target.equation.function.output_width
        return PopulationReferenceTarget(width, domain.vocabulary)
    if isinstance(runtime, NativeVectorRecipe):
        node, variable = runtime.target
        target = runtime.nodes[node]
        domain, span = target.domains[variable], target.spans[variable]
        # Vector forced scores preserve the declared span, including a scalar
        # threshold score. Other variables in this node are not target columns.
        return PopulationReferenceTarget(
            1 if domain.kind == "continuous" else span.score_stop - span.score_start,
            domain.vocabulary,
        )
    assert isinstance(runtime, ConvertedCausalRecipe)
    target = runtime.nodes[runtime.target.node].feature(runtime.target)
    # Numeric readout is scalar; captured converters expose one score per label
    # (distance/logit, or forced indicator), never their representation width.
    return PopulationReferenceTarget(max(1, len(target.vocabulary)), target.vocabulary)
