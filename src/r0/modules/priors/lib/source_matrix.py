"""Joint source activation and matrix preferences with complete selection bindings."""

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from r0.modules.priors.lib.categorical_preferences import CategoricalPreferences
from r0.modules.priors.lib.causal_nodes import ActivatedCausalNode, MatrixCausalNode
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.matrix_preference_prior import CausalMatrixPrior
from r0.modules.priors.lib.matrix_preferences import validate_matrix_preferences
from r0.modules.priors.lib.parameter_preferences import ParameterPreferences
from r0.modules.priors.lib.preference_binding import PreferenceBinding
from r0.modules.priors.lib.source_causal import SourceCausalRecipe
from r0.modules.priors.lib.transformed_weights import NormalizedWeights, TransformedWeights


@dataclass(frozen=True, slots=True)
class SourceMatrixCausalRecipe(SourceCausalRecipe):
    expected_construction: ClassVar[str] = "source-shared-matrices-v1"
    matrix_prior: CausalMatrixPrior = CausalMatrixPrior()
    numerical_preferences: ParameterPreferences = ParameterPreferences()
    categorical_preferences: CategoricalPreferences = CategoricalPreferences()
    preference_bindings: tuple[PreferenceBinding, ...] = ()
    matrix_activation_nodes: tuple[int, ...] = ()

    @property
    def source_selection_offset(self) -> int:
        return len(self.matrix_activation_nodes)

    @property
    def sampling_parameter_count(self) -> int:
        return (
            super(SourceMatrixCausalRecipe, self).sampling_parameter_count
            + self.numerical_preferences.parameter_count
            + self.categorical_preferences.parameter_count
            + sum(b.parameter_count for b in self.preference_bindings)
            + len(self.matrix_activation_nodes)
        )

    @property
    def sampling_record(self) -> dict[str, Any]:
        return {
            **super(SourceMatrixCausalRecipe, self).sampling_record,
            "matrix_prior": asdict(self.matrix_prior),
            "numerical_preferences": asdict(self.numerical_preferences),
            "categorical_preferences": asdict(self.categorical_preferences),
            "preference_bindings": tuple(asdict(b) for b in self.preference_bindings),
            "matrix_activation_nodes": self.matrix_activation_nodes,
            "selection_order": "transformed matrices in node order, then node input/output stages",
        }

    def __post_init__(self) -> None:
        validate_matrix_preferences(self)
        if (
            self.matrix_prior.mode != self.source_prior.mode
            or self.matrix_prior.categorical_policy != self.source_prior.policy
        ):
            raise ValueError("joint source and matrix preference policies must agree")
        if type(self.matrix_activation_nodes) is not tuple or any(
            type(i) is not int or not 0 <= i < len(self.nodes) for i in self.matrix_activation_nodes
        ):
            raise ValueError("matrix activation bindings require valid node indices")
        expected = tuple(
            i
            for i, node in enumerate(self.nodes)
            if isinstance(node, MatrixCausalNode)
            and isinstance(node.weight_construction, NormalizedWeights)
            and isinstance(node.weight_construction.source, TransformedWeights)
        )
        if self.matrix_activation_nodes != expected:
            raise ValueError("matrix activation bindings must cover transformed matrices in order")
        if len(self.source_selections.selections) < len(expected):
            raise ValueError("matrix activation bindings reference absent selections")
        for selection, index in zip(self.source_selections.selections, expected, strict=False):
            node = self.nodes[index]
            assert isinstance(node, MatrixCausalNode)
            assert isinstance(node.weight_construction, NormalizedWeights)
            assert isinstance(node.weight_construction.source, TransformedWeights)
            if (
                selection.selector != "plain"
                or selection.activation != node.weight_construction.source.activation
            ):
                raise ValueError("matrix activation differs from its shared plain selection")
        SourceCausalRecipe.__post_init__(self)

    def validate_source_nodes(self) -> None:
        for node in self.nodes:
            if not node.parents:
                if type(node) is not ActivatedCausalNode:
                    raise ValueError("joint source roots require Gaussian activated nodes")
            elif (
                type(node) is not MatrixCausalNode
                or node.exogenous != ExogenousDistribution()
                or node.noise_response != NoiseResponse()
            ):
                raise ValueError("joint source projections require Gaussian matrix nodes")
