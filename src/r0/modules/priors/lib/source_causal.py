"""Sampled source activation stages with replay-checked ancestral calibration."""

from dataclasses import asdict, dataclass
from typing import Any

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.calibrated_activations import CalibratedActivation
from r0.modules.priors.lib.categorical_preferences import CategoricalPreferences
from r0.modules.priors.lib.causal import CausalRecipe, causal_rng
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_nodes import ActivatedCausalNode
from r0.modules.priors.lib.parameter_preferences import ParameterPreferences
from r0.modules.priors.lib.source_activation_sampling import SourceActivationSelections
from r0.modules.priors.lib.source_causal_calibration import fit_source_nodes
from r0.modules.priors.lib.source_causal_prior import (
    SourceActivationPrior,
    SourceStageBinding,
    rescale_request,
    source_stages,
)

_EMPTY = SourceActivationSelections((), ParameterPreferences(), CategoricalPreferences())


@dataclass(frozen=True, slots=True)
class SourceCausalRecipe(CausalRecipe):
    source_prior: SourceActivationPrior = SourceActivationPrior()
    source_selections: SourceActivationSelections = _EMPTY
    rescale_preferences: ParameterPreferences = ParameterPreferences()
    source_bindings: tuple[SourceStageBinding, ...] = ()
    noise_units: tuple[float, ...] = ()
    calibration_seed: int = 0
    calibration_task_id: int = 0

    @property
    def admission_calibration_rows(self) -> int:
        return max(
            super(SourceCausalRecipe, self).admission_calibration_rows,
            self.source_prior.calibration_rows if self.source_prior.selector == "standard" else 0,
        )

    @property
    def sampling_parameter_count(self) -> int:
        return (
            4
            + len(self.noise_units)
            + self.source_selections.parameter_count
            + self.rescale_preferences.parameter_count
            + sum(b.parameter_count for b in self.source_bindings)
        )

    @property
    def sampling_record(self) -> dict[str, Any]:
        return {
            "source_prior": asdict(self.source_prior),
            "source_selections": asdict(self.source_selections),
            "rescale_preferences": asdict(self.rescale_preferences),
            "source_bindings": tuple(asdict(b) for b in self.source_bindings),
            "noise_units": self.noise_units,
            "calibration_seed": self.calibration_seed,
            "calibration_task_id": self.calibration_task_id,
            "calibration_stream": "source-activation-calibration-v1",
        }

    def __post_init__(self) -> None:
        if (
            self.version != "causal-v2"
            or self.construction != "source-activations-v1"
            or type(self.source_prior) is not SourceActivationPrior
            or type(self.source_selections) is not SourceActivationSelections
            or type(self.rescale_preferences) is not ParameterPreferences
            or type(self.source_bindings) is not tuple
            or any(type(b) is not SourceStageBinding for b in self.source_bindings)
        ):
            raise ValueError("source causal recipes require frozen priors, selections and bindings")
        if any(
            type(i) is not int or not 0 <= i < 2**63
            for i in (
                self.calibration_seed,
                self.calibration_task_id,
            )
        ):
            raise ValueError("source calibration stream identities must lie in [0,2**63)")
        if (
            type(self.noise_units) is not tuple
            or len(self.noise_units) != len(self.nodes)
            or any(not finite_number(u) or not 0 <= u < 1 for u in self.noise_units)
        ):
            raise ValueError("source noise-order units must cover every node")
        prior = self.source_prior
        if (
            self.source_selections.numerical.default_mode != prior.mode
            or self.source_selections.categorical.default_mode != prior.mode
            or self.source_selections.categorical.policy != prior.policy
            or self.rescale_preferences.default_mode != prior.mode
            or self.rescale_preferences.global_policy != "strict"
        ):
            raise ValueError("source activation preference policies disagree with their prior")
        CausalRecipe.__post_init__(self)
        if any(type(n) is not ActivatedCausalNode for n in self.nodes):
            raise ValueError("source activation construction requires Gaussian activated nodes")
        expected = source_stages(tuple(n.parents for n in self.nodes), prior)
        if (
            tuple((b.node, b.stage) for b in self.source_bindings) != expected
            or tuple(b.selection for b in self.source_bindings) != tuple(range(len(expected)))
            or len(self.source_selections.selections) != len(expected)
        ):
            raise ValueError("source bindings must cover every active stage and selection in order")
        if len(self.rescale_preferences.draws) != (
            len(expected) if prior.selector == "standard" else 0
        ):
            raise ValueError("source rescale bindings must cover every standard activation")
        for index, binding in enumerate(self.source_bindings):
            node = self.nodes[binding.node]
            assert isinstance(node, ActivatedCausalNode)
            actual = node.input_activation if binding.stage == "input" else node.output_activation
            selected = self.source_selections.selections[binding.selection]
            if selected.selector != prior.selector:
                raise ValueError("source selector binding differs from its prior")
            primitive = CausalActivation(actual.kind, actual.parameters)
            if primitive != selected.activation:
                raise ValueError("source activation binding differs from its selected primitive")
            if prior.selector == "standard":
                if type(actual) is not CalibratedActivation or binding.rescale_draw != index:
                    raise ValueError("source standard binding requires its calibration and rescale")
                draw = self.rescale_preferences.draws[index]
                if draw.request != rescale_request() or draw.values[0] != actual.scale:
                    raise ValueError("source rescale binding differs from its frozen scale")
            elif type(actual) is not CausalActivation or binding.rescale_draw is not None:
                raise ValueError("plain source binding cannot contain calibration or rescale")
        active = set(expected)
        for i, node in enumerate(self.nodes):
            assert isinstance(node, ActivatedCausalNode)
            after = self.noise_units[i] < prior.noise_after_probability
            if node.noise_position != ("after_output" if after else "before_output"):
                raise ValueError("source noise order differs from its retained unit")
            for stage, activation in (
                ("input", node.input_activation),
                ("output", node.output_activation),
            ):
                if (i, stage) not in active and activation != CausalActivation():
                    raise ValueError("inactive source activation stage must be identity")
        if prior.selector == "standard":
            fitted = fit_source_nodes(
                self.nodes,
                prior,
                self.source_selections,
                self.rescale_preferences,
                self.source_bindings,
                self.noise_units,
                lambda *keys: causal_rng(
                    self.calibration_seed,
                    self.calibration_task_id,
                    "source-activation-calibration-v1",
                    *keys,
                ),
                f"source-activation-calibration-v1/{self.calibration_seed}/{self.calibration_task_id}",
                self.limits,
            )
            if fitted != self.nodes:
                raise ValueError("source calibration differs from its independent ancestral replay")
