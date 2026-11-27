"""Fit source activation stages along an independent ancestral Gaussian sample."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import torch

from r0.modules.priors.lib.calibrated_activations import calibrate_source_activation
from r0.modules.priors.lib.causal_activations import CausalActivation, activate
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.causal_nodes import ActivatedCausalNode, CausalNode, MatrixCausalNode
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.parameter_preferences import ParameterPreferences
from r0.modules.priors.lib.source_activation_sampling import SourceActivationSelections
from r0.modules.priors.lib.source_causal_prior import SourceActivationPrior, SourceStageBinding


def fit_source_nodes(
    nodes: tuple[CausalNode, ...],
    prior: SourceActivationPrior,
    selections: SourceActivationSelections,
    rescales: ParameterPreferences,
    bindings: tuple[SourceStageBinding, ...],
    noise_units: tuple[float, ...],
    rng: Callable[..., np.random.Generator],
    calibration_identity: str,
    limits: CausalLimits,
) -> tuple[ActivatedCausalNode, ...]:
    """Caller admits geometry first; no task rows, query values or labels are accepted."""
    lookup = {(b.node, b.stage): b for b in bindings}
    values: list[torch.Tensor] = []
    fitted: list[ActivatedCausalNode] = []

    def stage(i: int, name: str, sample: torch.Tensor | None) -> CausalActivation:
        binding = lookup.get((i, name))
        if binding is None:
            return CausalActivation()
        primitive = selections.selections[binding.selection].activation
        if prior.selector == "plain":
            return primitive
        assert sample is not None
        assert binding.rescale_draw is not None
        return calibrate_source_activation(
            primitive,
            sample.numpy(),
            rng=rng("shift", i, name),
            scale=rescales.draws[binding.rescale_draw].values[0],
            calibration_id=f"{calibration_identity}/{i}/{name}",
            limits=limits,
        )

    for i, node in enumerate(nodes):
        if isinstance(node, MatrixCausalNode) and (
            node.exogenous != ExogenousDistribution() or node.noise_response != NoiseResponse()
        ):
            raise ValueError("source matrix calibration requires its Gaussian innovations")
        after = noise_units[i] < prior.noise_after_probability
        if prior.selector == "plain":
            input_activation, output_activation = stage(i, "input", None), stage(i, "output", None)
        else:
            noise = np.column_stack(
                [
                    rng("innovation", i, j).normal(size=prior.calibration_rows)
                    for j in range(len(node.bias))
                ]
            )
            innovation = torch.tensor(noise, dtype=torch.float64) * torch.tensor(
                node.noise_scale, dtype=torch.float64
            )
            argument = (
                torch.zeros_like(innovation) + torch.tensor(node.bias, dtype=torch.float64)
                if after or isinstance(node, MatrixCausalNode)
                else innovation + torch.tensor(node.bias, dtype=torch.float64)
            )
            parents = torch.cat([values[p] for p in node.parents], dim=1) if node.parents else None
            input_activation = stage(i, "input", parents)
            if parents is not None:
                argument = argument + activate(parents, input_activation) @ torch.tensor(
                    node.weights, dtype=torch.float64
                )
            if isinstance(node, MatrixCausalNode) and not after:
                argument = argument + innovation
            output_activation = stage(i, "output", argument)
            value = activate(argument, output_activation)
            if after:
                value = value + innovation
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite ancestral activation calibration; task was not retried")
            values.append(value)
        fitted.append(
            replace(
                node,
                input_activation=input_activation,
                output_activation=output_activation,
                noise_position="after_output" if after else "before_output",
            )
            if isinstance(node, MatrixCausalNode)
            else ActivatedCausalNode(
                parents=node.parents,
                weights=node.weights,
                bias=node.bias,
                noise_scale=node.noise_scale,
                input_activation=input_activation,
                output_activation=output_activation,
                noise_position="after_output" if after else "before_output",
            )
        )
    return tuple(fitted)
