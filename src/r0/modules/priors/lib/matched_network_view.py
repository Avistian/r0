"""Versioned calibrated views leave original v1 input-view archives unchanged."""

from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_fit_decoding import decode_fitted_raw_draw
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array, fitted_object
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.network_input_view import NetworkInputView, network_view_from_dict
from r0.modules.priors.lib.network_recipe import network_recipe_from_dict
from r0.modules.priors.lib.network_variance_match import NetworkVarianceMatch, admit_matching
from r0.modules.priors.lib.source_root_base import SourceRootBase


@dataclass(frozen=True, slots=True)
class MatchedNetworkInputView(NetworkInputView):
    matching: NetworkVarianceMatch | None = None
    version: str = "network-input-view-v2"

    def __post_init__(self) -> None:
        if (
            self.version != "network-input-view-v2"
            or type(self.matching) is not NetworkVarianceMatch
        ):
            raise ValueError("matched input view requires its v2 independent calibration")
        NetworkInputView(self.base, self.task, self.active_inputs)
        if self.matching.base != self.base or self.matching.after_inputs != self.active_inputs:
            raise ValueError("matching calibration differs from base archive or after mask")
        self.admit()

    def admit(self, *, rows: int | None = None) -> dict[str, int]:
        report = NetworkInputView.admit(self, rows=rows)
        parameters, storage = function_record_cost(self.matching, self.limits)
        if report["parameters"] + parameters > self.limits.parameters:
            raise ValueError("matched input view retained parameter limit exceeded")
        work = report["estimated_working_bytes"] + 4 * storage
        if work > self.limits.working_bytes:
            raise ValueError("matched input view retained population memory limit exceeded")
        return {
            **report,
            "parameters": report["parameters"] + parameters,
            "estimated_working_bytes": work,
        }


def matching_from_dict(data: object, *, limits: CausalLimits) -> NetworkVarianceMatch:
    admit_function_metadata(data, limits)
    try:
        fields = fitted_object(data, NetworkVarianceMatch)
        fields["base"] = network_recipe_from_dict(fields["base"], limits=limits)
        admit_matching(fields["base"], fields["rows"])
        width = fields["base"].function.input_width
        outputs = fields["base"].function.output_width
        rows = fields["rows"]
        for name in ("before_inputs", "after_inputs"):
            fields[name] = fitted_array(fields[name], width)
        for name in ("before_mean", "before_std", "after_mean", "after_std"):
            fields[name] = fitted_array(fields[name], outputs)
        for name in ("population", "before_scores", "after_scores"):
            columns = width if name == "population" else outputs
            fields[name] = tuple(
                fitted_array(row, columns) for row in fitted_array(fields[name], rows)
            )
        for name in ("initial", "final"):
            fields[name] = GeneratorSnapshot(**fitted_object(fields[name], GeneratorSnapshot))
        raw_count = width if isinstance(fields["base"].root, SourceRootBase) else 2
        fields["raw"] = tuple(
            decode_fitted_raw_draw(r) for r in fitted_array(fields["raw"], raw_count)
        )
        return NetworkVarianceMatch(**fields)
    except (KeyError, TypeError, OverflowError, RecursionError) as error:
        raise ValueError("malformed independent network matching record") from error


def matched_view_from_dict(
    data: dict[str, Any], *, limits: CausalLimits
) -> MatchedNetworkInputView:
    admit_function_metadata(data, limits)
    if (
        set(data) != {"base", "task", "active_inputs", "version", "matching"}
        or data["version"] != "network-input-view-v2"
    ):
        raise ValueError("matched network view fields or version differ")
    before = network_view_from_dict(
        {k: v for k, v in data.items() if k != "matching"} | {"version": "network-input-view-v1"},
        limits=limits,
    )
    matching = matching_from_dict(data["matching"], limits=limits)
    return MatchedNetworkInputView(
        before.base, before.task, before.active_inputs, matching=matching
    )
