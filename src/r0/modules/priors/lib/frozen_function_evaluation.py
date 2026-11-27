"""Deterministic admitted function evaluation on the input tensor's device."""

import math

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_composites import (
    FrozenDiscretizationFunction,
    FrozenEMFunction,
    FrozenMLPFunction,
    FrozenProductFunction,
)
from r0.modules.priors.lib.frozen_function_primitives import (
    FrozenLinearFunction,
    FrozenQuadraticFunction,
    FrozenRFFFunction,
    FrozenTreeFunction,
)

type FrozenFunction = (
    FrozenLinearFunction
    | FrozenQuadraticFunction
    | FrozenTreeFunction
    | FrozenDiscretizationFunction
    | FrozenRFFFunction
    | FrozenEMFunction
    | FrozenProductFunction
    | FrozenMLPFunction
)

_FUNCTION_TYPES = (
    FrozenLinearFunction,
    FrozenQuadraticFunction,
    FrozenTreeFunction,
    FrozenDiscretizationFunction,
    FrozenRFFFunction,
    FrozenEMFunction,
    FrozenProductFunction,
    FrozenMLPFunction,
)
_DEFAULT_LIMITS = CausalLimits()


def _geometry(function: FrozenFunction) -> tuple[tuple[int, ...], int, int]:
    widths = (function.input_width, function.output_width)
    if type(function) is FrozenLinearFunction:
        return widths, 4 * sum(widths), 0
    if type(function) is FrozenQuadraticFunction:
        k = len(function.indices) + 1
        return (*widths, k), 8 * function.output_width * k + 4 * sum(widths), 0
    if type(function) is FrozenTreeFunction:
        t, h = len(function.features), len(function.features[0])
        return widths, 4 * t * (h + function.output_width) + 4 * sum(widths), 0
    if type(function) is FrozenRFFFunction:
        k = len(function.phases)
        return (*widths, k), 4 * k + 4 * sum(widths), 0
    if type(function) is FrozenProductFunction:
        left, left_work, left_floor = _geometry(function.left)
        right, right_work, right_floor = _geometry(function.right)
        return (
            (*left, *right),
            left_work + right_work + 4 * function.output_width,
            max(left_floor, right_floor),
        )
    if type(function) is FrozenMLPFunction:
        geometries = tuple(_geometry(layer) for layer in function.linears)
        dimensions = tuple(w for geometry, _, _ in geometries for w in geometry)
        work = sum(working for _, working, _ in geometries)
        work += sum(a.working_extra_channels for a in function.activations)
        return dimensions, work, 0
    assert isinstance(function, (FrozenDiscretizationFunction, FrozenEMFunction))
    c = len(function.centers)
    return (*widths, c), 12 * c * function.input_width + 8 * c + 8 * sum(widths), c


def _distance_parts(
    inputs: torch.Tensor, centers: torch.Tensor, p: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive norm mantissa/exponent, including distances beyond binary64 range."""
    difference = inputs[:, None, :] - centers[None, :, :]
    overflow = ~torch.isfinite(difference).all(dim=-1, keepdim=True)
    if torch.any(overflow):
        half = inputs[:, None, :] / 2 - centers[None, :, :] / 2
        difference = torch.where(overflow, half, difference)
    absolute = difference.abs()
    largest = absolute.max(dim=-1).values
    largest_mantissa, exponent = torch.frexp(largest)
    coordinates, coordinate_exponents = torch.frexp(absolute)
    # Power-of-two scaling is exact, unlike division by each pair's maximum,
    # which can give unequal rounded norms to exact ties such as (15,0)/(13,2).
    relative_exponents = torch.where(absolute == 0, 0, coordinate_exponents - exponent[..., None])
    scaled = torch.ldexp(coordinates, relative_exponents)
    norm = (scaled**p).sum(dim=-1) ** (1 / p)
    mantissa, adjustment = torch.frexp(norm)
    scaled_exponent = exponent + adjustment
    # Preserve the direct equation's rounding when its powers and result are
    # representable. Rescaling ordinary fractional powers can introduce false
    # inequalities, e.g. sqrt(64) == sqrt(9) + sqrt(25).
    powers = absolute**p
    direct = powers.sum(dim=-1) ** (1 / p)
    direct_valid = torch.isfinite(direct) & ~((powers == 0) & (absolute != 0)).any(dim=-1)
    direct_mantissa, direct_exponent = torch.frexp(direct)
    mantissa = torch.where(direct_valid, direct_mantissa, mantissa)
    exponent = torch.where(direct_valid, direct_exponent, scaled_exponent)
    # A distance along one axis is its absolute coordinate for every p. Avoid
    # a needless pow/root round trip that can erase a one-ULP difference.
    one_axis = (absolute != 0).sum(dim=-1) <= 1
    mantissa = torch.where(one_axis, largest_mantissa, mantissa)
    largest_exponent = torch.frexp(largest)[1]
    exponent = torch.where(one_axis, largest_exponent, exponent)
    exponent = exponent + overflow.squeeze(-1).to(exponent.dtype)
    return mantissa, exponent


def _closest(inputs: torch.Tensor, centers: torch.Tensor, p: float) -> torch.Tensor:
    mantissa, exponent = _distance_parts(inputs, centers, p)
    # Comparing logarithms can collapse adjacent represented distances into ties.
    order = torch.where(mantissa == 0, -(2**30), exponent)
    eligible = order == order.min(dim=1, keepdim=True).values
    return torch.where(eligible, mantissa, math.inf).argmin(dim=1)


def _log_distances(inputs: torch.Tensor, centers: torch.Tensor, p: float) -> torch.Tensor:
    mantissa, exponent = _distance_parts(inputs, centers, p)
    return mantissa.log() + exponent.to(inputs.dtype) * math.log(2)


def _evaluate(function: FrozenFunction, inputs: torch.Tensor) -> torch.Tensor:
    if type(function) is FrozenLinearFunction:
        result = inputs @ inputs.new_tensor(function.matrix).T
    elif type(function) is FrozenQuadraticFunction:
        selected = inputs[:, function.indices]
        argument = torch.cat((selected, inputs.new_ones((len(inputs), 1))), dim=1)
        result = torch.einsum(
            "oij,bi,bj->bo", inputs.new_tensor(function.matrices), argument, argument
        )
    elif type(function) is FrozenTreeFunction:
        t, h = len(function.features), len(function.features[0])
        features = torch.tensor(function.features, dtype=torch.long, device=inputs.device)
        sides = inputs[:, features] > inputs.new_tensor(function.thresholds)
        bits = 2 ** torch.arange(h, dtype=torch.long, device=inputs.device)
        indices = (sides.to(dtype=torch.long) * bits).sum(dim=2)
        tree_indices = torch.arange(t, dtype=torch.long, device=inputs.device).expand(
            len(inputs), t
        )
        values = inputs.new_tensor(function.leaves)[tree_indices, indices]
        result = values.mean(dim=1)
        if not torch.isfinite(result).all():
            magnitude = values.abs().max(dim=1).values
            divisor = torch.where(magnitude == 0, torch.ones_like(magnitude), magnitude)
            stable = (values / divisor[:, None, :]).mean(dim=1) * divisor
            result = torch.where(torch.isfinite(result), result, stable)
    elif type(function) is FrozenRFFFunction:
        argument = inputs @ inputs.new_tensor(function.frequencies) + inputs.new_tensor(
            function.phases
        )
        result = argument.cos() @ inputs.new_tensor(function.coefficients)
    elif type(function) is FrozenDiscretizationFunction:
        centers = inputs.new_tensor(function.centers)
        indices = _closest(inputs, centers, function.p)
        targets = _evaluate(function.linear, centers)
        result = targets[indices]
    elif type(function) is FrozenProductFunction:
        result = _evaluate(function.left, inputs) * _evaluate(function.right, inputs)
    elif type(function) is FrozenMLPFunction:
        result, index = inputs, 0
        if function.start_activation:
            result = function.activations[index].apply(result)
            index += 1
        for i, linear in enumerate(function.linears):
            result = _evaluate(linear, result)
            if i < len(function.linears) - 1 or function.end_activation:
                result = function.activations[index].apply(result)
                index += 1
    else:
        assert isinstance(function, FrozenEMFunction)
        logs = _log_distances(inputs, inputs.new_tensor(function.centers), function.p)
        scales = inputs.new_tensor(function.scales).log()
        penalties = (function.q * (logs - scales)).exp()
        logits = -0.5 * math.log(2 * math.pi) - scales - penalties
        result = _evaluate(function.linear, logits.softmax(dim=1))
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite frozen function output; evaluation was not retried")
    return result


def admit_frozen_function(
    function: FrozenFunction, rows: int, limits: CausalLimits
) -> dict[str, int]:
    if type(function) not in _FUNCTION_TYPES or type(limits) is not CausalLimits:
        raise ValueError("function admission requires a frozen record and caller-owned limits")
    if type(rows) is not int or not 1 <= rows < 2**63:
        raise ValueError("function evaluation rows must be positive integers below 2**63")
    widths, working, floor = _geometry(function)
    return limits.admit_dimensions(
        widths,
        tuple(() for _ in widths),
        rows=rows,
        calibration_rows=0,
        features=function.input_width,
        outputs=function.output_width,
        extra_parameters=function.parameter_count,
        extra_working_channels=working,
        row_floor=floor,
    )


def evaluate_frozen_function(
    function: FrozenFunction,
    inputs: torch.Tensor,
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> torch.Tensor:
    if type(function) not in _FUNCTION_TYPES or type(limits) is not CausalLimits:
        raise ValueError("function evaluation requires a frozen record and caller-owned limits")
    if (
        type(inputs) is not torch.Tensor
        or inputs.ndim != 2
        or not inputs.is_floating_point()
        or len(inputs) < 1
        or inputs.shape[1] != function.input_width
    ):
        raise ValueError("function inputs must be nonempty floating rows with the declared width")
    admit_frozen_function(function, len(inputs), limits)
    if not torch.isfinite(inputs).all():
        raise ValueError("function inputs must be finite")
    return _evaluate(function, inputs.to(dtype=torch.float64))
