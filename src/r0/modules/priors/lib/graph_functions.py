"""Family-specific graph steps; generation mathematics stays in the shared evaluators."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.exact_gp import GaussianProcessRecipe
from r0.modules.priors.lib.generate import recipe_from_dict
from r0.modules.priors.lib.kernels import kernel_covariance
from r0.modules.priors.lib.signals import (
    combine_signals,
    component_inputs,
    evaluate_mechanism,
    multiply_signals,
)


def _normalize(
    card: dict[str, Any], raw: torch.Tensor, parent: str, add: Callable[..., None]
) -> torch.Tensor:
    if card["version"] == "linear-v1":
        signal = raw
        parameters = {"expected_signal_variance": card["expected_signal_variance"]}
        formula = "raw signal (recorded coefficients already have unit L2 norm)"
    else:
        signal = (raw - torch.tensor(card["center"], dtype=torch.float64)) / torch.tensor(
            card["scale"], dtype=torch.float64
        )
        parameters = {
            key: card[key] for key in ("center", "scale", "normalization", "calibration_rows")
        }
        formula = "(raw signal - independent calibration center) / calibration scale"
    add("signal", "Normalized latent signal", [parent], formula, signal, parameters)
    return signal


def analytic_steps(card: dict[str, Any], x: torch.Tensor, add: Callable[..., None]) -> torch.Tensor:
    def tensor(key: str) -> torch.Tensor:
        return torch.tensor(card[key], dtype=torch.float64)

    version = card["version"]
    parents = ["inputs"]
    if version in ("linear-v1", "grouped-linear-v1"):
        weights = tensor("weights")
        add(
            "weights",
            "Saved coefficients",
            [],
            "recorded coefficient matrix",
            weights,
            {
                "axes": "signal feature by output channel",
                "active": card["active"],
                "groups": card.get("groups"),
                "selected_groups": card.get("selected_groups"),
            },
        )
        parents += ["weights"]
        formula = "input signal columns @ coefficients"
    elif version == "polynomial-v1":
        basis = x[:, : len(card["powers"])] ** torch.tensor(card["powers"])
        add(
            "basis",
            "Integer powers",
            ["inputs"],
            "input_j ** power_j",
            basis,
            {"powers": card["powers"]},
        )
        add(
            "weights",
            "Saved coefficients",
            [],
            "recorded coefficient matrix",
            tensor("weights"),
            {"axes": "powered feature by output channel"},
        )
        parents, formula = ["basis", "weights"], "powered inputs @ coefficients"
    elif version == "quadratic-v1":
        projection = x[:, : len(card["directions"][0])] @ tensor("directions").T
        add(
            "projection",
            "Quadratic directions",
            ["inputs"],
            "inputs @ directions.T",
            projection,
            {"directions": card["directions"]},
        )
        add("squared", "Squared projections", ["projection"], "projection ** 2", projection**2, {})
        add(
            "squared-inputs",
            "Diagonal terms",
            ["inputs"],
            "signal inputs ** 2",
            x[:, : len(card["directions"][0])] ** 2,
            {},
        )
        parents = ["inputs", "squared", "squared-inputs"]
        formula = (
            "interaction_strength * (projection² @ weights) + "
            "(1-interaction_strength) * (inputs² @ ((directions²).T @ weights)) + "
            "linear_strength * (inputs @ linear_weights)"
        )
    elif version == "fourier-v1":
        phase = x[:, : len(card["frequencies"][0])] @ tensor("frequencies").T + tensor("phases")
        add(
            "phase",
            "Frequency projection and phase",
            ["inputs"],
            "inputs @ frequencies.T + phases",
            phase,
            {"frequencies": card["frequencies"], "phases": card["phases"]},
        )
        add("basis", "Cosine features", ["phase"], "cos(phase)", torch.cos(phase), {})
        parents, formula = ["basis"], "cosine features @ coefficients"
    elif version == "smooth-v1":
        projection = x[:, : len(card["directions"][0])] @ tensor("directions").T
        add(
            "projection",
            "Feature projections",
            ["inputs"],
            "inputs @ directions.T",
            projection,
            {"directions": card["directions"]},
        )
        z = (projection - tensor("centers")) / tensor("widths")
        add(
            "arguments",
            "Shift and scale",
            ["projection"],
            "(projection - centers) / widths",
            z,
            {"centers": card["centers"], "widths": card["widths"]},
        )
        sinusoidal = card["family"].endswith("sinusoidal")
        basis = torch.sin(z) if sinusoidal else torch.exp(-0.5 * z.square())
        add(
            "basis",
            "Smooth components",
            ["arguments"],
            "sin(z)" if sinusoidal else "exp(-z²/2)",
            basis,
            {},
        )
        parents, formula = ["basis"], "smooth components @ coefficients"
    else:
        raise ValueError(f"unsupported analytic graph version: {version}")
    # Identity normalization exposes raw evaluator output without inventing another
    # generator or resampling. Linear-v1 ignores these fields by construction.
    raw = evaluate_mechanism({**card, "center": [0.0], "scale": [1.0]}, x)
    parameters = {
        key: card[key]
        for key in (
            "weights",
            "coefficients",
            "linear_weights",
            "interaction_strength",
            "linear_strength",
        )
        if key in card
    }
    add("raw", "Raw mechanism output", parents, formula, raw, parameters)
    return _normalize(card, raw, "raw", add)


def composite_steps(
    card: dict[str, Any], x: torch.Tensor, add: Callable[..., None]
) -> torch.Tensor:
    leaves: list[torch.Tensor] = []
    for i, child in enumerate(card["components"]):
        mapped = component_inputs(child, x)
        value = multiply_signals(evaluate_mechanism(child["mechanism"], mapped), child["strength"])
        leaves.append(value)
        add(
            f"component-{i}",
            f"Component {i}: {child['mechanism']['family']}",
            ["inputs"],
            "strength * saved child function(mapped inputs)",
            value,
            {
                "input_columns": child["input_columns"],
                "child_active": child["mechanism"]["active"],
                "strength": child["strength"],
                "seed": child["seed"],
            },
        )
    counter = 0

    def expression(node: Any) -> tuple[str, torch.Tensor]:
        nonlocal counter
        if type(node) is int:
            return f"component-{node}", leaves[node]
        key = f"expression-{counter}"
        counter += 1
        children = [expression(child) for child in node["children"]]
        result = combine_signals(
            [value for _, value in children], node["operation"], node["strength"]
        )
        add(
            key,
            node["operation"].capitalize(),
            [key for key, _ in children],
            f"strength * {node['operation']}(children in recorded order)",
            result,
            {"strength": node["strength"]},
        )
        return key, result

    key, raw = expression(card["expression"])
    return _normalize(card, raw, key, add)


def gp_steps(
    card: dict[str, Any],
    recipe_record: dict[str, Any],
    indices: Sequence[int],
    add: Callable[..., None],
) -> torch.Tensor:
    recipe = recipe_from_dict(recipe_record)
    if not isinstance(recipe, GaussianProcessRecipe):
        raise ValueError("direct GP graph requires its original GP recipe")
    support = np.asarray(card["support"], dtype=np.float64)
    task_indices = np.asarray(card["task_support_indices"], dtype=np.int64)[list(indices)]
    selected = support[task_indices]
    for i, kernel in enumerate(recipe.kernels):
        values, _ = kernel_covariance(selected, (kernel,), "sum")
        parameters = {
            **card["kernels"][i],
            "value_axes": "selected row by selected row",
            "row_indices": list(indices),
            "weight_application": "unweighted here; weight applied by combined covariance",
        }
        if kernel.kind == "gibbs":
            parameters.update(
                local_length="l_d(x) = length_scale_d * exp(length_slope_d * modulation_d(x))",
                amplitude="A(x) = exp(sum_d amplitude_slope_d * modulation_d(x))",
            )
        else:
            parameters["distance"] = "r² = sum_d (delta_d / length_scale_d)² on kernel axes"
        add(
            f"kernel-{i}",
            f"Kernel {i}: {kernel.kind}",
            ["inputs"],
            {
                "rbf": "exp(-r²/2)",
                "matern12": "exp(-r)",
                "matern32": "(1+sqrt(3)*r)*exp(-sqrt(3)*r)",
                "matern52": "(1+sqrt(5)*r+5*r²/3)*exp(-sqrt(5)*r)",
                "rational_quadratic": "(1+r²/(2*alpha))**(-alpha)",
                "spectral": "exp(-r²/2) * product_d cos(2*pi*frequency_d*delta_d)",
                "gibbs": (
                    "A(x)*A(x') * product_d [sqrt(2*l_d(x)*l_d(x')/(l_d(x)²+l_d(x')²)) "
                    "* exp(-delta_d²/(l_d(x)²+l_d(x')²))]"
                ),
            }[kernel.kind],
            torch.tensor(values),
            parameters,
        )
    covariance, _ = kernel_covariance(support, recipe.kernels, recipe.combine)
    add(
        "covariance",
        "Combined covariance",
        [f"kernel-{i}" for i in range(len(recipe.kernels))],
        "sum(weight_i * K_i)"
        if recipe.combine == "sum"
        else "product(K_i), using signed logs for Gibbs combinations",
        torch.tensor(covariance[np.ix_(task_indices, task_indices)]),
        {
            "full_shape": list(covariance.shape),
            "display": "selected task-row covariance submatrix",
            "row_indices": list(indices),
        },
    )
    eigenvalues, vectors = np.linalg.eigh(covariance)
    factor = (vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ vectors.T
    normals = np.asarray(card["standard_normal"], dtype=np.float64)
    add(
        "normals",
        "Saved joint standard normals",
        [],
        "original standard-normal matrix Z",
        torch.tensor(normals[:16]),
        {
            "full_shape": list(normals.shape),
            "support_indices_shown": list(range(min(16, len(normals)))),
            "display": "first up to 16 support points, not task rows",
        },
    )
    raw = torch.tensor(factor[task_indices]) @ torch.tensor(normals)
    add(
        "joint-draw",
        "Original joint GP draw",
        ["covariance", "normals"],
        "raw = symmetric_sqrt(full K)[task support indices, :] @ full Z",
        raw,
        {
            "value_kind": "original_joint_draw",
            "factorization": card["factorization"],
            "roundoff_tolerance": card["roundoff_tolerance"],
            "task_support_indices": task_indices.tolist(),
            "full_support_count": len(support),
            "additional_values_sampled": False,
        },
    )
    return _normalize(card, raw, "joint-draw", add)
