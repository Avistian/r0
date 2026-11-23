"""Readable scalar calculations for retained #23 cards; no sampler/helper calls."""

import math
from typing import Any


def worked_row(card: dict[str, Any]) -> dict[str, Any]:
    family = card["proposal"]["family"]
    parameters, trace = card["mechanism"]["parameters"], card["trace"]
    x = card["source_rows"][0]["values"]
    steps: dict[str, Any] = {"features": x}
    if family == "linear":
        terms = [a * b for a, b in zip(x, parameters["weights"], strict=True)]
        value = sum(terms)
        steps.update(terms=terms, rule="sum(x_j * w_j)")
    elif family == "mlp":
        hidden = x
        layers = []
        for i, weights in enumerate(parameters["weights"]):
            outputs = [
                sum(hidden[j] * weights[j][k] for j in range(len(hidden)))
                for k in range(len(weights[0]))
            ]
            if i + 1 < len(parameters["weights"]):
                outputs = [
                    math.tanh(v) if parameters["activation"] == "tanh" else max(0, v)
                    for v in outputs
                ]
            layers.append(outputs)
            hidden = outputs
        value = hidden[0]
        steps.update(layers=layers, rule="scalar affine sums, hidden activation, linear output")
    elif family == "scm":
        nodes: list[float] = []
        for i, parents in enumerate(parameters["parents"]):
            weighted = sum(
                nodes[j] * w for j, w in zip(parents, parameters["weights"][i], strict=True)
            )
            nodes.append((math.tanh(weighted) if parents else 0) + trace["node_noise"][0][i])
        value = nodes[parameters["target_node"]]
        steps.update(
            nodes=nodes, parents=parameters["parents"], rule="tanh(parent sum) + node noise"
        )
    elif family == "tree":
        path = []
        node = 0
        for _ in range(parameters["depth"]):
            column, threshold = parameters["split_features"][node], parameters["thresholds"][node]
            right = x[column] > threshold
            path.append({"node": node, "column": column, "threshold": threshold, "right": right})
            node = node * 2 + 1 + int(right)
        leaf = node - len(parameters["split_features"])
        value = parameters["leaves"][leaf]
        steps.update(path=path, leaf=leaf, rule="x <= threshold goes left")
    elif family == "interaction":
        degree, rule = parameters["degree"], parameters["rule"]
        value = (
            math.prod(x[:degree])
            if rule == "product"
            else float(sum(v > 0 for v in x[:degree]) % 2)
            if rule == "parity"
            else math.prod(x[1:degree])
            if x[0] > 0
            else -1.0
        )
        steps.update(rule=rule, degree=degree)
    else:
        # Rebuild Cholesky from the declared kernel, not the recorded factor.
        knots = parameters["knots"]
        factor = [[0.0] * len(knots) for _ in knots]
        for i, a in enumerate(knots):
            for j in range(i + 1):
                b = knots[j]
                covariance = parameters["signal_variance"] * math.exp(
                    -((a - b) ** 2) / (2 * parameters["length_scale"] ** 2)
                ) + (parameters["diagonal_component_variance"] if i == j else 0)
                residual = covariance - sum(factor[i][k] * factor[j][k] for k in range(j))
                factor[i][j] = math.sqrt(residual) if i == j else residual / factor[j][j]
        index = trace["support_indices"][0]
        terms = [factor[index][j] * parameters["standard_normal"][j] for j in range(len(knots))]
        value = sum(terms)
        steps.update(
            rule="independent scalar Cholesky(K), then Lz at support index",
            index=index,
            factor=factor,
            terms=terms,
        )
    if card["proposal"]["task_type"] == "regression":
        noise = trace["target_noise"][0]
        expected: float | str = value + noise
        steps["target_adaptation"] = {"kind": "identity plus recorded noise", "noise": noise}
    else:
        adapter = card["mechanism"]["target_adapter"]
        base = adapter["mapping"][sum(value >= t for t in adapter["thresholds"])]
        final = trace["replacement_class"][0] if trace["replacement_event"][0] else base
        expected = card["episode"]["classes"][final]
        steps["target_adaptation"] = {
            "thresholds": adapter["thresholds"],
            "mapping": adapter["mapping"],
            "base": base,
            "replacement_event": trace["replacement_event"][0],
            "replacement_class": trace["replacement_class"][0],
        }
    observed = card["source_targets"][0]
    difference = None if isinstance(expected, str) else observed - expected
    passed = (
        observed == expected
        if isinstance(expected, str)
        else math.isclose(observed, expected, abs_tol=1e-10, rel_tol=1e-10)
    )
    return {
        "proposal_id": card["proposal_id"],
        "family": family,
        "steps": steps,
        "latent_expected": value,
        "latent_observed": trace["latent"][0],
        "target_expected": expected,
        "target_observed": observed,
        "difference": difference,
        "status": "PASS" if passed else "FAIL",
    }
