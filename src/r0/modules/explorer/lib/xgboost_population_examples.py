"""All authored population examples with explicit actual-parent XGBoost lifecycles."""

from dataclasses import asdict
from typing import Any

from r0.modules import priors as p
from r0.modules.explorer.lib.population_dag_examples import population_dag_examples


def xgboost_population_examples() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, example in population_dag_examples().items():
        request = example["request"]
        base = p.recipe_from_dict(request["base"])
        assert isinstance(
            base,
            (
                p.FunctionalCausalRecipe,
                p.NativeCausalRecipe,
                p.NativeVectorRecipe,
                p.ConvertedCausalRecipe,
            ),
        )
        for population in ("support-only", "batch-refit-reference", "independent-ancestors"):
            plans: list[dict[str, Any]] = []
            for binding in request["replacements"]:
                node = base.nodes[int(binding["node"])]
                equation = node if isinstance(node, p.FunctionalCausalNode) else node.equation
                rows = (
                    base.task.context_rows
                    if population == "support-only"
                    else 32
                    if population == "independent-ancestors"
                    else base.task.rows
                )
                plan = p.PopulationXGBoostPlan(
                    rows=rows,
                    input_width=equation.function.input_width,
                    output_width=equation.function.output_width,
                    estimators=2,
                    depth=2,
                )
                plans.append(dict(node=binding["node"], plan=asdict(plan)))
            result[
                f"Population XGBoost: {population} · {label.removeprefix('Population DAG: ')}"
            ] = {
                "topology": "xgboost-population",
                "base": request["base"],
                "plans": plans,
                "population": population,
                "fit_seed": 71,
                "fit_task_id": 73,
                "engine": "PCG64",
                "raw_limits": asdict(p.RawDrawLimits()),
            }
    return result
