"""Guarded causal-pair actions use the same replay and persistence contracts as Python."""

import json
import zipfile
from pathlib import Path
from typing import Any

from r0.modules.generation.lib.causal_pairs import load_causal_pair, save_causal_pair
from r0.modules.generation.lib.loading import load_group
from r0.modules.priors import (
    CausalInterventionRecipe,
    CausalLimits,
    TaskPair,
    compare_causal_pair,
    construct_population_graph_change,
    explain_generation,
    generate_task,
    recipe_from_dict,
)


def causal_job(request: dict[str, Any], out: Path) -> dict[str, Any]:
    kind = request["causal_action"]
    budget = request.get("max_bytes", 256 * 1024**2)
    if type(budget) is not int or budget < 65536:
        raise ValueError("causal comparison storage limit must be at least 64 KiB")
    limits = CausalLimits(**request.get("causal_limits", {}))
    out.mkdir(parents=True, exist_ok=False)
    if kind in ("create", "sample_policy", "gp_target_edit"):
        before_path = Path(request["before_archive"])
        if before_path.stat().st_size > budget:
            raise ValueError("original archive exceeds input storage limit")
        before = next(load_group(before_path))
        record = json.loads(before.recipe_json)
        if before.episode_id != request["before_episode_id"]:
            raise ValueError("original dataset identity changed")
        if kind == "gp_target_edit":
            from r0.modules.generation.lib.gp_target_edit import gp_target_edit

            pair = gp_target_edit(before, request, limits=limits)
        else:
            if kind == "sample_policy":
                from r0.modules.generation.lib.intervention_policy_construction import (
                    construct_intervention_policy,
                )

                recipe = construct_intervention_policy(
                    record, request["policy_json"], limits=limits
                )
            elif record["recipe"]["version"] == "recursive-population-graph-v1":
                recipe = construct_population_graph_change(
                    record["recipe"],
                    json.loads(request["interventions_json"]),
                    mode=request["mode"],
                    population=request["population"],
                    limits=limits,
                )
            else:
                changes = json.loads(request["interventions_json"])
                recipe = recipe_from_dict(
                    {
                        "version": "causal-intervention-v1",
                        "base": record["recipe"],
                        "interventions": changes,
                        "mode": request["mode"],
                        "population": request["population"],
                    },
                    limits=limits,
                )
                if not isinstance(recipe, CausalInterventionRecipe):
                    raise ValueError("causal comparison requires an intervention recipe")
            after = generate_task(recipe, seed=record["seed"], task_id=record["task_id"])
            card = compare_causal_pair(before, after, limits=limits)
            pair = TaskPair(before, after, json.dumps(card, sort_keys=True))
        card = json.loads(pair.card_json)
        path = out / "pair"
        # One total budget reserves both group files and their completion records.
        save_causal_pair(path, pair, max_bytes=budget // 2, limits=limits)
    else:
        path = Path(request["pair_path"])
        if (
            sum(
                (path / name).stat().st_size
                for name in ("before.zip", "before.json", "after.zip", "after.json", "pair.json")
            )
            > budget
        ):
            raise ValueError("saved causal pair exceeds input storage limit")
        pair = load_causal_pair(path, limits=limits)
        card = json.loads(pair.card_json)
    if sum(p.stat().st_size for p in path.iterdir() if p.is_file()) > budget:
        raise ValueError("causal pair exceeds total storage limit")
    summary: dict[str, Any] = {"status": "COMPLETE", "pair_path": str(path), "card": card}
    if kind == "inspect":
        summary["before"] = explain_generation(pair.before, request["indices"], limits=limits)
        summary["after"] = explain_generation(pair.after, request["indices"], limits=limits)
    elif kind == "export":
        bundle = out / "causal-pair.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in ("before.zip", "before.json", "after.zip", "after.json", "pair.json"):
                archive.write(path / name, name)
            archive.writestr(
                "README.txt",
                "Extract this directory, then use "
                "r0.modules.generation.load_causal_pair(Path(directory)). Both ZIP groups "
                "include original recipe, predictor arrays, truth arrays and completion "
                "receipts. pair.json binds their identities and comparison card.\n",
            )
        if bundle.stat().st_size > budget:
            raise ValueError("causal comparison download exceeds total storage limit")
        summary["download"] = str(bundle)
    elif kind not in ("create", "sample_policy", "gp_target_edit", "open"):
        raise ValueError("unsupported causal comparison action")
    if len(json.dumps(summary, allow_nan=False).encode()) > budget:
        raise ValueError("causal comparison response exceeds output limit")
    return summary
