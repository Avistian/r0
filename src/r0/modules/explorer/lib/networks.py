"""Broad standalone architecture presets and an authored frozen network editor."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from r0.modules import priors

NOTICE = (
    "Standalone random network: input features feed a frozen architecture, not a causal "
    "graph or parent reducer. Structural masks stay fixed across rows. Zero or disconnected "
    "draws remain valid. Editing creates a linked authored network; original sampling "
    "decisions and raw draws remain in the original archive. It is not an intervention."
)


def network_examples() -> dict[str, Any]:
    task = priors.LinearRecipe(
        rows=64, context_rows=48, features=2, signal_features=2, active_features=2
    )
    examples: dict[str, Any] = {}
    for family in (
        "dense",
        "individual-sparse",
        "block-sparse",
        "skip",
        "residual",
        "irregular",
        "gated",
    ):
        request = priors.NetworkSamplingRequest(2, (6, 4, 1), families=(family,))
        raw = asdict(request)
        raw.pop("limits")
        examples[f"Standalone network: {family}"] = {
            "topology": "network",
            "task": asdict(task),
            "network_prior": raw,
            "root": None,
        }
    for profile, family in (("p1-individual", "individual-sparse"), ("p1-block", "block-sparse")):
        request = priors.NetworkSamplingRequest(
            2,
            (6, 4, 1),
            families=(family,),
            fan_in=False,
            mask_profile=profile,
            dropout_profile="p1-beta" if profile == "p1-individual" else "fixed",
        )
        raw = asdict(request)
        raw.pop("limits")
        examples[f"Standalone network: {profile} source laws"] = {
            "topology": "network",
            "task": asdict(task),
            "network_prior": raw,
            "root": None,
        }
    left = priors.NetworkAffine(
        ((1.0, -0.5), (-0.3, 0.8)),
        (0.1, -0.2),
        mask=((True, False), (True, True)),
        activation=priors.CausalActivation("tanh"),
    )
    right = priors.NetworkAffine(
        ((-0.7, 0.2), (0.4, -1.0)), (0.0, 0.3), activation=priors.CausalActivation("relu")
    )
    gate = priors.NetworkAffine(((0.8, 0.0), (0.0, -0.9)), (0.0, 0.0))
    skip = priors.NetworkProjection(0, priors.NetworkAffine(((0.2, 0.0), (0.0, 0.2)), (0.0, 0.0)))
    output = priors.NetworkAffine(((0.6, -0.8, 1.2, 0.4),), (0.15,))
    network = priors.FrozenNetworkFunction(
        2,
        (
            priors.NetworkBlock((0,), (left, right), gate, (skip,)),
            priors.NetworkBlock((0, 1), (output,)),
        ),
    )
    examples["Standalone network: sparse + gated + skip (authored)"] = {
        "topology": "network-authored",
        "task": asdict(task),
        "network": asdict(network),
        "root": None,
    }
    return examples


def network_editor(recipe: priors.NetworkRecipe | priors.NetworkInputView) -> dict[str, Any]:
    base = recipe.base if isinstance(recipe, priors.NetworkInputView) else recipe
    authored = replace(base, network=base.function)
    editable = (
        priors.NetworkInputView(authored, recipe.task, recipe.active_inputs)
        if isinstance(recipe, priors.NetworkInputView)
        else authored
    )
    return network_editor_record(
        asdict(editable), recipe.version, isinstance(recipe, priors.MatchedNetworkInputView)
    )


def network_editor_record(editable: dict[str, Any], version: str, matched: bool) -> dict[str, Any]:
    base = editable.get("base", editable)
    return {
        "kind": "mechanism-edit",
        "network": True,
        "matching_edit": matched,
        "construction": version,
        "editable_recipe": editable,
        "notice": NOTICE,
        "nodes": [
            {
                "index": i,
                "parents": b["parents"],
                "width": len(b["experts"][0]["bias"]),
                "experts": len(b["experts"]),
                "gated": b["gate"] is not None,
                "skip_sources": [s["source"] for s in b["skips"]],
            }
            for i, b in enumerate(base["network"]["blocks"])
        ],
    }


def guarded_network_editor(
    task: priors.TrainingTask,
    data: dict[str, Any],
    output_dir: Path,
    *,
    limits: priors.CausalLimits,
) -> dict[str, Any] | None:
    from r0.modules.generation import run_guarded_document

    raw = json.loads(task.recipe_json)["recipe"]
    if raw.get("version") not in (
        "standalone-network-v1",
        "network-input-view-v1",
        "network-input-view-v2",
    ):
        return None
    result = run_guarded_document(
        {
            "validation": "network-editor",
            "recipe": raw,
            "causal_limits": asdict(limits),
            "max_seconds": data.get("max_seconds", 60),
            "max_bytes": data.get("max_bytes", 256 * 1024**2),
        },
        output_dir / "network-editor",
    )
    if result.timed_out:
        raise ValueError("Network editor preparation timed out; dataset unchanged")
    if result.returncode or result.summary is None:
        raise ValueError("Network editor preparation failed: " + result.stderr[-2000:])
    return network_editor_record(
        cast(dict[str, Any], result.summary["recipe"]),
        raw["version"],
        raw["version"] == "network-input-view-v2",
    )


def network_details(recipe: dict[str, Any]) -> dict[str, Any]:
    view = (
        recipe
        if recipe.get("version") in ("network-input-view-v1", "network-input-view-v2")
        else None
    )
    network = (view["base"] if view else recipe)["network"]
    profiled = network.get("version") == "sampled-hyperprior-network-v1"
    sampled = network.get("version") == "sampled-structural-network-v1" or profiled
    function = network["function"] if sampled else network
    notice = NOTICE
    if profiled:
        profile = network["profile"]
        effective = network["request"]
        notice += (
            f" Scalar laws: depth {profile['depth']['family']}, "
            f"width {profile['width']['family']}, weight {profile['weight_scale']['family']}. "
            f"Realized block widths {effective['widths']}; weight scale "
            f"{effective['weight_scale']:.6g}; width scope {profile['width_scope']}. "
            "Scalar source laws do not make the surrounding architecture a full paper reproduction."
        )
    from r0.modules.explorer.lib.source_root_details import source_root_details

    return {
        "kind": "standalone-network",
        "source_root": source_root_details((view["base"] if view else recipe).get("root")),
        "input_view": None
        if view is None
        else {
            "active_inputs": view["active_inputs"],
            "observed_features": view["task"]["features"],
            "fill": 0.0,
            "position": "after original joint root",
            "variance": "independent baseline matching estimate"
            if view.get("matching")
            else "uncontrolled; no normalization",
            "matching": None
            if not view.get("matching")
            else {
                key: view["matching"][key]
                for key in (
                    "rows",
                    "before_inputs",
                    "after_inputs",
                    "before_mean",
                    "before_std",
                    "after_mean",
                    "after_std",
                    "version",
                )
            },
        },
        "notice": notice,
        "sampled": sampled,
        "request": network.get("request"),
        "choices": network.get("choices", []),
        "raw_draw_count": len(network.get("raw", [])) + len(network.get("hyper_raw", [])),
        "stream_roles": ["hyper", "named", "raw"]
        if profiled
        else ["named", "raw"]
        if sampled
        else [],
        "hyperprior": None
        if not profiled
        else {
            "profile": network["profile"],
            "parameters": network["parameters"],
            "raw_draws": network["hyper_raw"],
            "initial": network["hyper_initial"],
            "final": network["hyper_final"],
            "effective_widths": network["request"]["widths"],
            "effective_weight_scale": network["request"]["weight_scale"],
            "scope": (
                "Source scalar laws on an r0 architecture; capacity rejections "
                "are not retries or source truncation"
            ),
        },
        "blocks": len(function["blocks"]),
        "input_width": function["input_width"],
        "output_width": len(function["blocks"][-1]["experts"][0]["bias"]),
        "source_semantics": (
            "Named P1 mask/dropout profiles are source-law adaptations. Residual, skip, irregular "
            "and gated architecture combinations are r0 extensions; no trained labels."
        ),
    }


def validate_editor_draft(
    data: dict[str, Any], output_dir: Path, *, limits: priors.CausalLimits, source: str
) -> dict[str, Any]:
    from r0.modules.generation import run_guarded_document

    original = json.loads(source)["recipe"]
    matched = original.get("version") == "network-input-view-v2"
    result = run_guarded_document(
        {
            "validation": "network-draft",
            "recipe": data["recipe"],
            "matching_source": original if matched else None,
            "matching_policy": data.get("matching_policy", "recalibrate" if matched else None),
            "causal_limits": asdict(limits),
            "max_seconds": data.get("max_seconds", 60),
            "max_bytes": data.get("max_bytes", 256 * 1024**2),
        },
        output_dir / "draft-validation",
    )
    if result.timed_out:
        raise ValueError("Network draft validation timed out; accepted draft unchanged")
    if result.returncode or result.summary is None:
        detail = result.stderr.strip().splitlines()
        raise ValueError(
            "Invalid network draft: " + (detail[-1] if detail else "validation failed")
        )
    return {"recipe": result.summary["recipe"]}
