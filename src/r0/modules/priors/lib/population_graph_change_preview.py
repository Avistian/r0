"""Array-free effective equation preview for population intervention admission only."""

from typing import Any, cast

from r0.modules.priors.lib.fitted_tree_decoding import fitted_array


def population_change_preview(base: dict[str, Any], changes: Any) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for value in fitted_array(base["nodes"], len(base["nodes"])):
        if type(value) is not dict:
            raise ValueError("population node preview requires an object")
        node = dict(cast(dict[str, Any], value))
        if "equation" in node:
            node["equation"] = dict(node["equation"])
        nodes.append(node)
    for value in fitted_array(changes, 4 * len(nodes)):
        if type(value) is not dict:
            raise ValueError("population change preview requires an object")
        change = cast(dict[str, Any], value)
        index = change.get("node")
        if type(index) is not int or not 0 <= index < len(nodes):
            raise ValueError("population change node is outside the graph")
        node = nodes[index]
        equation = node.get("equation", node)
        version = change.get("version")
        if version == "soft-causal-intervention-v1":
            for key in ("function", "noise_scale", "noise_response", "exogenous"):
                if change.get(key) is not None:
                    equation[key] = change[key]
        elif version in ("stage-causal-intervention-v1", "stage-causal-intervention-v2"):
            if change.get("root") is not None:
                equation["root"] = change["root"]
            if change.get("replace_pipeline"):
                equation["pipeline"] = change.get("pipeline")
            if change.get("domain") is not None:
                node["domain"] = change["domain"]
        elif version == "native-vector-domain-intervention-v1":
            domains = list(fitted_array(node["domains"], len(node["domains"])))
            variable = change.get("variable")
            if type(variable) is not int or not 0 <= variable < len(domains):
                raise ValueError("population changed variable is outside the node")
            domains[variable] = change["domain"]
            node["domains"] = domains
        elif version == "converter-soft-intervention-v1":
            node["converters"] = [
                change["converter"] if c["name"] == change["capture"] else c
                for c in node["converters"]
            ]
        elif version not in (
            "hard-causal-intervention-v1",
            "native-vector-hard-intervention-v1",
            "converter-hard-intervention-v1",
        ):
            raise ValueError("population change requires a supported explicit version")
    return {**base, "nodes": nodes}
