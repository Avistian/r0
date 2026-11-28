"""Finite GP root inputs with explicit graph mechanisms and typed target choices."""

from dataclasses import asdict, replace
from typing import Any

from r0.modules import priors as p


def gp_root_examples() -> dict[str, Any]:
    result: dict[str, Any] = {}
    identity = p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0)))
    convolution = p.FrozenFeatureConv1D(
        2, 1, (p.FeatureConv1DLayer(p.Conv1DGeometry(1, 1), (((2.0,),),), (1.0,)),)
    )
    network = p.FrozenNetworkFunction(
        2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, 0.0), (0.0, -1.0)), (1.0, 0.5)),)),)
    )
    for family in ("functional", "native", "vector", "converted"):
        for name, function in (("linear", identity), ("Conv", convolution), ("network", network)):
            for categorical in (False, True):
                task = p.LinearRecipe(
                    rows=16,
                    context_rows=8,
                    features=2,
                    signal_features=2,
                    active_features=2,
                    task_type="multiclass" if categorical else "regression",
                    classes=3 if categorical else 0,
                )
                root = p.FunctionalCausalNode((), function, p.JointRootBase("gaussian", 2))
                child = p.FunctionalCausalNode(
                    (0,),
                    p.FrozenLinearFunction(
                        ((1.0, 0.0), (0.0, 1.0), (-1.0, 1.0)) if categorical else ((2.0, -1.0),)
                    ),
                )
                domain = (
                    p.VariableDomain("categorical", ("low", "middle", "high"), "softmax")
                    if categorical
                    else p.VariableDomain()
                )
                if family == "functional":
                    base = p.FunctionalCausalRecipe(
                        task,
                        (root, child),
                        ((0, 0), (0, 1)),
                        tuple((1, i) for i in range(3)) if categorical else ((1, 0),),
                    )
                elif family == "native":
                    base = p.NativeCausalRecipe(
                        task,
                        (p.NativeCausalNode(root), p.NativeCausalNode(child, domain)),
                        ((0, 0), (0, 1)),
                        (1, 0),
                    )
                elif family == "vector":
                    base = p.NativeVectorRecipe(
                        task,
                        (
                            p.NativeVectorNode(root, (p.VariableDomain(),) * 2),
                            p.NativeVectorNode(child, (domain,)),
                        ),
                        ((0, 0), (0, 1)),
                        (1, 0),
                    )
                else:
                    converters = (
                        (
                            p.CategoricalConverter(
                                "label",
                                0,
                                3,
                                domain.vocabulary,
                                "softmax_id",
                                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                                mean=(0.0,) * 3,
                                scale=(1.0,) * 3,
                                bias=(0.0,) * 3,
                            ),
                        )
                        if categorical
                        else ()
                    )
                    base = p.ConvertedCausalRecipe(
                        task,
                        (p.ConvertedCausalNode(root), p.ConvertedCausalNode(child, converters)),
                        (p.ConvertedReadout(0, channel=0), p.ConvertedReadout(0, channel=1)),
                        p.ConvertedReadout(1, capture="label")
                        if categorical
                        else p.ConvertedReadout(1, channel=0),
                    )
                request = p.GPColumnRequest(
                    rows=16,
                    columns=2,
                    kernels=(p.GPColumnKernel(0.1, 1.0), p.GPColumnKernel(0.3, 0.75)),
                )
                controls = {
                    "topology": "gp-root-dag",
                    "request": asdict(request),
                    "base": asdict(base),
                    "bindings": [asdict(p.GPRootBinding(0, (0, 1)))],
                }
                target_kind = "multiclass" if categorical else "regression"
                result[f"GP root DAG: {family} {name} / {target_kind}"] = controls
                if family == "functional" and not categorical:
                    binary = replace(task, task_type="binary", classes=2)
                    result[f"GP root DAG: {family} {name} / binary"] = {
                        **controls,
                        "base": {**asdict(base), "task": asdict(binary)},
                    }
    return result
