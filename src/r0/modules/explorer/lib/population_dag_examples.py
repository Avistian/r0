"""Explicit row-owned forest mechanisms mixed with frozen causal equations."""

from dataclasses import asdict
from typing import Any

from r0.modules import priors as p


def population_dag_examples() -> dict[str, Any]:
    examples: dict[str, Any] = {}
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    identity = p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0)))
    conv = p.FrozenFeatureConv1D(
        2, 1, (p.FeatureConv1DLayer(p.Conv1DGeometry(1, 1), (((2.0,),),), (1.0,)),)
    )
    network = p.FrozenNetworkFunction(
        2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, 0.0), (0.0, -1.0)), (1.0, 0.5)),)),)
    )
    for family in ("functional", "converted", "native", "vector"):
        for kind, function in (("linear", identity), ("Conv", conv), ("network", network)):
            nodes = (
                p.FunctionalCausalNode((), function, p.JointRootBase("gaussian", 2)),
                p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 1.0),))),
                p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((3.0,),))),
            )
            if family == "functional":
                base = p.FunctionalCausalRecipe(task, nodes, ((0, 0), (0, 1)), ((2, 0),))
            elif family == "native":
                base = p.NativeCausalRecipe(
                    task, tuple(p.NativeCausalNode(n) for n in nodes), ((0, 0), (0, 1)), (2, 0)
                )
            elif family == "vector":
                base = p.NativeVectorRecipe(
                    task,
                    tuple(
                        p.NativeVectorNode(n, (p.VariableDomain(),) * n.function.output_width)
                        for n in nodes
                    ),
                    ((0, 0), (0, 1)),
                    (2, 0),
                )
            else:
                converter = p.CategoricalConverter(
                    "sign", 0, 1, ("low", "high"), "neighbor_disc", ((-1.0,), (1.0,))
                )
                base = p.ConvertedCausalRecipe(
                    task,
                    tuple(
                        p.ConvertedCausalNode(n, (converter,) if i == 0 else ())
                        for i, n in enumerate(nodes)
                    ),
                    (p.ConvertedReadout(0, capture="sign"), p.ConvertedReadout(0, channel=1)),
                    p.ConvertedReadout(2, channel=0),
                )
            forest = p.RecursiveForestPlan(
                estimators=1,
                min_depth=1,
                max_depth=2,
                lifecycle="fresh-forest-per-row",
                populations=16,
            )
            request = p.RecursivePopulationGraphRequest(
                base, (p.RecursiveNodePopulationPlan(1, forest),)
            )
            examples[f"Population DAG: {family} {kind} → row forest → linear"] = {
                "topology": "recursive-population-dag",
                "request": asdict(request),
                "engine": "PCG64",
            }
    examples.update(_typed_examples())
    examples.update(_mixed_vector_examples())
    return examples


def _typed_examples() -> dict[str, Any]:
    from dataclasses import replace

    examples: dict[str, Any] = {}
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=2,
        signal_features=2,
        active_features=2,
        task_type="multiclass",
        classes=3,
    )
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    child = p.FunctionalCausalNode(
        (0,), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0), (-1.0, 1.0)))
    )
    for family in ("functional", "native", "vector", "converted"):
        domain = p.VariableDomain("categorical", ("a", "b", "c"), "softmax")
        if family == "functional":
            base = p.FunctionalCausalRecipe(
                task, (root, child), ((0, 0), (0, 1)), ((1, 0), (1, 1), (1, 2))
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
            converter = p.CategoricalConverter(
                "label",
                0,
                3,
                domain.vocabulary,
                "softmax_id",
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                mean=(0.0,) * 3,
                scale=(1.0,) * 3,
                bias=(0.0,) * 3,
            )
            base = p.ConvertedCausalRecipe(
                task,
                (p.ConvertedCausalNode(root), p.ConvertedCausalNode(child, (converter,))),
                (p.ConvertedReadout(0, channel=0), p.ConvertedReadout(0, channel=1)),
                p.ConvertedReadout(1, capture="label"),
            )
        forest = p.RecursiveForestPlan(
            estimators=1,
            min_depth=1,
            max_depth=1,
            source="r0-extension-v1",
            leaf_mean=(0.0,) * 3,
            leaf_scale=(1.0,) * 3,
            lifecycle="fresh-forest-per-row",
            populations=16,
        )
        request = p.RecursivePopulationGraphRequest(
            base, (p.RecursiveNodePopulationPlan(1, forest),)
        )
        examples[f"Population DAG: {family} vector scores and stochastic categories"] = {
            "topology": "recursive-population-dag",
            "request": asdict(request),
            "engine": "PCG64",
        }
        if family in ("native", "vector"):
            categorical = replace(domain, emission="argmax")
            labeled = replace(
                base,
                nodes=(
                    base.nodes[0],
                    p.NativeCausalNode(child, categorical)
                    if family == "native"
                    else p.NativeVectorNode(child, (categorical,)),
                ),
            )
            labels = p.RecursiveForestPlan(
                estimators=2,
                min_depth=1,
                max_depth=1,
                task="classification",
                classes=3,
                class_assignment="random-start-cycle",
                tie_break="smallest",
                leaf_mean=(),
                leaf_scale=(),
                lifecycle="fresh-forest-per-row",
                populations=16,
            )
            request = p.RecursivePopulationGraphRequest(
                labeled, (p.RecursiveNodePopulationPlan(1, labels),)
            )
            examples[f"Population DAG: {family} majority-vote labels"] = {
                "topology": "recursive-population-dag",
                "request": asdict(request),
                "engine": "PCG64",
            }
    return examples


def _mixed_vector_examples() -> dict[str, Any]:
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))),
        p.JointRootBase("gaussian", 2),
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 2.0, 3.0),)))
    domains = (p.VariableDomain("binary", ("low", "high"), "softmax"), p.VariableDomain())
    base = p.NativeVectorRecipe(
        task,
        (p.NativeVectorNode(root, domains), p.NativeVectorNode(child, (p.VariableDomain(),))),
        ((0, 0), (0, 1)),
        (1, 0),
    )
    forest = p.RecursiveForestPlan(
        estimators=1,
        min_depth=1,
        max_depth=1,
        source="r0-extension-v1",
        leaf_mean=(0.0, 0.0, 1.0),
        leaf_scale=(1.0, 1.0, 0.5),
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    downstream = p.RecursiveForestPlan(
        input_width=3,
        intervals=((-3.0, 3.0),) * 3,
        estimators=1,
        min_depth=1,
        max_depth=1,
        lifecycle="fresh-forest-per-row",
        populations=16,
    )
    entries: dict[str, Any] = {}
    for multiple in (False, True):
        plans = (p.RecursiveNodePopulationPlan(0, forest),)
        if multiple:
            plans += (p.RecursiveNodePopulationPlan(1, downstream),)
        request = p.RecursivePopulationGraphRequest(base, plans)
        entries[
            "Population DAG: mixed vector root"
            + (" and descendant populations" if multiple else " with numeric descendant")
        ] = {"topology": "recursive-population-dag", "request": asdict(request), "engine": "PCG64"}
    return entries
