"""Small four-runtime GP-root graph fixture shared by its public API tests."""

from r0.modules import priors as p


def gp_root_recipe(family: str):
    population = p.realize_gp_columns(p.GPColumnRequest(rows=16), seed=47, task_id=59)
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, -1.0),)))
    if family == "functional":
        base = p.FunctionalCausalRecipe(task, (root, child), ((0, 0), (0, 1)), ((1, 0),))
    elif family == "native":
        base = p.NativeCausalRecipe(
            task, (p.NativeCausalNode(root), p.NativeCausalNode(child)), ((0, 0), (0, 1)), (1, 0)
        )
    elif family == "vector":
        base = p.NativeVectorRecipe(
            task,
            (
                p.NativeVectorNode(root, (p.VariableDomain(), p.VariableDomain())),
                p.NativeVectorNode(child, (p.VariableDomain(),)),
            ),
            ((0, 0), (0, 1)),
            (1, 0),
        )
    else:
        base = p.ConvertedCausalRecipe(
            task,
            (p.ConvertedCausalNode(root), p.ConvertedCausalNode(child)),
            (p.ConvertedReadout(0, channel=0), p.ConvertedReadout(0, channel=1)),
            p.ConvertedReadout(1, channel=0),
        )
    return p.GPRootDAGRecipe(population, base, (p.GPRootBinding(0, (0, 1)),))
