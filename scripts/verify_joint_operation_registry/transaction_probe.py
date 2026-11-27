"""Independent raw-draw oracle, transaction failure matrix, and public real-fit abort."""

import copy
import itertools
import pickle

import numpy as np
from common import rejected, require, save, start

from r0.modules import priors

ENGINES = (
    np.random.PCG64,
    np.random.PCG64DXSM,
    np.random.MT19937,
    np.random.Philox,
    np.random.SFC64,
)


def states(generators):
    return pickle.dumps([generator.bit_generator.state for generator in generators])


def make(engine, mode):
    generators = [np.random.Generator(engine(seed)) for seed in (1901, 1902, 1903, 1904)]
    context = priors.JointSamplingContext(
        generators[0], mode=mode, streams=dict(zip(("a", "b", "c"), generators[1:], strict=True))
    )
    return context, generators


def draw(scope, index):
    scope.numerical(f"n-{index}", priors.NumericalParameter("shared", -2.0, 3.0))
    scope.categorical(f"c-{index}", priors.CategoricalParameter("category", ("x", "y")))
    return (
        scope.stream("a").normal(size=7),
        scope.stream("b").integers(0, 1000, size=11),
        scope.stream("c").uniform(-1, 2, size=5),
    )


def body(graph, exercise_failure):
    first = draw(graph, "first")
    parent = graph.stream("a")
    if exercise_failure:
        before = graph.freeze()
        before_states = states([graph.stream(name) for name in ("a", "b", "c")])
        try:
            with graph.transaction("child") as child:
                draw(child, "discarded")
                raise ArithmeticError("controlled abort")
        except ArithmeticError:
            pass
        require(graph.freeze() == before, "caught abort altered ledger")
        require(
            states([graph.stream(name) for name in ("a", "b", "c")]) == before_states,
            "caught abort altered raw streams",
        )
    with graph.transaction("child") as child:
        rejected(lambda: graph.stream("a"))
        rejected(lambda: graph.freeze_registry())
        second = draw(child, "second")
        stale = child.stream("a")
    rejected(lambda: child.stream("a"))
    before = states([parent])
    stale.random(100)
    require(states([parent]) == before, "stale child aliases parent")
    with graph.transaction("empty"):
        pass
    return first, second, parent


def main():
    output = start()
    cases = []
    for engine in ENGINES:
        for mode in ("local", "global", "meta"):
            context, generators = make(engine, mode)
            frozen, original = context.freeze_registry(), states(generators)
            try:
                with context.transaction("graph") as graph:
                    body(graph, True)
                    raise ArithmeticError("outer abort")
            except ArithmeticError:
                pass
            require(
                context.freeze_registry() == frozen and states(generators) == original,
                "outer abort committed data",
            )
            with context.transaction("graph") as graph:
                first, second, stale = body(graph, True)
            current = states(generators)
            stale.random(100)
            require(states(generators) == current, "stale outer aliases caller")
            reference, refs = make(engine, mode)
            with reference.transaction("graph") as graph:
                body(graph, False)
            require(
                context.freeze_registry() == reference.freeze_registry(), "rollback cache drift"
            )
            require(states(generators) == states(refs), "rollback RNG drift")
            independent = [np.random.Generator(engine(seed)) for seed in (1902, 1903, 1904)]
            for actual in (first, second):
                expected = (
                    independent[0].normal(size=7),
                    independent[1].integers(0, 1000, size=11),
                    independent[2].uniform(-1, 2, size=5),
                )
                for left, right in zip(actual, expected, strict=True):
                    require(np.array_equal(left, right), "raw draw oracle mismatch")
            require(states(generators[1:]) == states(independent), "raw state oracle mismatch")
            spans = [
                (item.operation, item.event_start, item.event_stop)
                for item in context.freeze_registry().operations
            ]
            require(
                spans
                == [(("graph",), 0, 4), (("graph", "child"), 2, 4), (("graph", "empty"), 4, 4)],
                "literal span mismatch",
            )
            rejected(lambda context=context: context.transaction("graph").__enter__())
            cases.append(f"{engine.__name__}/{mode}: recovery, cache, raw, stale, literal spans")
        for changed in range(4):
            context, generators = make(engine, "meta")
            before = context.freeze_registry()
            controls = copy.deepcopy(generators)
            try:
                with context.transaction("graph") as graph:
                    draw(graph, "discarded")
                    generators[changed].normal(size=3)
            except ValueError as error:
                require("caller RNG changed" in str(error), "wrong external-mutation rejection")
            else:
                raise RuntimeError("external mutation committed")
            controls[changed].normal(size=3)
            require(states(generators) == states(controls), "external draws overwritten")
            require(context.freeze_registry() == before, "external mutation committed ledger")
            cases.append(f"{engine.__name__}/external-{changed}")
    for left, right in itertools.combinations(range(4), 2):
        generators = [np.random.default_rng(seed) for seed in range(4)]
        generators[right] = np.random.Generator(generators[left].bit_generator)
        before = states(generators)
        rejected(
            lambda generators=generators: priors.JointSamplingContext(
                generators[0], streams=dict(zip(("a", "b", "c"), generators[1:], strict=True))
            )
        )
        require(states(generators) == before, "alias rejection consumed RNG")
        cases.append(f"alias-{left}-{right}")
    context, generators = make(np.random.PCG64, "local")
    initial, frozen = states(generators), context.freeze_registry()
    try:
        with context.transaction("graph") as graph:
            plan = priors.sample_joint_root(
                graph,
                "real",
                width=2,
                calibration_rows=4,
                prior=priors.JointRootPrior(
                    families=("covariance",),
                    function=priors.JointFunctionPrior(
                        functions="tree",
                        matrix=priors.JointMatrixPrior(
                            families=("gaussian",), vector_family="uniform"
                        ),
                    ),
                ),
            )
            realization = priors.materialize_joint_root(
                plan,
                graph.freeze(),
                rng=graph.stream("a"),
                calibration_rng=graph.stream("b"),
                fit_rng=graph.stream("c"),
                calibration_id="oracle",
            )
            require(len(realization.outputs) == 4, "real fit absent")
            raise ArithmeticError("abort after real fit")
    except ArithmeticError:
        pass
    require(
        context.freeze_registry() == frozen and states(generators) == initial,
        "real-fit abort leaked state",
    )
    cases.append("real covariance/tree fit then abort")
    save(output, "result", {"verdict": "PASS", "cases": cases})
    print(f"PASS: {len(cases)} independent transaction scenarios")


if __name__ == "__main__":
    main()
