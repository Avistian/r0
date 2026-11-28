"""Independent symbolic moment and literal pipeline-ordering probes."""

import math
from dataclasses import replace

import numpy as np

from r0.modules import priors as p
from r0.modules.priors.lib.functional_analysis import functional_certificates

L = p.FrozenLinearFunction
C = p.LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8)
Q = p.FrozenQuadraticFunction(1, (0,), (((1.0, 0.0), (0.0, 0.0)),))


def frozen(kind="identity", **kw):
    return p.CalibratedActivation(
        kind=kind,
        input_mean=(0.0,),
        input_inverse_std=(1.0,),
        shift=(0.0,),
        output_mean=(0.0,),
        output_inverse_std=(1.0,),
        shift_rows=(0,),
        calibration_rows=1,
        calibration_id="literal",
        calibration_digest="0" * 64,
        **kw,
    )


def root(df=3.0, factor=()):
    return p.FunctionalCausalNode(
        (),
        L(((1.0,),)),
        root=p.JointRootBase("uniform", 1),
        noise_scale=(1.0,),
        exogenous=p.ExogenousDistribution(p.InputDistribution("student_t", df=df), factor=factor),
    )


def recipe(r, f, pipe=None):
    return p.FunctionalCausalRecipe(
        C, (r, p.FunctionalCausalNode((0,), f, pipeline=pipe)), ((0, 0),), ((1, 0),)
    )


def must_reject(label, call, text):
    try:
        call()
    except ValueError as e:
        if text not in str(e):
            raise AssertionError((label, str(e))) from e
        print("PASS reject", label, str(e), flush=True)
    else:
        raise AssertionError("REJECT product admitted " + label)


for name, f, bound in [
    ("linear", L(((2.0,),)), 3.0),
    ("quadratic", Q, 1.5),
    ("tree", p.FrozenTreeFunction(1, ((0,),), ((0.0,),), (((-1.0,), (1.0,)),)), math.inf),
    ("rff", p.FrozenRFFFunction(((1.0,),), (0.0,), ((1.0,),)), math.inf),
    ("discretization", p.FrozenDiscretizationFunction(((-1.0,), (1.0,)), L(((1.0,),))), math.inf),
    ("em", p.FrozenEMFunction(((-1.0,), (1.0,)), (1.0, 1.0), L(((-1.0, 1.0),))), math.inf),
    ("product", p.FrozenProductFunction(Q, L(((1.0,),))), 1.0),
    (
        "mlp",
        p.FrozenMLPFunction((L(((1.0,),)),), (frozen("tanh"),), start_activation=True),
        math.inf,
    ),
]:
    rec = recipe(root(), f)
    state, _ = functional_certificates(rec.nodes)
    assert state[1][0].order == bound, (name, state)
    pipe = p.CausalNodePipeline(1, stages=(frozen(),))
    if bound <= 2:
        must_reject(
            "post-" + name, lambda f=f, pipe=pipe: recipe(root(), f, pipe), "second moments"
        )
    else:
        recipe(root(), f, pipe)
    print("PASS moment", name, bound, flush=True)
must_reject(
    "MLP input t_2",
    lambda: recipe(
        root(2.0), p.FrozenMLPFunction((L(((1.0,),)),), (frozen(),), start_activation=True)
    ),
    "second moments",
)
must_reject(
    "MLP square output t_3",
    lambda: recipe(
        root(), p.FrozenMLPFunction((L(((1.0,),)),), (frozen("square"),), start_activation=True)
    ),
    "second moments",
)
# X^3 with t_6 innovations has no finite second moment; Holder bound must remain <=2.
must_reject(
    "shared cubic boundary t_6",
    lambda: recipe(
        root(6.0),
        p.FrozenProductFunction(Q, L(((1.0,),))),
        p.CausalNodePipeline(1, stages=(frozen(),)),
    ),
    "second moments",
)
for r in (
    root(1.0, ((0.0,),)),
    replace(root(1.0), noise_scale=(0.0,)),
    replace(root(1.0), noise_response=p.NoiseResponse(0.0, 0.0)),
):
    rec = recipe(r, L(((1.0,),)), p.CausalNodePipeline(1, stages=(frozen(),)))
    assert functional_certificates(rec.nodes)[0][1][0].order == math.inf
    print("PASS absent innovation path admits normalization")
# Exact zero linear removes heavy-tail dependency before an internal fitted stage.
zero = recipe(root(1.0), p.FrozenMLPFunction((L(((0.0,),)),), (frozen(),), end_activation=True))
assert not functional_certificates(zero.nodes)[0][1][0].sources
# Unknown unbounded exponential stays unknown; bounded exponential is certifiable.
recipe(
    replace(root(), noise_scale=()), L(((1.0,),)), p.CausalNodePipeline(1, stages=(frozen("exp"),))
)
must_reject(
    "unbounded exp normalization",
    lambda: recipe(root(), L(((1.0,),)), p.CausalNodePipeline(1, stages=(frozen("exp"),))),
    "second moments",
)
# Converter raw/readout/state behavior with nontrivial affine warp and structural noise.
warp = frozen(parameters=(("output_scale", 2.0), ("output_shift", 1.0)))
for variant in ("historical-tuple", "historical-graph", "readout-only"):
    r = p.FunctionalCausalNode((), L(((1.0,),)), root=p.JointRootBase("uniform", 1))
    child = p.FunctionalCausalNode(
        (0,),
        Q,
        noise_scale=(0.5,),
        noise_response=p.NoiseResponse(1.0, 2.0, 2.0),
        pipeline=p.CausalNodePipeline(
            1,
            stages=(p.CausalActivation("identity", (("output_shift", 2.0),)),),
            channel_importance=(3.0,),
            converters=(p.NumericalConverter(0, warp, variant),),
            node_importance=4.0,
        ),
    )
    rec = p.FunctionalCausalRecipe(
        C, (r, child, p.FunctionalCausalNode((1,), L(((1.0,),)))), ((1, 0),), ((2, 0),)
    )
    task = p.generate_task(rec, seed=42, task_id=17)
    trace = p.explain_generation(task, [15, 1, 15, 0])
    nodes = {n["id"]: n for n in trace["nodes"]}

    def raw(k, nodes=nodes):
        return np.array(nodes[k]["values"])

    x = raw("causal-0")
    output = x * x
    before = 3 * (output + 2)
    warped = 2 * before + 1
    expected_readout = before if variant == "historical-tuple" else warped
    expected_state = 4 * (
        (before if variant == "readout-only" else warped)
        + 0.5 * (1 + 2 * output**2) * raw("exogenous-1")
    )
    np.testing.assert_allclose(task.inputs.numpy()[[15, 1, 15, 0]], expected_readout, rtol=1e-14)
    np.testing.assert_allclose(task.scores.numpy()[[15, 1, 15, 0]], expected_state, rtol=1e-14)
    assert trace["signal_max_error"] == trace["target_max_error"] == 0
    print("PASS converter literal", variant, flush=True)
print(
    "PASS: symbolic moments, zero/factor source paths, internal MLP "
    "normalization, and three converter variants."
)
