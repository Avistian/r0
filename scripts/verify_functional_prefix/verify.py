import dataclasses
import hashlib
import json
import pathlib
import sys
from unittest.mock import patch

sys.path.insert(0, sys.argv[1])
import numpy as np
import torch

from r0.modules import priors as p

OUT = pathlib.Path(sys.argv[2])
OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)


def encode(x):
    if isinstance(x, torch.Tensor):
        return {"dtype": str(x.dtype), "shape": list(x.shape), "values": x.tolist()}
    if dataclasses.is_dataclass(x):
        return {f.name: encode(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [encode(v) for v in x]
    return x


def dump(name, x):
    (OUT / (name + ".json")).write_text(json.dumps(encode(x), sort_keys=True, indent=2) + "\n")


def activation():
    return p.CalibratedActivation(
        kind="identity",
        input_mean=(0.0,),
        input_inverse_std=(1.0,),
        shift=(1.0,),
        output_mean=(0.0,),
        output_inverse_std=(1.0,),
        shift_rows=(0,),
        calibration_rows=1,
        calibration_id="independent-plus-one",
        calibration_digest="1" * 64,
    )


def base(family):
    if family == "covariance":
        return p.JointRootBase(
            family,
            2,
            gaussian_factor=((1.0, 2.0), (3.0, -1.0)),
            weights=p.UniformWeightVector((0.5, 0.5)),
            factor=((0.5, 1.0), (1.5, -0.5)),
            covariance_base="gaussian",
        )
    return p.JointRootBase(family, 2)


def nodes(family, variant):
    return (
        p.FunctionalCausalNode(
            (),
            p.FrozenLinearFunction(((1.0, 2.0), (-2.0, 1.0))),
            root=base(family),
            noise_scale=(0.2, 0.3),
        ),
        p.FunctionalCausalNode(
            (0,),
            p.FrozenLinearFunction(((2.0, -1.0),)),
            noise_scale=(0.7,),
            noise_response=p.NoiseResponse(offset=1.0, strength=0.4),
            pipeline=p.CausalNodePipeline(
                1,
                channel_importance=(3.0,),
                node_importance=2.0,
                converters=(p.NumericalConverter(0, activation(), variant),),
            ),
        ),
        p.FunctionalCausalNode((1, 0), p.FrozenLinearFunction(((5.0, 2.0, -3.0),))),
    )


for family in ["uniform", "gaussian", "ball", "covariance"]:
    for variant in ["historical-tuple", "historical-graph", "readout-only"]:
        for version in [1, 2]:
            controls = p.LinearRecipe(
                rows=16, context_rows=8, features=1, signal_features=1, active_features=1
            )
            klass = p.FunctionalCausalRecipe if version == 1 else p.CalibratedFunctionalCausalRecipe
            kw = {} if version == 1 else {"calibration_rows": 32}
            recipe = klass(controls, nodes(family, variant), ((1, 0),), ((2, 0),), **kw)
            encoded = json.dumps(dataclasses.asdict(recipe), sort_keys=True)
            decoded = p.recipe_from_dict(json.loads(encoded))
            assert decoded == recipe
            task = p.generate_task(decoded, seed=517, task_id=21)
            trace = p.explain_generation(task, [15, 3, 15, 0])
            assert trace["signal_max_error"] == trace["target_max_error"] == 0
            dump(
                f"{family}-{variant}-v{version}",
                {
                    "serialized_recipe": encoded,
                    "task": task,
                    "trace": trace,
                    "admission": recipe.admit(),
                    "extra_rows_admission": recipe.admit(evaluation_rows=77),
                },
            )
print("PASS 24 complete graph snapshots: JSON recipe/card/trace/tensors/admission")
if not hasattr(p, "FunctionalPrefix"):
    sys.exit(0)
log = []


def record(name, data):
    log.append({"name": name, "result": encode(data)})
    print("PASS", name)


def keyed(pop):
    def provider(*keys):
        return np.random.default_rng(
            int.from_bytes(hashlib.sha256(json.dumps([pop, keys]).encode()).digest()[:8], "little")
        )

    return provider


# Literal tree outputs (4,-2), reordered unequal parent widths: (3,4,-2) dot (100,10,1)=338.
r = p.FunctionalCausalNode(
    (),
    p.FrozenTreeFunction(1, ((0,),), ((0.0,),), (((4.0, -2.0), (4.0, -2.0)),)),
    root=p.JointRootBase("uniform", 1),
)
c = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 0.5),)))
d = p.FunctionalCausalNode((1, 0), p.FrozenLinearFunction(((100.0, 10.0, 1.0),)))
pr = p.FunctionalPrefix((r, c, d), rows=5)
result = p.evaluate_functional_prefix(pr, rng=keyed(1))
assert result.states[2].tolist() == [[338.0]] * 5
record("ordered parent states and changed dimensions", result)
# Independent NumPy equation distinguishes structural noise from converter readouts.
for variant in ["historical-tuple", "historical-graph", "readout-only"]:
    rr = dataclasses.replace(
        r,
        function=p.FrozenTreeFunction(1, ((0,),), ((0.0,),), (((2.0,), (2.0,)),)),
        noise_scale=(1.0,),
        pipeline=p.CausalNodePipeline(
            1,
            channel_importance=(3.0,),
            node_importance=2.0,
            converters=(p.NumericalConverter(0, activation(), variant),),
        ),
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((4.0,),)))
    out = p.evaluate_functional_prefix(
        p.FunctionalPrefix((rr, child), rows=5), rng=lambda *keys: np.random.default_rng(71)
    )
    raw = np.random.default_rng(71).standard_normal(5)
    expected = 2 * ((6.0 if variant == "readout-only" else 7.0) + raw)
    np.testing.assert_array_equal(out.states[0].numpy()[:, 0], expected)
    np.testing.assert_array_equal(out.states[1].numpy()[:, 0], 4 * expected)
    assert out.readouts[0].tolist() == [[6.0 if variant == "historical-tuple" else 7.0]] * 5
    record("converter " + variant, out)
# Caller owns reproducible distinct population namespaces.
rr = p.FunctionalCausalNode(
    (), p.FrozenLinearFunction(((1.0,),)), root=p.JointRootBase("gaussian", 1)
)
pr = p.FunctionalPrefix((rr,), rows=5)
a = p.evaluate_functional_prefix(pr, rng=keyed(1))
b = p.evaluate_functional_prefix(pr, rng=keyed(1))
c = p.evaluate_functional_prefix(pr, rng=keyed(2))
assert torch.equal(a.states[0], b.states[0])
assert not torch.equal(a.states[0], c.states[0])
record("caller population namespaces", [a, b, c])


def rejects(name, make, fragment):
    try:
        make()
    except ValueError as e:
        if fragment not in str(e):
            raise AssertionError((name, str(e))) from e
        record(name, str(e))
        return
    raise AssertionError(name + " accepted")


# Every selected allocation entry point raises; invalid construction must produce only ValueError.
def blocked(*args, **kwargs):
    raise AssertionError("numeric allocation before rejection")


with (
    patch.object(np, "array", blocked),
    patch.object(np, "asarray", blocked),
    patch.object(np, "empty", blocked),
    patch.object(np, "zeros", blocked),
    patch.object(torch, "tensor", blocked),
    patch.object(torch, "empty", blocked),
):
    for rows in [True, 0, -1, 2**63]:
        rejects(
            "invalid rows " + str(rows),
            lambda rows=rows: p.FunctionalPrefix((rr,), rows=rows),
            "rows",
        )
    rejects(
        "forward parent",
        lambda: p.FunctionalPrefix(
            (rr, p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((1.0,),)))), rows=5
        ),
        "parent-before-child",
    )
    rejects(
        "input width mismatch",
        lambda: p.FunctionalPrefix(
            (rr, p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0, 2.0),)))), rows=5
        ),
        "parent state widths",
    )
    rejects(
        "node count cap",
        lambda: p.FunctionalPrefix(
            (rr, p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),)))),
            rows=5,
            limits=p.CausalLimits(nodes=1),
        ),
        "limit",
    )
    for key in ["node_channels", "total_channels", "parameters", "working_bytes"]:
        rejects(
            "cap " + key,
            lambda key=key: p.FunctionalPrefix((r,), rows=5, limits=p.CausalLimits(**{key: 1})),
            "limit",
        )
    rejects(
        "huge rows admitted before arrays",
        lambda: p.FunctionalPrefix((rr,), rows=2**62),
        "working memory",
    )
    heavy = dataclasses.replace(
        rr,
        noise_scale=(1.0,),
        exogenous=p.ExogenousDistribution(p.InputDistribution("student_t", df=1.5)),
    )
    normalized = p.FunctionalCausalNode(
        (0,),
        p.FrozenMLPFunction(
            (p.FrozenLinearFunction(((1.0,),)),), (activation(),), start_activation=True
        ),
    )
    rejects(
        "finite fit does not certify heavy-tailed parent",
        lambda: p.FunctionalPrefix((heavy, normalized), rows=5),
        "second moments",
    )


# Retained metadata charged, and mutable dataclasses are accepted: no deep immutability claim.
@dataclasses.dataclass
class Extra:
    payload: object = ()


extra = Extra()
pr = p.FunctionalPrefix((rr,), rows=5, retained=(extra,))
before = pr.admit()
extra.payload = (1.0,) * 100
after = pr.admit()
assert after["parameters"] == before["parameters"] + 100
record("mutable retained metadata recharged qualification", {"before": before, "after": after})
extra.payload = (1.0,) * 1_000_001
rejects(
    "readmission rejects grown retained metadata before rng",
    lambda: p.evaluate_functional_prefix(pr, rng=blocked),
    "limit",
)
# Same-size retained mutation remains legal; tuple/frozen wrapper is not a deep hash seal.
extra.payload = (2.0,)
p.evaluate_functional_prefix(pr, rng=keyed(1))
record(
    "same-size retained mutation is accepted",
    "qualification: immutable shell, mutable retained dataclass not sealed",
)
dump("prefix-results", log)
print("ALL BOUNDED CHECKS PASS")
