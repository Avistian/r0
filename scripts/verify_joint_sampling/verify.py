"""Independent public-API checkpoint checks; no pytest or production interpolation oracle."""

import copy
import hashlib
import json
import platform
import sys
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from r0.modules import priors as p

OUT = Path(sys.argv[1])
OUT.mkdir(exist_ok=True)
checks = 0
rejections = 0
cases = 0


def save(name, value):
    (OUT / (name + ".json")).write_text(json.dumps(value, sort_keys=True, indent=2))


def ok(label, condition):
    global checks
    checks += 1
    print(json.dumps({"check": checks, "label": label, "pass": bool(condition)}), flush=True)
    assert condition, label


def reject(label, operation, text=None):
    global rejections
    try:
        operation()
    except (ValueError, TypeError, KeyError) as exc:
        rejections += 1
        print(
            json.dumps({"rejection": label, "type": type(exc).__name__, "message": str(exc)}),
            flush=True,
        )
        ok(label, text is None or text in str(exc))
    else:
        ok(label, False)


def raw(record):
    return json.loads(json.dumps(asdict(record)))


def state(rng):
    return json.dumps(rng.bit_generator.state, sort_keys=True, default=lambda x: x.tolist())


def independent_counts(r):
    # Explicit record-field accounting; no production count/admission helpers.
    draws = len(r["numerical"]["draws"]) + len(r["categorical"]["draws"])
    values = sum(len(d["values"]) for d in r["numerical"]["draws"])
    values += len(r["categorical"]["draws"])
    parameters = 0
    numerical_records = [r["numerical"]]
    for pref in r["categorical"]["preferences"]:
        if pref["unit"] is not None:
            parameters += 1
        if pref["weights"] is not None:
            w = pref["weights"]
            parameters += len(w["draws"]) + len(w["permutation"]) + 2
        if pref["hyperparameters"] is not None:
            h = pref["hyperparameters"]
            numerical_records.append(h)
            draws += len(h["draws"])
            values += sum(len(d["values"]) for d in h["draws"])
    parameters += 3 * len(r["categorical"]["draws"])
    for nr in numerical_records:
        for pref in nr["preferences"]:
            parameters += 2 + len(pref["request"]["shape"]) + len(pref["units"])
            parameters += 2 if pref["mode"] == "meta" else 0
        for d in nr["draws"]:
            parameters += 3 + len(d["request"]["shape"]) + len(d["units"]) + len(d["values"])
    for e in r["events"]:
        if e["kind"] == "forced_identity":
            draws += 1
        else:
            parameters += 1
    return draws, values, parameters


def run_recipe(mode="meta", policy="shared", limits=None, seed=128):
    global cases
    cases += 1
    rng = np.random.default_rng(seed)
    context = p.JointSamplingContext(
        rng, mode=mode, categorical_policy=policy, **({"limits": limits} if limits else {})
    )
    for owner in ("matrix", "node", "converter"):
        with context.transaction(owner) as s:
            s.numerical(
                "weight",
                p.NumericalParameter("simple_random_weights_std_scale", 0.0001, 10.0, scale="log"),
            )
            s.categorical(
                "activation", p.CategoricalParameter("acts_torch", ("sin", "relu", "tanh"))
            )
            s.forced_identity("target", reason="target-warp-disabled")
    return context.freeze(), rng


save(
    "environment",
    {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "source": "662fb89e098126990c2b6d5a86a4e6c189fc0321",
        "base": "32c24ec0fde27ec2ae096272f66e584ad3760e4c",
        "priors_file": p.__file__,
    },
)
save(
    "inputs",
    {
        "modes": ["meta", "local", "global"],
        "policies": ["shared", "refresh"],
        "seed": 128,
        "owners": ["matrix", "node", "converter"],
        "engines": ["PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64"],
    },
)
for mode in ("meta", "local", "global"):
    for policy in ("shared", "refresh"):
        r, _ = run_recipe(mode, policy)
        save(f"sharing-{mode}-{policy}", raw(r))
        ok(
            "numerical shared identity " + mode + policy,
            [d.preference for d in r.numerical.draws] == [0, 0, 0],
        )
        refresh = mode == "meta" and policy == "refresh"
        ok(
            "categorical shared identity " + mode + policy,
            [d.preference for d in r.categorical.draws] == ([0, 1, 2] if refresh else [0, 0, 0]),
        )
        ok("effective numerical mode", all(x.mode == mode for x in r.numerical.preferences))
        ok("effective categorical mode", all(x.mode == mode for x in r.categorical.preferences))
        if mode == "meta":
            nested = r.categorical.preferences[0].hyperparameters
            outer = r.numerical.preferences[0]
            nested_scale = nested.preferences[1]
            ok(
                "nested same-name scope is distinct",
                outer.request.name == nested_scale.request.name
                and (outer.mean, outer.concentration)
                != (nested_scale.mean, nested_scale.concentration),
            )
            if refresh:
                ok(
                    "fresh nested scopes each refresh",
                    len(
                        {
                            (
                                x.hyperparameters.preferences[1].mean,
                                x.hyperparameters.preferences[1].concentration,
                            )
                            for x in r.categorical.preferences
                        }
                    )
                    == 3,
                )
        count = independent_counts(raw(r))
        ok(
            "independent aggregate accounting",
            count == (r.draw_count, r.value_count, r.parameter_count),
        )
        ok("JSON roundtrip", p.joint_sampling_from_dict(raw(r)) == r)
        exact = p.PreferenceLimits(*count)
        tight_r, _ = run_recipe(mode, policy, exact)
        ok("exact aggregate limits", independent_counts(raw(tight_r)) == count)
        ok("tight JSON roundtrip", p.joint_sampling_from_dict(raw(tight_r)) == tight_r)
        for cap in ("max_draws", "max_values", "max_parameters"):
            below = replace(exact, **{cap: getattr(exact, cap) - 1})
            reject(
                "one below " + mode + policy + cap,
                lambda mode=mode, policy=policy, below=below: run_recipe(mode, policy, below),
                "limit",
            )
            damaged = raw(tight_r)
            damaged["limits"] = asdict(below)
            damaged["numerical"]["draws"][0]["units"] = ["poison-before-decode"]
            save("preflight-" + mode + policy + cap, damaged)
            reject(
                "raw preflight " + mode + policy + cap,
                lambda damaged=damaged: p.joint_sampling_from_dict(damaged),
                "limit",
            )

# Same name across distinct effective modes; independent arithmetic oracle.
c = p.JointSamplingContext(np.random.default_rng(15))
with c.transaction("modes") as s:
    for mode in ("meta", "local", "global"):
        s.numerical(mode, p.NumericalParameter("same", -3.0, 7.0, mode=mode))
        s.categorical("cat-" + mode, p.CategoricalParameter("same", ("a", "b"), mode=mode))
r = c.freeze()
ok(
    "different effective scopes",
    [d.preference for d in r.numerical.draws] == [0, 1, 2]
    and [d.preference for d in r.categorical.draws] == [0, 1, 2],
)
for d in r.numerical.draws:
    expected = float(Fraction(-3) * (1 - Fraction(d.units[0])) + Fraction(7) * Fraction(d.units[0]))
    ok("independent rational interpolation", d.values == (expected,))

for policy in ("scalar-quantile-v1", "strict-contract-v1"):
    c = p.JointSamplingContext(np.random.default_rng(315), mode="global", global_policy=policy)
    with c.transaction("global") as s:
        shape = () if policy.startswith("scalar") else (2, 3)
        req = p.NumericalParameter("q", -2.0, 5.0, shape=shape)
        a = s.numerical("a", req)
        b = s.numerical("b", req)
        ok(
            "global reuse " + policy,
            a.preference == b.preference and a.units == b.units and a.values == b.values,
        )
        changed = replace(req, low=3.0, high=17.0)
        if shape:
            reject(
                "strict bounds rejected",
                lambda changed=changed: s.numerical("bad", changed),
                "contract",
            )
            reject(
                "strict shape rejected",
                lambda req=req: s.numerical("bad", replace(req, shape=(6,))),
                "contract",
            )
        else:
            b = s.numerical("remapped", changed)
            u = Fraction(a.units[0])
            ok("scalar bounds independent oracle", b.values == (float(3 * (1 - u) + 17 * u),))
            reject(
                "scalar array rejected",
                lambda req=req: s.numerical("bad", replace(req, shape=(1,))),
                "scalar",
            )
        for kwargs in ({"integer": "round"}, {"boundary_mass": True}, {"scale": "log", "low": 1.0}):
            before = s.freeze()
            reject(
                "law mutation",
                lambda req=req, kwargs=kwargs: s.numerical("bad", replace(req, **kwargs)),
                "contract",
            )
            ok("rejected law atomic", s.freeze() == before)
    save("global-" + policy, raw(c.freeze()))

# Changing metadata is rejected in local/meta too; shape and bounds remain variable.
for mode in ("meta", "local"):
    c = p.JointSamplingContext(np.random.default_rng(81), mode=mode)
    with c.transaction("shape") as s:
        req = p.NumericalParameter("scale", 1.0, 2.0)
        a = s.numerical("a", req)
        b = s.numerical("b", replace(req, low=-10.0, high=10.0, shape=(2, 2)))
        ok("variable meta/local geometry", a.preference == b.preference and len(b.values) == 4)
        reject(
            "local/meta law mutation",
            lambda req=req: s.numerical("bad", replace(req, scale="log")),
            "contract",
        )

# Abort and caught errors must leave exact RNG and records unchanged, all engines.
req = p.NumericalParameter("sigma", 0.0, 1.0)
cat = p.CategoricalParameter("act", ("a", "b", "c"))
for engine in (
    np.random.PCG64,
    np.random.PCG64DXSM,
    np.random.MT19937,
    np.random.Philox,
    np.random.SFC64,
):
    rng = np.random.Generator(engine(78))
    c = p.JointSamplingContext(rng)
    before = c.freeze()
    initial = state(rng)

    def abort(c=c):
        with c.transaction("root") as parent:
            parent.numerical("sigma", req)
            with parent.transaction("child") as child:
                child.categorical("act", cat)
            raise ValueError("intentional abort")

    reject("whole abort " + engine.__name__, abort, "intentional")
    ok("whole rollback " + engine.__name__, c.freeze() == before and state(rng) == initial)
    handles = []
    with c.transaction("root") as parent:
        handles.append(parent)
        parent.numerical("sigma", req)
        before_child = parent.freeze()

        def child_abort(handles=handles):
            with parent.transaction("child") as child:
                handles.append(child)
                child.categorical("act", cat)
                raise ValueError("caught child")

        reject("caught nested error", child_abort, "caught child")
        ok("caught nested rollback", parent.freeze() == before_child)
        reject("caught duplicate", lambda: parent.numerical("sigma", req), "role")
        parent.categorical("act", cat)
        before_bad = parent.freeze()
        reject(
            "caught catalog change",
            lambda: parent.categorical("bad", replace(cat, choices=("x", "y"))),
            "contract",
        )
        ok("catalog rollback", before_bad == parent.freeze())
        with parent.transaction("child") as child:
            child.forced_identity("target", reason="disabled")
            reject("busy parent draw", lambda: parent.numerical("busy", req), "active child")
            reject("busy root freeze", c.freeze, "active child")
        parent.numerical("tail", req)
    for handle in handles:
        reject("stale handle", handle.freeze, "closed")
    control_rng = np.random.Generator(engine(78))
    control = p.JointSamplingContext(control_rng)
    with control.transaction("root") as parent:
        parent.numerical("sigma", req)
        parent.categorical("act", cat)
        with parent.transaction("child") as child:
            child.forced_identity("target", reason="disabled")
        parent.numerical("tail", req)
    ok(
        "caught failures RNG equivalence",
        c.freeze() == control.freeze() and state(rng) == state(control_rng),
    )
    before = c.freeze()

    def external_mutation(c=c, rng=rng):
        with c.transaction("external") as s:
            s.numerical("sigma", req)
            rng.random(9)

    control_rng.random(9)
    reject("external mutation " + engine.__name__, external_mutation, "RNG changed")
    ok(
        "external state preserved " + engine.__name__,
        c.freeze() == before and state(rng) == state(control_rng),
    )

# Ownership tampering: retain every changed input.
r, _ = run_recipe()
for label, change in (
    ("missing_event", lambda d: d["events"].pop(0)),
    ("duplicate_ref", lambda d: d["events"][3].update(draw_index=0)),
    ("absent_ref", lambda d: d["events"][0].update(draw_index=500)),
    ("bool_ref", lambda d: d["events"][0].update(draw_index=True)),
    ("wrong_kind", lambda d: d["events"][0].update(kind="categorical")),
    ("duplicate_role", lambda d: d["events"][1].update(role="weight")),
    ("reopened_path", lambda d: d["events"][6].update(operation=["matrix"])),
    ("empty_path", lambda d: d["events"][0].update(operation=[])),
    ("deep_path", lambda d: d["events"][0].update(operation=["x"] * 17)),
    ("bad_reason", lambda d: d["events"][2].update(reason=" bad")),
    ("identity_ref", lambda d: d["events"][2].update(draw_index=0)),
    ("sample_reason", lambda d: d["events"][0].update(reason="unexpected")),
    ("long_role", lambda d: d["events"][0].update(role="é" * 65)),
    ("child_limits", lambda d: d["numerical"]["limits"].update(max_draws=9999)),
    ("wrong_mode", lambda d: d["categorical"].update(default_mode="local")),
    ("wrong_policy", lambda d: d.update(global_policy="strict-contract-v1")),
    (
        "orphan_preference",
        lambda d: d["numerical"]["preferences"].append(
            copy.deepcopy(d["numerical"]["preferences"][0])
        ),
    ),
    ("wrong_pref_ref", lambda d: d["numerical"]["draws"][0].update(preference=9)),
    ("changed_scope_name", lambda d: d["numerical"]["draws"][1]["request"].update(name="other")),
    (
        "changed_cat_catalog",
        lambda d: d["categorical"]["draws"][1]["request"].update(choices=["x", "y", "z"]),
    ),
):
    d = raw(r)
    change(d)
    save("tamper-" + label, d)
    reject("tamper " + label, lambda d=d: p.joint_sampling_from_dict(d))

# Valid local law mutation, independently reconstructed, must fail joint validation.
c = p.JointSamplingContext(np.random.default_rng(18), mode="local")
with c.transaction("law") as s:
    s.numerical("first", p.NumericalParameter("law", 0.0, 1.0))
    s.numerical("second", p.NumericalParameter("law", 0.0, 1.0))
d = raw(c.freeze())
d["numerical"]["draws"][1]["request"]["integer"] = "round"
d["numerical"]["draws"][1]["values"] = [round(d["numerical"]["draws"][1]["units"][0])]
save("tamper-valid-child-law", d)
reject("valid child illegal joint law", lambda: p.joint_sampling_from_dict(d), "law contract")

# Maximum path depth and parent resume; operations consisting only of descendants.
c = p.JointSamplingContext(np.random.default_rng(11))


def nested(ctx, depth):
    if depth == 16:
        ctx.forced_identity("leaf", reason="disabled")

        def too_deep():
            with ctx.transaction("seventeen"):
                pass

        reject("depth 17", too_deep, "depth")
    else:
        with ctx.transaction("d" + str(depth)) as sub:
            nested(sub, depth + 1)


with c.transaction("d0") as s:
    nested(s, 1)
    s.forced_identity("resume", reason="disabled")
r = c.freeze()
ok(
    "depth 16 decode and parent resume",
    len(r.events[0].operation) == 16 and p.joint_sampling_from_dict(raw(r)) == r,
)


def reopen():
    with c.transaction("d0"):
        pass


reject("descendant-owned parent reopen", reopen, "completed")
for _ in range(2):
    with c.transaction("empty"):
        pass
ok("empty leaves no events", c.freeze() == r)

# Forced-only budgets and uint63 arithmetic without allocations proportional to caps.
for cap in (1, 17, 2**63 - 1):
    rng = np.random.default_rng(15)
    initial = state(rng)
    c = p.JointSamplingContext(rng, limits=p.PreferenceLimits(cap, 1, 1))
    with c.transaction("forced") as s:
        s.forced_identity("identity", reason="disabled")
    r = c.freeze()
    ok("forced-only exact accounting", independent_counts(raw(r)) == (1, 0, 0))
    ok("forced-only no RNG", state(rng) == initial)
    ok("forced-only decode", p.joint_sampling_from_dict(raw(r)) == r)

# Hash every actually imported project source; retain exact import custody.
imports = {}
for name, module in sorted(sys.modules.items()):
    filename = getattr(module, "__file__", None)
    if (name == "r0" or name.startswith("r0.")) and filename and Path(filename).is_file():
        imports[name] = {
            "path": filename,
            "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
        }
save("imports", imports)
save(
    "totals",
    {
        "checks": checks,
        "expected_rejections": rejections,
        "recipe_invocations": cases,
        "verdict": "PASS_COMPONENT",
        "limitations": [
            "No graph integration",
            "No ancestral fitting",
            "No browser/Colab/GPU",
            "No production-mixture or predictive validation",
        ],
    },
)
print(
    json.dumps(
        {
            "checks": checks,
            "expected_rejections": rejections,
            "recipe_invocations": cases,
            "verdict": "PASS_COMPONENT",
        }
    )
)
