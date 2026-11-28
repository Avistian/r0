"""Independent complete-subtree request and raw decoder adversaries."""

import copy
import json
from dataclasses import asdict, replace

import numpy as np
import torch
from common import ROOT, check, raw, reject, save

from r0.modules import priors as p

fixtures = json.loads((ROOT / "results/choices/activation-fixtures.json").read_text())
_, rp, rr = fixtures["standard"]["tanh"]
record = p.joint_sampling_from_dict(rr)
plan = p.joint_activation_plan_from_dict(rp, decisions=record)
# Rebuild each request with one independently altered semantic contract. Preserve all roles.
for field, value in [
    ("name", "another-scale"),
    ("mode", "local"),
    ("low", 0.9),
    ("high", 11.0),
    ("shape", (2,)),
    ("scale", "log"),
    ("boundary_mass", True),
]:
    rng = np.random.default_rng(17)
    cc = p.JointSamplingContext(rng, mode="local")
    with cc.transaction("act") as child:
        child.categorical("selector", record.categorical.draws[0].request)
        request = replace(record.numerical.draws[0].request, **{field: value})
        child.numerical("scale", request)
        child.categorical("primitive", record.categorical.draws[1].request)
    reject("semantic/activation-" + field, lambda cc=cc: plan.validate(cc.freeze()))
# Extra descendant event and extra same-operation event must fail, unrelated subtree is legal.
for nested in (False, True):
    cc = p.JointSamplingContext(np.random.default_rng(13), mode="local")
    with cc.transaction("act") as child:
        p0 = p.sample_joint_activation(child, "base")
        if nested:
            with child.transaction("base-extra") as extra:
                extra.forced_identity("extra", reason="probe")
        else:
            child.forced_identity("extra", reason="probe")
    p0.validate(cc.freeze())
    check("semantic/unrelated-permitted", True)
    moved = tuple(
        replace(e, operation=(*p0.operation, "child")) if e.kind == "forced_identity" else e
        for e in cc.freeze().events
    )
    reject(
        "semantic/extra-subtree",
        lambda cc=cc, p0=p0, moved=moved: p0.validate(replace(cc.freeze(), events=moved)),
    )
for index in range(len(record.events)):
    reject(
        "semantic/missing-event",
        lambda index=index: plan.validate(
            replace(record, events=record.events[:index] + record.events[index + 1 :])
        ),
    )
for change in ({"operation": ("act", "child")}, {"role": "exponent"}):
    reject(
        "semantic/wrong-branch-path",
        lambda change=change: plan.validate(
            replace(record, events=(*record.events[:-1], replace(record.events[-1], **change)))
        ),
    )
for key, value in [
    ("version", "future"),
    ("unknown", 1),
    ("operation", ["x"] * 17),
    ("operation", ["x" * 129]),
    ("operation", [True]),
]:
    data = raw(plan)
    data[key] = value
    reject(
        "raw/activation-" + key,
        lambda data=data: p.joint_activation_plan_from_dict(data, decisions=record),
    )

for selector, kind in [
    ("plain", "integer_power"),
    ("plain", "relu_power"),
    ("plain", "signed_power"),
    ("plain", "reciprocal_abs_power"),
]:
    _, rp, rr = fixtures[selector][kind]
    ledger = p.joint_sampling_from_dict(rr)
    pp = p.joint_activation_plan_from_dict(rp, decisions=ledger)
    rng = np.random.default_rng(11)
    before = copy.deepcopy(rng.bit_generator.state)
    fitted = p.fit_joint_activation(pp, ledger, None, rng=rng, calibration_id="unused")
    check("plain/no-fit-" + kind, fitted == pp.activation and before == rng.bit_generator.state)
    x = np.array([[-2.0], [0.0], [1.5]])
    exponent = pp.activation.parameters[0][1]
    expected = {
        "integer_power": lambda x=x, exponent=exponent: x ** int(exponent),
        "relu_power": lambda x=x, exponent=exponent: np.maximum(x, 0) ** exponent,
        "signed_power": lambda x=x, exponent=exponent: np.sign(x) * abs(x) ** exponent,
        "reciprocal_abs_power": lambda x=x, exponent=exponent: (abs(x) + 0.001) ** (-exponent),
    }[kind]()
    check(
        "plain/power-arithmetic-" + kind,
        np.allclose(fitted.apply(torch.tensor(x)).numpy(), expected, rtol=1e-12, atol=1e-12),
    )

cc = p.JointSamplingContext(np.random.default_rng(44))
prior = p.JointNumericalConverterPrior(disallow_target_warping=True)
forced = p.sample_joint_converter(cc, "target", channel=0, feature_group="y", prior=prior)
ledger = cc.freeze()
for change in ({"reason": "different"}, {"role": "selector"}, {"operation": ("another",)}):
    reject(
        "forced/semantic-change",
        lambda change=change: forced.validate(
            replace(ledger, events=(replace(ledger.events[0], **change),))
        ),
    )
for field, value in [
    ("feature_group", "Y"),
    ("prior", raw(replace(prior, disallow_target_warping=False))),
    ("mode", "act"),
]:
    data = raw(forced)
    data[field] = value
    reject(
        "forced/plan-change",
        lambda data=data: p.joint_converter_plan_from_dict(data, decisions=ledger),
    )

# Whole-composite sampling rollback at every observed local choice-budget boundary.
for kind in ("activation", "converter"):

    def build(context, kind=kind):
        return (
            p.sample_joint_activation(context, "plan")
            if kind == "activation"
            else p.sample_joint_converter(context, "plan", channel=0, feature_group="x")
        )

    cc = p.JointSamplingContext(np.random.default_rng(0), mode="meta")
    baseline = build(cc)
    ledger = cc.freeze()
    caps = dict(
        max_draws=ledger.draw_count,
        max_values=ledger.value_count,
        max_parameters=ledger.parameter_count,
    )
    save("choice-budget-" + kind, {"caps": caps, "plan": baseline, "record": ledger})
    for field in ("exact", *caps):
        limit = caps.copy()
        if field != "exact":
            limit[field] -= 1
        rng = np.random.default_rng(0)
        other = p.JointSamplingContext(rng, mode="meta", limits=p.PreferenceLimits(**limit))
        snapshot = (asdict(other.freeze()), copy.deepcopy(rng.bit_generator.state))
        if field == "exact":
            check("choice/exact-" + kind, build(other) == baseline)
        else:
            reject("choice/below-" + kind + "-" + field, lambda other=other: build(other))
            check(
                "choice/atomic-" + kind,
                snapshot == (asdict(other.freeze()), rng.bit_generator.state),
            )

cc = p.JointSamplingContext(np.random.default_rng(2))
empty = cc.freeze()
for policy in ("scalarized-weight-batches-v1", "source-weight-batches-v1"):
    data = {
        "operation": ["w"],
        "width": 4,
        "batch": 100001,
        "scalarization": policy,
        "decays": [],
        "scales": [],
    }
    reject(
        "raw/weight-preadmission-" + policy,
        lambda data=data: p.joint_weight_plan_from_dict(data, decisions=empty),
    )
    rng = np.random.default_rng(2)
    cc = p.JointSamplingContext(
        rng, limits=p.PreferenceLimits(max_draws=2, max_values=2, max_parameters=2)
    )
    before = (asdict(cc.freeze()), copy.deepcopy(rng.bit_generator.state))
    reject(
        "choice/huge-batch-preadmission",
        lambda cc=cc, policy=policy: p.sample_joint_weights(
            cc, "huge", width=4, batch=100001, scalarization=policy
        ),
    )
    check("choice/huge-rollback", before == (asdict(cc.freeze()), rng.bit_generator.state))
print("adversaries complete")

# In-range stale numeric fields and swapped branch roles must still reject.
cc = p.JointSamplingContext(np.random.default_rng(0), mode="global")
converter = p.sample_joint_converter(cc, "conv", channel=0, feature_group="x")
ledger = cc.freeze()
check("semantic/kumar-fixture", converter.mode == "kumar")
assert converter.shapes is not None
changed = (0.3 if converter.shapes[0] == 0.2 else 0.2, converter.shapes[1])
reject("semantic/stale-kumar-value", lambda: replace(converter, shapes=changed).validate(ledger))
reject(
    "semantic/kumar-swapped-roles",
    lambda: converter.validate(
        replace(
            ledger,
            events=(
                ledger.events[0],
                replace(ledger.events[1], role="b"),
                replace(ledger.events[2], role="a"),
            ),
        )
    ),
)
cc = p.JointSamplingContext(np.random.default_rng(1))
weights = p.sample_joint_weights(cc, "weights", width=4, batch=3)
ledger = cc.freeze()
stale = (1.0 if weights.decays[0] != 1.0 else 2.0, *weights.decays[1:])
reject("semantic/stale-weight-value", lambda: replace(weights, decays=stale).validate(ledger))
reject("semantic/wrong-weight-width-contract", lambda: replace(weights, width=5).validate(ledger))
check(
    "scope/uniform-registry-pending",
    p.JointWeightPlan(("unregistered",), 2, family="uniform").parameter_count == 2,
)
