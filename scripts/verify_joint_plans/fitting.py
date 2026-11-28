"""Independent fit constants, budget arithmetic, generation and rollback."""

import copy
import json
from dataclasses import asdict, replace

import numpy as np
import torch
from common import OUT, ROOT, check, raw, reject, save

from r0.modules import priors as p

fixtures = json.loads((ROOT / "results/choices/activation-fixtures.json").read_text())
sample = np.array([[-3.0], [-1.0], [0.0], [2.0], [6.0]])
query = np.array([[-2.0], [0.25], [4.0]])
for kind in ("tanh", "signed_power"):
    seed, raw_plan, raw_record = fixtures["standard"][kind]
    record = p.joint_sampling_from_dict(raw_record)
    plan = p.joint_activation_plan_from_dict(raw_plan, decisions=record)
    rng = np.random.default_rng(83)
    control = np.random.default_rng(83)
    index = int(control.integers(5, size=1)[0])
    mu, inv = sample.mean(0), 1 / (sample.std(0) + 1e-4)
    shift = -((sample - mu) * inv)[index]
    scaled = plan.scale * ((sample - mu) * inv + shift)

    def primitive(x, kind=kind, plan=plan):
        return (
            np.tanh(x)
            if kind == "tanh"
            else np.sign(x) * abs(x) ** plan.activation.parameters[0][1]
        )

    transformed = primitive(scaled)
    out_mu, out_inv = transformed.mean(0), 1 / (transformed.std(0) + 1e-4)
    expected = (primitive(plan.scale * ((query - mu) * inv + shift)) - out_mu) * out_inv
    count = 10 + (kind == "signed_power")
    working = 8 * 5 * 26 + 128 * count
    limits = p.CausalLimits(parameters=count, working_bytes=working)
    fit = p.fit_joint_activation(
        plan, record, sample, rng=rng, calibration_id="independent-five-rows", limits=limits
    )
    check(
        "fit/standard-arithmetic-" + kind,
        np.allclose(fit.apply(torch.tensor(query)).numpy(), expected, rtol=1e-12, atol=1e-12),
        {"expected": expected, "fit": fit},
    )
    check("fit/standard-stream", rng.bit_generator.state == control.bit_generator.state)
    before = asdict(fit)
    fit.apply(torch.tensor([[300.0], [0.0], [-200.0]]))
    check("fit/no-refit", asdict(fit) == before)
    check(
        "fit/constants",
        np.allclose(fit.input_mean, mu)
        and np.allclose(fit.input_inverse_std, inv)
        and np.allclose(fit.shift, shift)
        and np.allclose(fit.output_mean, out_mu)
        and np.allclose(fit.output_inverse_std, out_inv),
    )
    for name in ("parameters", "working_bytes"):
        snapshot = copy.deepcopy(rng.bit_generator.state)
        reject(
            "fit/standard-one-below",
            lambda name=name, plan=plan, record=record, limits=limits, rng=rng: (
                p.fit_joint_activation(
                    plan,
                    record,
                    sample,
                    rng=rng,
                    calibration_id="independent-five-rows",
                    limits=replace(limits, **{name: getattr(limits, name) - 1}),
                )
            ),
        )
        check("fit/standard-reject-rng", snapshot == rng.bit_generator.state)
    save(
        "standard-" + kind,
        {
            "seed": seed,
            "plan": plan,
            "decisions": record,
            "sample": sample,
            "query": query,
            "fit": fit,
            "expected": expected,
            "limits": limits,
        },
    )

for mode in ("id", "kumar", "act"):
    for variant in ("historical-tuple", "historical-graph", "readout-only"):
        prior = p.JointNumericalConverterPrior(
            allow_kumaraswamy=mode == "kumar", allow_activation=mode == "act", variant=variant
        )
        for seed in range(100):
            cc = p.JointSamplingContext(np.random.default_rng(seed), mode="local")
            plan = p.sample_joint_converter(
                cc, "converter", channel=0, feature_group="x", prior=prior
            )
            if plan.mode == mode and (mode != "act" or plan.activation.activation.kind == "tanh"):
                break
        check(
            "fit/converter-branch-found",
            plan.mode == mode and (mode != "act" or plan.activation.activation.kind == "tanh"),
        )
        record = cc.freeze()
        parameters, channels = {"id": (3, 16), "kumar": (10, 32), "act": (11, 26)}[mode]
        work = 8 * 5 * channels + 128 * parameters
        limits = p.CausalLimits(parameters=parameters, working_bytes=work)
        rng = np.random.default_rng(7)
        converter = p.fit_joint_converter(
            plan, record, sample, rng=rng, calibration_id="independent-five-rows", limits=limits
        )
        check("fit/converter-metadata", converter.parameter_count == parameters - 2)
        if mode == "id":
            expected = query
            warped = 0.25
        elif mode == "kumar":
            a, b = plan.shapes
            expected = 1 - (1 - np.clip((query + 3) / 9, 0, 1) ** a) ** b
            warped = 1 - (1 - (3.25 / 9) ** a) ** b
        else:
            index = int(np.random.default_rng(7).integers(5, size=1)[0])
            standardized = (sample - sample.mean(0)) / (sample.std(0) + 1e-4)
            shift = -standardized[index]
            transformed = np.tanh(plan.activation.scale * (standardized + shift))

            def warp(values, plan=plan, shift=shift, transformed=transformed):
                return (
                    np.tanh(
                        plan.activation.scale
                        * ((values - sample.mean(0)) / (sample.std(0) + 1e-4) + shift)
                    )
                    - transformed.mean(0)
                ) / (transformed.std(0) + 1e-4)

            expected = warp(query)
            warped = float(warp(np.array([[0.25]]))[0, 0])
        check(
            "fit/converter-arithmetic",
            np.allclose(
                converter.transform.apply(torch.tensor(query)).numpy(),
                expected,
                atol=1e-12,
                rtol=1e-12,
            ),
        )
        for name in ("parameters", "working_bytes"):
            snapshot = copy.deepcopy(rng.bit_generator.state)
            reject(
                "fit/converter-one-below",
                lambda name=name, plan=plan, record=record, limits=limits, rng=rng: (
                    p.fit_joint_converter(
                        plan,
                        record,
                        sample,
                        rng=rng,
                        calibration_id="independent-five-rows",
                        limits=replace(limits, **{name: getattr(limits, name) - 1}),
                    )
                ),
            )
            check("fit/converter-rollback", snapshot == rng.bit_generator.state)
        recipe = p.CausalPipelineRecipe(
            task=p.LinearRecipe(
                rows=16, context_rows=8, features=1, signal_features=1, active_features=1
            ),
            nodes=(
                p.ActivatedCausalNode(
                    bias=(0.25,), noise_scale=(0.0,), noise_position="after_output"
                ),
                p.ActivatedCausalNode(
                    parents=(0,),
                    weights=((3.0,),),
                    noise_scale=(0.0,),
                    noise_position="after_output",
                ),
            ),
            pipelines=(
                p.CausalNodePipeline(channels=1, converters=(converter,), node_importance=1.5),
                p.CausalNodePipeline(channels=1),
            ),
            observed=((0, 0),),
            target=((1, 0),),
            version="causal-v2",
        )
        restored = p.recipe_from_dict(raw(recipe))
        check("fit/recipe-json", restored == recipe)
        task = p.generate_task(restored, seed=61, task_id=13)
        feature = 0.25 if variant == "historical-tuple" else warped
        score = 1.125 if variant == "readout-only" else 4.5 * warped
        check("fit/generation-feature", np.allclose(task.inputs.numpy(), feature, atol=1e-12))
        check("fit/generation-target", np.allclose(task.scores.numpy(), score, atol=1e-12))
        trace = p.explain_generation(task, [0, 15])
        check("fit/trace", trace["target_max_error"] < 1e-12)
        save(
            mode + "-" + variant,
            {
                "sampling_seed": seed,
                "plan": plan,
                "record": record,
                "sample": sample,
                "query": query,
                "converter": converter,
                "recipe": recipe,
                "task": task,
                "trace": trace,
                "limits": limits,
            },
        )
        # Channels and variants are graph destinations, not ledger-owned sampling fields.
        for key, value in [
            ("mode", "wrong"),
            ("operation", "converter"),
            ("shapes", {"a": 1, "b": 2}),
            ("channel", True),
        ]:
            data = raw(plan)
            data[key] = value
            reject(
                "converter/decode-" + key,
                lambda data=data, record=record: p.joint_converter_plan_from_dict(
                    data, decisions=record
                ),
            )

# All constant samples remain valid; mechanisms are frozen, not silently discarded.
for kind in ("identity", "tanh", "softmax"):
    _, rp, rr = fixtures["standard"][kind]
    record = p.joint_sampling_from_dict(rr)
    plan = p.joint_activation_plan_from_dict(rp, decisions=record)
    fit = p.fit_joint_activation(
        plan, record, np.zeros((5, 1)), rng=np.random.default_rng(2), calibration_id="zero"
    )
    check(
        "fit/constant-" + kind,
        np.array_equal(
            fit.apply(torch.zeros((2, 1), dtype=torch.float64)).numpy(), np.zeros((2, 1))
        ),
    )
# Exp fails numerically after the private staged stream has selected shift rows.
_, rp, rr = fixtures["standard"]["exp"]
record = p.joint_sampling_from_dict(rr)
plan = p.joint_activation_plan_from_dict(rp, decisions=record)
rows = int((800 / plan.scale) ** 2) + 10
sample = np.zeros((rows, 1))
sample[-1] = 1e6
rng = np.random.default_rng(17)
before = copy.deepcopy(rng.bit_generator.state)
reject(
    "fit/late-exp-overflow",
    lambda: p.fit_joint_activation(plan, record, sample, rng=rng, calibration_id="overflow"),
)
check("fit/late-rng-rollback", before == rng.bit_generator.state)
save(
    "late-failure",
    {"plan": plan, "record": record, "rows": rows, "zeros_except_last": 1e6, "rng_seed": 17},
)
print("fitting complete", OUT)
