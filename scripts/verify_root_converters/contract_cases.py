import json
from dataclasses import asdict, replace

import numpy as np
from common import (
    close,
    fitted,
    ok,
    plain,
    reject,
    save,
    task_settings,
)

from r0.modules import priors as p


def verify_moments(act, root, pipe, child):
    # Mathematical moment gates are checked against Student-t integrability thresholds.
    for variant in ["historical-tuple", "historical-graph", "readout-only"]:
        for df in [1.5, 3.0, 5.0]:
            rr = replace(
                root,
                root_exogenous=p.ExogenousDistribution(
                    inputs=p.InputDistribution(distribution="student_t", df=df)
                ),
                noise_scale=(0.0, 0.0, 0.0),
            )
            pi = replace(
                pipe,
                stages=(),
                channel_importance=(),
                l2=None,
                converters=(p.NumericalConverter(channel=0, transform=act, variant=variant),),
            )

            def build(rr=rr, pi=pi):
                return p.CausalPipelineRecipe(
                    task=task_settings(),
                    nodes=(rr, child),
                    observed=((0, 0), (0, 1)),
                    target=((1, 0),),
                    version="causal-v2",
                    pipelines=(pi, p.CausalNodePipeline(channels=3)),
                )

            if df <= 4:
                reject(f"moments-{variant}-{df}", build)
            else:
                ok(f"moments-{variant}-{df}", build() is not None)
            ok(
                f"bounded-moments-{variant}-{df}",
                build(replace(rr, output_activation=p.CausalActivation("tanh"))) is not None,
            )
    for pwr in [0.5, 1.0, 2.0, 3.0]:
        rr = replace(
            root,
            root_exogenous=p.ExogenousDistribution(
                inputs=p.InputDistribution(distribution="student_t", df=3.0)
            ),
            noise_response=p.NoiseResponse(offset=0.1, strength=0.7, power=pwr),
        )
        t = p.generate_task(plain(rr), seed=213, task_id=4)
        order = json.loads(t.mechanism_json)["moment_contract"]["target_orders"]
        close(f"moment-noise-power-{pwr}", order, [min(3, 3 / pwr)])


def verify_records():
    # Resource estimate independently counted for width-3 root, no activation/converter extras.
    rr = p.RootInputCausalNode(
        bias=(0.0, 0.0, 0.0),
        root_scale=(1.0, 1.0, 1.0),
        noise_scale=(0.1, 0.2, 0.3),
        root_exogenous=p.ExogenousDistribution(
            inputs=p.InputDistribution(distribution="student_t", df=0.8, bound=1.1)
        ),
    )
    params = 2 * 3 + 4 + 3 + 5 + 3  # bias/noise, innovation law, response, root law, scales
    working = 8 * 256 * (6 * 3 + 2 * 2 + 4 * 1 + 4 + 16 * 3) + 128 * params
    rec = plain(rr)
    card = json.loads(p.generate_task(rec, seed=23, task_id=12).mechanism_json)
    ok(
        "resource-parameter-count",
        card["resource_admission"]["parameters"] == params,
        expected=params,
        actual=card["resource_admission"],
    )
    ok(
        "resource-working-bytes",
        card["resource_admission"]["estimated_working_bytes"] == working,
        expected=working,
        actual=card["resource_admission"],
    )
    exact = replace(rec, limits=p.CausalLimits(parameters=params, working_bytes=working))
    ok("exact-resource-boundary", exact is not None)
    reject(
        "one-byte-resource-over",
        lambda: replace(rec, limits=p.CausalLimits(working_bytes=working - 1)),
    )
    reject(
        "one-parameter-resource-over",
        lambda: replace(rec, limits=p.CausalLimits(parameters=params - 1)),
    )
    reject("parented-root", lambda: replace(rr, parents=(0,), weights=((1.0, 0.0, 0.0),)))
    reject("bad-root-width", lambda: replace(rr, root_scale=(1.0,)))
    reject(
        "bad-root-factor-width",
        lambda: replace(rr, root_exogenous=p.ExogenousDistribution(factor=((1.0,),))),
    )
    reject("bad-root-version", lambda: replace(rr, root_version="root-input-v2"))
    reject("bad-root-law-type", lambda: replace(rr, root_exogenous={"inputs": {}}))
    reject(
        "before-output-pipeline",
        lambda: p.CausalPipelineRecipe(
            task=rec.task,
            nodes=(replace(rr, noise_position="before_output"),),
            observed=rec.observed,
            target=rec.target,
            version="causal-v2",
            pipelines=(p.CausalNodePipeline(channels=3),),
        ),
    )
    reject(
        "uncalibrated-converter",
        lambda: p.NumericalConverter(channel=0, transform=p.CausalActivation("tanh")),
    )
    wide, _ = fitted("tanh", np.array([[-1.0, 2.0], [0.2, 0.6], [1.3, -0.8]]), 488)
    reject("wide-converter", lambda: p.NumericalConverter(channel=0, transform=wide))
    bad = asdict(rec)
    bad["nodes"][0]["root_version"] = "root-input-v9"
    save("rejected-record.json", bad)
    reject("decode-bad-root-version", lambda: p.recipe_from_dict(bad))
