import json
from dataclasses import replace

import numpy as np
from common import (
    close,
    draw,
    fitted,
    ok,
    plain,
    save,
    task_settings,
)

from r0.modules import priors as p


def verify_ancestry():
    soft, _ = fitted("softmax", np.array([[-3.0], [-0.5], [0.7], [1.8]]), 489)
    for variant in ["historical-tuple", "historical-graph", "readout-only"]:
        rr = p.RootInputCausalNode(
            bias=(0.0, 0.0),
            root_scale=(1.0, 1.0),
            noise_scale=(0.0, 0.0),
            root_exogenous=p.ExogenousDistribution(factor=((1.0, 1.0), (0.0, 0.0))),
        )
        child2 = p.StochasticCausalNode(
            parents=(0,),
            weights=((1.0,), (0.0,)),
            noise_scale=(0.0,),
            noise_position="after_output",
        )
        ss = replace(task_settings(), features=1, signal_features=1, active_features=1)
        rec = p.CausalPipelineRecipe(
            task=ss,
            nodes=(rr, child2),
            observed=((0, 1),),
            target=((1, 0),),
            version="causal-v2",
            pipelines=(
                p.CausalNodePipeline(
                    channels=2,
                    converters=(p.NumericalConverter(channel=0, transform=soft, variant=variant),),
                ),
                p.CausalNodePipeline(channels=1),
            ),
        )
        for branch, target in [("state", ((1, 0),)), ("readout", ((0, 0),))]:
            t = p.generate_task(replace(rec, target=target), seed=444, task_id=2)
            card = json.loads(t.mechanism_json)
            expected = (
                [0]
                if (branch == "state" and variant == "readout-only")
                or (branch == "readout" and variant == "historical-tuple")
                else []
            )
            ok(
                "ancestry-" + variant + "-" + branch,
                card["active"] == expected,
                actual=card["active"],
                expected=expected,
            )
            save(
                "ancestry-" + variant + "-" + branch + ".json",
                dict(card=card, trace=p.explain_generation(t, [0, 9, 47])),
            )


def verify_calibration(root):
    # Independently reconstruct target calibration with separate stream prefix.
    rr = replace(root, output_activation=p.CausalActivation("tanh"))
    rec = replace(plain(rr), normalize=True, calibration_rows=64)
    t = p.generate_task(rec, seed=127, task_id=61)
    rd = draw(rr.root_exogenous, 64, 3, 127, 61, root=True, cal=True)
    aff = np.asarray(rr.bias) + rd * np.asarray(rr.root_scale)
    sc = np.asarray(rr.noise_scale) * (0.4 + 0.2 * abs(aff) ** 2)
    inn = sc * draw(rr.exogenous, 64, 3, 127, 61, cal=True)
    y = (np.tanh(aff) + inn)[:, 2]
    card = json.loads(t.mechanism_json)
    close("independent-calibration-center", card["calibration"]["center"], [y.mean()])
    close("independent-calibration-scale", card["calibration"]["scale"], [y.std()])
    save(
        "independent-calibration.json",
        dict(card=card, root_draw=rd, affine=aff, innovation=inn, target=y),
    )
