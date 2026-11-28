"""Independent authored graph, record, admission, moment and local-save verification."""

import copy
import hashlib
import json
import math
import os
from dataclasses import asdict, replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import torch
from custody import record_custody
from numerical_oracle import expected_cdf

from r0.modules import priors
from r0.modules.explorer import ExplorerSession

OUT = Path(os.environ["EVIDENCE_DIR"])
RESULTS = []


def retain(name, data):
    RESULTS.append(dict(name=name, **data))
    with (OUT / "raw" / "integration.jsonl").open("a") as output:
        output.write(json.dumps(RESULTS[-1], default=lambda v: v.tolist()) + "\n")


def rejects(name, action):
    try:
        action()
    except (ValueError, TypeError, KeyError) as error:
        retain(name, dict(status="EXPECTED_REJECTION", error=str(error)))
    else:
        raise AssertionError(f"{name}: invalid record accepted")


def l2_expected(sample):
    with localcontext() as context:
        context.prec = 1000
        energy = sum((Decimal.from_float(float(x)) ** 2 for x in sample.flat), Decimal(0))
        return float(1 / ((energy / len(sample)).sqrt() + Decimal.from_float(1e-8)))


def main():
    record_custody(OUT / "raw" / "integration-custody.json")
    sample = np.array([[-2.0, -3.0, 1.0], [3.0, 2.0, 4.0], [0.0, 0.0, 2.0], [1.0, -1.0, 3.0]])
    l2 = priors.calibrate_l2_normalization(sample, calibration_id="independent-integration")
    kuma = priors.calibrate_kumaraswamy(
        sample, a=0.7, b=1.3, calibration_id="independent-integration"
    )
    eye = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for kind, transform in [("l2", l2), ("kumaraswamy", kuma)]:
        expected_digest = hashlib.sha256(b"4,3:" + sample.astype("<f8").tobytes()).hexdigest()
        assert transform.calibration_digest == expected_digest
        assert transform.parameter_count == (3 if kind == "l2" else 11)
        assert transform.working_extra_channels == (12 if kind == "l2" else 48)
        for node_type in (
            priors.ActivatedCausalNode,
            priors.StochasticCausalNode,
            priors.MatrixCausalNode,
        ):
            for placement in ("input", "output"):
                for task_type, classes in [("regression", 0), ("binary", 2), ("multiclass", 3)]:
                    kwargs = dict(
                        parents=(0,),
                        weights=eye,
                        bias=(0.0, 0.0, 0.0),
                        noise_scale=(0.0, 0.0, 0.0),
                        **{f"{placement}_activation": transform},
                    )
                    if node_type is priors.MatrixCausalNode:
                        kwargs["weight_construction"] = priors.FactorizedWeights(
                            left=eye, right=eye, diagonal=(1.0, 1.0, 1.0)
                        )
                    node = node_type(**kwargs)
                    controls = priors.LinearRecipe(
                        rows=16,
                        context_rows=8,
                        features=3,
                        signal_features=3,
                        active_features=3,
                        task_type=task_type,
                        classes=classes,
                    )
                    recipe = priors.CausalRecipe(
                        task=controls,
                        nodes=(
                            priors.CausalNode(bias=(0.0, 0.0, 0.0), noise_scale=(1.0, 1.0, 1.0)),
                            node,
                        ),
                        observed=((0, 0), (0, 1), (0, 2)),
                        target=tuple((1, i) for i in range(3 if classes == 3 else 1)),
                        version="causal-v2",
                    )
                    restored = priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
                    assert restored == recipe
                    generated = priors.generate_task(recipe, seed=3928, task_id=0)
                    expected = []
                    for row in generated.inputs.tolist():
                        expected.append(
                            [
                                x * l2_expected(sample)
                                if kind == "l2"
                                else expected_cdf(
                                    x,
                                    float(sample[:, j].min()),
                                    float(sample[:, j].max()),
                                    0.7,
                                    1.3,
                                )
                                for j, x in enumerate(row[: len(recipe.target)])
                            ]
                        )
                    torch.testing.assert_close(
                        generated.scores,
                        torch.tensor(expected, dtype=torch.float64),
                        atol=1e-14,
                        rtol=2e-12,
                    )
                    if task_type == "regression":
                        torch.testing.assert_close(generated.targets, generated.scores[:, 0])
                    trace = priors.explain_generation(generated, [15, 0, 7])
                    assert trace["target_max_error"] < 1e-12
                    longer = replace(recipe, task=replace(controls, rows=32, context_rows=5))
                    expanded = priors.generate_task(longer, seed=3928, task_id=0)
                    torch.testing.assert_close(
                        expanded.scores[:16], generated.scores, rtol=0, atol=0
                    )
                    mechanism = json.loads(generated.mechanism_json)
                    admission = mechanism["resource_admission"]
                    exact = replace(
                        recipe.limits,
                        parameters=admission["parameters"],
                        working_bytes=admission["estimated_working_bytes"],
                    )
                    replace(recipe, limits=exact)
                    name = f"{kind}-{node_type.__name__}-{placement}-{task_type}"
                    for field in ("parameters", "working_bytes"):
                        rejects(
                            name + "-limit-" + field,
                            lambda field=field, recipe=recipe, exact=exact: replace(
                                recipe, limits=replace(exact, **{field: getattr(exact, field) - 1})
                            ),
                        )
                    retain(
                        name,
                        dict(
                            status="PASS",
                            recipe=asdict(recipe),
                            expected=expected,
                            scores=generated.scores,
                            trace=trace,
                            admission=admission,
                        ),
                    )
                    if node_type is priors.ActivatedCausalNode and placement == "output":
                        session = ExplorerSession(OUT / "sessions" / name)
                        result = session.request(
                            "generate",
                            {"record": {"recipe": asdict(recipe), "seed": 3928, "task_id": 0}},
                        )
                        reopened = ExplorerSession(OUT / "sessions" / name).request(
                            "open", {"archive": result["archive"], "task_id": 0}
                        )
                        assert reopened["view"] == result["view"]
                        replay = ExplorerSession(OUT / "sessions" / (name + "-replay"))
                        replay.request("generate", {"record_json": result["recipe_json"]})
                        assert replay.request("row", {"index": 15}) == session.request(
                            "row", {"index": 15}
                        )
                        assert (
                            session.request("graph", {"indices": [0, 15]})["graph"][
                                "target_max_error"
                            ]
                            < 1e-12
                        )
                        retain(name + "-saved", dict(status="PASS", archive=result["archive"]))
        # Calibration allocation formula independently specialized to one root and 3 channels.
        count = 6 + transform.parameter_count
        working = (
            8 * 4 * (6 * 3 + 2 * 3 + 4 * 3 + 4 + transform.working_extra_channels) + 128 * count
        )
        fitter = priors.calibrate_l2_normalization if kind == "l2" else priors.calibrate_kumaraswamy
        arguments = {} if kind == "l2" else dict(a=0.7, b=1.3)
        exact = priors.CausalLimits(parameters=count, working_bytes=working)
        assert (
            fitter(sample, calibration_id="independent-integration", limits=exact, **arguments)
            == transform
        )
        for field in ("parameters", "working_bytes"):
            rejects(
                kind + "-fitting-" + field,
                lambda field=field, fitter=fitter, exact=exact, arguments=arguments: fitter(
                    sample,
                    calibration_id="independent-integration",
                    limits=replace(exact, **{field: getattr(exact, field) - 1}),
                    **arguments,
                ),
            )
        for change in [
            dict(channels=0),
            dict(channels=True),
            dict(calibration_rows=0),
            dict(calibration_id=""),
            dict(calibration_digest="Z" * 64),
            dict(calibration_version="unknown"),
            dict(factor=0),
            dict(factor=math.inf),
        ]:
            rejects(
                kind + "-record-" + str(change),
                lambda change=change, transform=transform: replace(transform, **change),
            )
        for values in [
            torch.ones(3),
            torch.ones(2, 2),
            torch.ones(2, 3, dtype=torch.int64),
            torch.full((2, 3), math.nan),
            torch.full((2, 3), math.inf),
        ]:
            rejects(
                kind + "-apply-invalid",
                lambda values=values, transform=transform: transform.apply(values),
            )
        for values in [np.array([]), np.array([[1]]), np.ones((0, 3)), np.full((2, 3), math.nan)]:
            rejects(
                kind + "-sample-invalid",
                lambda values=values, fitter=fitter, arguments=arguments: fitter(
                    values, calibration_id="invalid", **arguments
                ),
            )
    for changes in [
        dict(parameters=(("a", 0.0), ("b", 1.0))),
        dict(parameters=(("b", 1.0), ("a", 1.0))),
        dict(factor=2.0),
        dict(minimum=(4.0, 2.0, 1.0)),
        dict(minimum=(1.0,)),
        dict(maximum=(math.inf, 2.0, 4.0)),
    ]:
        rejects("kuma-invalid-shape-range", lambda changes=changes: replace(kuma, **changes))
    for kind in ("l2_normalize", "kumaraswamy"):
        rejects("primitive-bypass-" + kind, lambda kind=kind: priors.CausalActivation(kind=kind))
    # Saved decoder exercises malformed frozen metadata, not only direct constructors.
    encoded = json.loads(json.dumps(asdict(recipe)))
    for field, value in [
        ("calibration_rows", 0),
        ("calibration_version", "wrong"),
        ("calibration_digest", "broken"),
        ("channels", 2),
    ]:
        damaged = copy.deepcopy(encoded)
        damaged["nodes"][1]["output_activation"][field] = value
        rejects("decoder-" + field, lambda damaged=damaged: priors.recipe_from_dict(damaged))
    tiny = priors.calibrate_kumaraswamy(
        np.array([[0.0], [1e-30]]), a=1.0, b=1.0, calibration_id="diagnostic-fixture"
    )
    values = torch.tensor([[-1e-30], [0.0], [1e-30], [1.5e-30], [3e-30]], dtype=torch.float64)
    diagnostic = tiny.boundary_diagnostic(values)
    assert diagnostic["outside_calibration_fraction"] == [0.6]
    assert diagnostic["domain_clamp_fraction"] == [0.4]
    retain("diagnostic-clamp-vs-outside", dict(status="PASS", inputs=values, diagnostic=diagnostic))
    heavy = priors.ExogenousDistribution(
        inputs=priors.InputDistribution(distribution="student_t", df=0.5)
    )
    for kind, transform in [("l2", l2), ("kuma", kuma)]:
        for position in ("before_output", "after_output"):
            node = priors.StochasticCausalNode(
                bias=(0.0, 0.0, 0.0),
                noise_scale=(1.0, 1.0, 1.0),
                output_activation=transform,
                exogenous=heavy,
                noise_position=position,
            )
            candidate = priors.CausalRecipe(
                task=priors.LinearRecipe(
                    rows=16, context_rows=8, features=1, signal_features=1, active_features=1
                ),
                nodes=(node, priors.CausalNode()),
                observed=((1, 0),),
                target=((0, 0),),
                version="causal-v2",
                calibration_rows=32,
            )
            generated = priors.generate_task(candidate, seed=3928, task_id=0)
            certificate = json.loads(generated.mechanism_json)["moment_contract"]
            expected = "all" if kind == "kuma" and position == "before_output" else 0.5
            assert certificate["target_orders"] == [expected]
            if expected == "all":
                normalized = replace(candidate, normalize=True)
                priors.generate_task(normalized, seed=3928, task_id=0)
            else:
                rejects(
                    kind + "-" + position + "-second-moment",
                    lambda candidate=candidate: replace(candidate, normalize=True),
                )
            retain(kind + "-" + position + "-moments", dict(status="PASS", certificate=certificate))
    result = dict(
        status="PASS",
        checks=len(RESULTS),
        generated_compositions=36,
        saved_workflows=6,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (OUT / "integration-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
