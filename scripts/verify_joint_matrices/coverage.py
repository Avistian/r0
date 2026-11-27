"""Public API source laws, policies, retained counts, JSON, generation and traces."""

import copy
import itertools
import json
from dataclasses import asdict

import numpy as np
from support import (
    ROOT,
    check,
    close,
    finish,
    matrix,
    matrix_count,
    point_count,
    points,
    raw,
    reject,
)

from r0.modules import priors as p

variants = [(f, "gaussian") for f in ("gaussian", "importance", "factorized", "kernel")]
variants += [("transformed", f) for f in ("gaussian", "importance", "factorized", "kernel")]
fixtures = []
for index, (mode, category, vector, batch_policy, variant) in enumerate(
    itertools.product(
        ("meta", "local", "global"),
        ("shared", "refresh"),
        ("polynomial", "uniform"),
        ("scalarized-weight-batches-v1", "source-weight-batches-v1"),
        variants,
    )
):
    family, child = variant
    global_policy = (
        "strict-contract-v1" if batch_policy.startswith("source") else "scalar-quantile-v1"
    )
    named_rng = np.random.default_rng(103)
    context = p.JointSamplingContext(
        named_rng, mode=mode, categorical_policy=category, global_policy=global_policy
    )
    before, state = context.freeze(), copy.deepcopy(named_rng.bit_generator.state)
    prior = p.JointMatrixPrior(
        families=(family,),
        simple_families=(child,),
        vector_family=vector,
        scalarization=batch_policy,
    )

    def sample(context=context, prior=prior):
        return p.sample_joint_matrix(context, "matrix", rows=2, columns=3, batch=2, prior=prior)

    strict_fail = (
        mode == "global"
        and vector == "polynomial"
        and batch_policy.startswith("source")
        and (family == "kernel" or family == "transformed" and child == "kernel")
    )
    if strict_fail:
        reject(sample, "strict incompatible kernel " + str(index))
        check(
            context.freeze() == before and named_rng.bit_generator.state == state,
            "planning rollback",
        )
        continue
    plan = sample()
    decisions = context.freeze()
    check(p.joint_matrix_plan_from_dict(raw(plan), decisions=decisions) == plan, "plan replay")
    check(p.joint_sampling_from_dict(raw(decisions)) == decisions, "ledger replay")
    raw_rng = np.random.default_rng(817)
    oracle_rng = np.random.default_rng(817)
    result = p.materialize_joint_matrix(plan, decisions, rng=raw_rng)
    expected = matrix(plan, oracle_rng)
    close(result.materialize(), expected, "independent literal matrix " + str(index))
    check(raw_rng.bit_generator.state == oracle_rng.bit_generator.state, "full raw order")
    check(result.parameter_count == matrix_count(plan), "retained parameter formula")
    payload = raw(result)
    restored = p.joint_matrix_realization_from_dict(payload, decisions=decisions)
    check(restored == result, "realization replay")
    if family == "transformed":
        check(plan.child.family == child and plan.activation.selector == "plain", "simple child")
        events = decisions.events
        activation_index = next(i for i, e in enumerate(events) if e.operation[-1] == "activation")
        check(
            all(e.operation[1] == "child" for e in events[1:activation_index]),
            "child before activation",
        )
    kernel_plan = plan.child if family == "transformed" else plan
    if kernel_plan.family == "kernel":
        cloud = result.point_cloud
        check(cloud.shape == (10, 3), "one batch-shared cloud")
        check(cloud.parameter_count == point_count(kernel_plan.points), "point parameter formula")
        check(
            not any(d.request.name == "kernel_point_law" for d in decisions.categorical.draws),
            "no fake selector",
        )
        check(
            decisions.events[-1 if family == "kernel" else activation_index - 1].role == "distance",
            "distance after cloud",
        )
        for i, outer in enumerate(result.matrices):
            source = outer.source.child.source if family == "transformed" else outer.source
            check(source.point_construction is None, "no duplicated cloud")
            close(
                source.left_points, np.asarray(cloud.materialize())[i * 5 : i * 5 + 2], "left slice"
            )
            close(
                source.right_points,
                np.asarray(cloud.materialize())[i * 5 + 2 : i * 5 + 5],
                "right slice",
            )
    if index < 8:
        fixtures.append({"index": index, "decisions": asdict(decisions), "realization": payload})
        for member in restored.matrices:
            values = member.materialize()
            recipe = p.CausalRecipe(
                task=p.LinearRecipe(
                    rows=16, context_rows=8, features=1, signal_features=1, active_features=1
                ),
                nodes=(
                    p.ActivatedCausalNode(bias=(1.25, -2.5), noise_scale=(0.0, 0.0)),
                    p.MatrixCausalNode(
                        parents=(0,),
                        weights=values,
                        bias=(0.0, 0.0, 0.0),
                        noise_scale=(0.0, 0.0, 0.0),
                        weight_construction=member,
                    ),
                ),
                observed=((0, 0),),
                target=((1, 2),),
                version="causal-v2",
            )
            decoded = p.recipe_from_dict(raw(recipe))
            task = p.generate_task(decoded, seed=551, task_id=12)
            close(
                task.scores.numpy(),
                np.full((16, 1), 1.25 * values[0][2] - 2.5 * values[1][2]),
                "generation",
            )
            check(p.explain_generation(task, [0, 7, 15])["target_max_error"] < 1e-12, "trace")
    print(
        json.dumps(
            {
                "case": index,
                "mode": mode,
                "category": category,
                "vector": vector,
                "policy": batch_policy,
                "family": family,
                "child": child,
                "status": "PASS",
            }
        ),
        flush=True,
    )
(ROOT / "results" / "fixtures.json").write_text(json.dumps(fixtures, indent=2))
check(
    p.JointMatrixPrior(families=("gaussian",)).simple_families
    == ("gaussian", "importance", "factorized", "kernel"),
    "Gaussian top keeps simple catalog",
)
for vector in ("polynomial", "uniform"):
    ctx = p.JointSamplingContext(np.random.default_rng(78), mode="local")
    plan = p.sample_joint_points(ctx, "points", count=7, dimension=2, vector_family=vector)
    rng, control = np.random.default_rng(5), np.random.default_rng(5)
    cloud = p.materialize_joint_points(plan, ctx.freeze(), rng=rng)
    expected, details = points(plan, control)
    close(cloud.materialize(), expected, "standalone point orientation and signed scaling")
    check(cloud.scale_coefficient == details["coefficient"], "signed scale coefficient")
    check(rng.bit_generator.state == control.bit_generator.state, "point raw order")
    check(
        p.joint_point_plan_from_dict(raw(plan), decisions=ctx.freeze()) == plan, "point plan replay"
    )
finish("coverage")
