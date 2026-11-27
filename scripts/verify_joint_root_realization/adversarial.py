"""Adversarial root checks through exported APIs and record constructors."""

from dataclasses import replace

import numpy as np
from oracle import base_oracle, close, require

from r0.modules import priors as p


def additional(roots, record, rejected, make_plan, materialize, state, payload):
    selections = {}
    for weight in ("uniform", "polynomial"):
        for seed in range(30):
            plan, decisions = make_plan("covariance", "lin", seed=seed, weight=weight)
            key = weight, plan.covariance_base
            if key in selections:
                continue
            root = materialize(plan, decisions)
            selections[key] = root, decisions
            b = root.calibration.base
            rng = np.random.default_rng(731)
            close(b.gaussian_factor, rng.normal(size=(2, 2)), "raw covariance A")
            if weight == "polynomial":
                draws = rng.normal(size=2)
                permutation = rng.permutation(2)
                close(b.weights.draws, draws, "raw polynomial draws")
                close(b.weights.permutation, permutation, "polynomial permutation")
                scores = (
                    b.weights.log_noise_scale * draws - b.weights.decay * np.log(np.arange(1, 3))
                )[permutation]
                scores -= scores.max()
                unnormalized = np.exp(scores)
                values = unnormalized / np.sqrt(np.mean(unnormalized**2))
            else:
                values = rng.random(2)
                close(b.weights.draws, values, "raw uniform weights")
            close(b.factor, np.asarray(b.gaussian_factor) * values[None], "column factor oracle")
            _, _, expected = base_oracle(b, np.random.default_rng(953), 5)
            close(root.calibration.values, expected, "covariance law")
            require(
                p.joint_root_realization_from_dict(payload(root), decisions=decisions) == root,
                "covariance roundtrip",
            )
            record(
                "covariance-combination",
                weight=weight,
                base=plan.covariance_base,
                seed=seed,
                status="PASS",
            )
    require(len(selections) == 4, "both covariance laws and weights exercised")
    root, decisions = selections["uniform", "gaussian"]
    base = replace(
        root.calibration.base,
        gaussian_factor=((1.0, 2.0), (3.0, 4.0)),
        weights=p.UniformWeightVector((0.5, 0.25)),
        factor=((0.5, 0.5), (1.5, 1.0)),
    )
    sample_type = type(root.calibration)
    sample_type(base, ((2.0, -4.0),), (), ((-1.0, -1.0),))
    covariance = np.asarray(base.factor) @ np.asarray(base.factor).T
    close(covariance, [[0.5, 1.25], [1.25, 3.25]], "literal covariance")
    rejected("literal-reject-row-weights", lambda: replace(base, factor=((0.5, 1.0), (0.75, 1.0))))
    sample_type(
        replace(base, gaussian_factor=((0.0, 0.0), (0.0, 0.0)), factor=((0.0, 0.0), (0.0, 0.0))),
        ((2.0, -4.0),),
        (),
        ((0.0, 0.0),),
    )
    ball = roots["ball", "lin"][0].calibration.base
    sample_type(ball, ((3.0, 4.0), (3.0, 4.0)), (0.25, 0.0), ((0.3, 0.4), (0.0, 0.0)))
    rejected("ball-is-not-sphere", lambda: sample_type(ball, ((3.0, 4.0),), (0.25,), ((0.6, 0.8),)))
    rejected(
        "zero-direction-no-retry", lambda: sample_type(ball, ((0.0, 0.0),), (0.25,), ((0.0, 0.0),))
    )
    record(
        "literal-covariance-ball",
        status="PASS",
        gaussian_covariance=covariance.tolist(),
        uniform_covariance=(covariance / 3).tolist(),
    )
    # Actual numerically valid stored root that overflows only on a later fresh draw.
    for seed in range(30):
        plan, decisions = make_plan("covariance", "lin", seed=seed, width=1, rows=3)
        if plan.covariance_base == "gaussian":
            break
    root = materialize(plan, decisions)
    data = payload(root)
    b = data["calibration"]["base"]
    b["gaussian_factor"] = [[1.6e308]]
    b["weights"]["draws"] = [0.95]
    b["factor"] = [[1.6e308 * 0.95]]
    data["calibration"]["draws"] = [[0.0]] * 3
    data["calibration"]["values"] = [[0.0]] * 3
    data["outputs"] = [[0.0]] * 3
    extreme = p.joint_root_realization_from_dict(data, decisions=decisions)
    rng = np.random.default_rng(3)
    before = state([rng])
    rejected(
        "real-postdraw-overflow",
        lambda: p.sample_joint_root_values(
            extreme, decisions, rows=2, rng=rng, sample_id="overflow"
        ),
    )
    require(state([rng]) == before, "real overflow rollback")
    recovered = p.sample_joint_root_values(root, decisions, rows=2, rng=rng, sample_id="recovery")
    require(
        recovered
        == p.sample_joint_root_values(
            root, decisions, rows=2, rng=np.random.default_rng(3), sample_id="recovery"
        ),
        "real overflow recovery",
    )
    record("real-overflow-recovery", status="PASS")
    # Find exact observed caller-owned admission boundaries, without relying on a helper estimate.
    plan, decisions = make_plan("uniform", "lin", rows=5)
    root = materialize(plan, decisions)

    def minimum(field, operation):
        lo, hi = 1, getattr(p.CausalLimits(), field)
        while lo < hi:
            mid = (lo + hi) // 2
            try:
                operation(replace(p.CausalLimits(), **{field: mid}))
            except ValueError:
                lo = mid + 1
            else:
                hi = mid
        operation(replace(p.CausalLimits(), **{field: lo}))
        return lo

    for field in ("parameters", "working_bytes"):

        def child_call(limits):
            return p.materialize_joint_function(
                plan.function,
                decisions,
                rng=np.random.default_rng(731),
                fit_rng=np.random.default_rng(1171),
                calibration=np.asarray(root.calibration.values),
                calibration_id=root.calibration_id,
                limits=limits,
            )

        def root_call(limits):
            return materialize(plan, decisions, limits=limits)

        child_min = minimum(field, child_call)
        root_min = minimum(field, root_call)
        require(root_min > child_min, "root aggregate must exceed child")
        limits = replace(p.CausalLimits(), **{field: root_min - 1})
        child_call(limits)
        streams = [np.random.default_rng(s) for s in (731, 953, 1171)]
        before = state(streams)
        rejected(
            "aggregate-" + field,
            lambda streams=streams, limits=limits: materialize(
                plan, decisions, streams, limits=limits
            ),
        )
        require(state(streams) == before, "aggregate fail before commit")

        def emission_call(limits):
            return p.sample_joint_root_values(
                root,
                decisions,
                rows=7,
                rng=np.random.default_rng(3),
                sample_id="limited",
                limits=limits,
            )

        emission_min = minimum(field, emission_call)
        require(emission_min > root_min, "emission aggregate exceeds retained root")
        rejected(
            "emission-aggregate-" + field,
            lambda field=field, emission_min=emission_min: emission_call(
                replace(p.CausalLimits(), **{field: emission_min - 1})
            ),
        )
        record(
            "aggregate-boundary",
            field=field,
            child=child_min,
            root=root_min,
            emission=emission_min,
            status="PASS",
            meaning="admission estimate, not measured RSS",
        )
    # Raw stream draw order is G(R,d) then U(R): repartitioning calls changes rows.
    root, decisions = roots["ball", "lin"]
    single = p.sample_joint_root_values(
        root, decisions, rows=7, rng=np.random.default_rng(9), sample_id="single"
    )
    rng = np.random.default_rng(9)
    first = p.sample_joint_root_values(root, decisions, rows=3, rng=rng, sample_id="first")
    second = p.sample_joint_root_values(root, decisions, rows=4, rng=rng, sample_id="second")
    require(
        single.sample.values != first.sample.values + second.sample.values,
        "document source batch draw order",
    )
    record(
        "ball-raw-partition",
        status="EXPECTED_DIFFERENCE",
        fixed_function_chunks="verified separately",
    )
