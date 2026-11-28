"""Portable read-only root review; writes one fresh run directory per execution."""

import copy
import hashlib
import json
import os
import pickle
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from oracle import base_oracle, close, evaluate, require

from r0.modules import priors as p

BUNDLE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("R0_REVIEW_ROOT", Path.cwd())).resolve()
OUT = Path(tempfile.mkdtemp(prefix="run-", dir=BUNDLE))
LOG = OUT / "checks.jsonl"
HEAD = "78632956b1caf236ba521397d564311e2b006465"


def record(name, **details):
    with LOG.open("a") as stream:
        stream.write(json.dumps({"check": name, **details}) + "\n")


def rejected(name, operation):
    try:
        operation()
    except ValueError as error:
        record(name, status="PASS", reason=str(error))
    else:
        record(name, status="REJECT", reason="accepted invalid input")
        raise RuntimeError(f"{name}: invalid input accepted")


def state(generators):
    return pickle.dumps([g.bit_generator.state for g in generators])


def make_plan(base, function, seed=47, weight="uniform", width=2, rows=5):
    context = p.JointSamplingContext(np.random.default_rng(seed), mode="local")
    plan = p.sample_joint_root(
        context,
        "audit",
        width=width,
        calibration_rows=rows,
        prior=p.JointRootPrior(
            families=(base,),
            function=p.JointFunctionPrior(
                functions=function,
                frequencies=7,
                matrix=p.JointMatrixPrior(families=("gaussian",), vector_family=weight),
            ),
        ),
    )
    return plan, context.freeze()


def materialize(plan, decisions, streams=None, **kwargs):
    streams = streams or [np.random.default_rng(s) for s in (731, 953, 1171)]
    return p.materialize_joint_root(
        plan,
        decisions,
        rng=streams[0],
        calibration_rng=streams[1],
        fit_rng=streams[2],
        calibration_id="independent-calibration",
        **kwargs,
    )


def payload(value):
    return json.loads(json.dumps(asdict(value)))


def verify_pins():
    require(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == HEAD,
        "HEAD pin",
    )
    require(Path(p.__file__).resolve() == ROOT / "src/r0/modules/priors/__init__.py", "import pin")
    pins = json.loads((BUNDLE / "pins.json").read_text())
    for relative, digest in pins["source"].items():
        require(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative)
    for relative, digest in pins["cached"].items():
        require(hashlib.sha256((BUNDLE / relative).read_bytes()).hexdigest() == digest, relative)
    record("source-pins", head=HEAD, import_path=p.__file__, status="PASS")


def matrix_cases():
    roots = {}
    for base in ("uniform", "ball", "gaussian", "covariance"):
        for function in ("lin", "quad", "gp", "tree", "disc", "em", "mlp", "prod"):
            plan, decisions = make_plan(base, function)
            streams = [np.random.default_rng(s) for s in (731, 953, 1171)]
            root = materialize(plan, decisions, streams)
            roots[base, function] = root, decisions
            b = root.calibration.base
            raw, radii, expected = base_oracle(b, np.random.default_rng(953), 5)
            close(root.calibration.draws, raw, "raw calibration")
            close(root.calibration.radial_uniform, radii, "radii calibration")
            close(root.calibration.values, expected, "base transform")
            oracle_fit = p.materialize_joint_function(
                plan.function,
                decisions,
                rng=copy.deepcopy(np.random.default_rng(731))
                if base != "covariance"
                else covariance_raw_state(plan, root),
                fit_rng=np.random.default_rng(1171),
                calibration=np.asarray(root.calibration.values),
                calibration_id=root.calibration_id,
            )
            require(oracle_fit == root.function, "root must bind identical standalone function fit")
            frozen = pickle.dumps(root)
            emission_rng = np.random.default_rng(1979)
            emitted = p.sample_joint_root_values(
                root, decisions, rows=11, rng=emission_rng, sample_id="query"
            )
            oracle_rng = np.random.default_rng(1979)
            raw, radii, x = base_oracle(b, oracle_rng, 11)
            close(emitted.sample.draws, raw, "raw emission")
            close(emitted.sample.radial_uniform, radii, "emission radii")
            close(emitted.sample.values, x, "emission transform")
            require(
                state([oracle_rng]) == state([emission_rng]), "exact emission stream consumption"
            )
            if function != "mlp":
                close(emitted.outputs, evaluate(root.function.function, x), "independent equation")
            chunks = np.concatenate(
                [
                    p.evaluate_frozen_function(
                        root.function.function, torch.tensor(part, dtype=torch.float64)
                    ).numpy()
                    for part in (x[:1], x[1:6], x[6:])
                ]
            )
            close(emitted.outputs, chunks, "fixed function chunk invariance")
            changed = p.sample_joint_root_values(
                root, decisions, rows=3, rng=np.random.default_rng(1980), sample_id="changed-query"
            )
            require(changed.sample.draws != emitted.sample.draws[:3], "changed emission draws")
            require(pickle.dumps(root) == frozen, "no refit")
            require(
                p.joint_root_realization_from_dict(payload(root), decisions=decisions) == root,
                "root roundtrip",
            )
            require(
                p.joint_root_emission_from_dict(payload(emitted), decisions=decisions) == emitted,
                "emission roundtrip",
            )
            record(
                "base-function",
                base=base,
                function=function,
                independent_equation=function != "mlp",
                status="PASS",
            )
    return roots


def covariance_raw_state(plan, root):
    rng = np.random.default_rng(731)
    gaussian = rng.normal(size=(plan.width, plan.width))
    weights = rng.random(plan.width)
    close(root.calibration.base.gaussian_factor, gaussian, "covariance first raw draw")
    close(root.calibration.base.weights.draws, weights, "covariance weights follow Gaussian")
    return rng


def corruptions(roots):
    root, decisions = roots["ball", "tree"]
    original = payload(root)
    cases = [
        ("boolean-width", lambda d: d["plan"].__setitem__("width", True)),
        ("bool-radius", lambda d: d["calibration"]["radial_uniform"].__setitem__(0, True)),
        ("bool-draw", lambda d: d["calibration"]["draws"][0].__setitem__(0, True)),
        ("stale-value", lambda d: d["calibration"]["values"][0].__setitem__(0, 99.0)),
        ("stale-output", lambda d: d["outputs"][0].__setitem__(0, 99.0)),
        ("missing-calibration", lambda d: d.pop("calibration")),
        ("unknown-field", lambda d: d.__setitem__("unowned", 1)),
        ("wrong-version", lambda d: d.__setitem__("version", "other")),
        ("wrong-fit-identity", lambda d: d.__setitem__("calibration_id", "different")),
        ("ragged", lambda d: d["calibration"]["draws"][0].append(1.0)),
        ("nonfinite", lambda d: d["calibration"]["draws"][0].__setitem__(0, float("inf"))),
    ]
    for name, change in cases:
        data = copy.deepcopy(original)
        change(data)
        rejected(
            name, lambda data=data: p.joint_root_realization_from_dict(data, decisions=decisions)
        )
    emitted = p.sample_joint_root_values(
        root, decisions, rows=4, rng=np.random.default_rng(5), sample_id="decode"
    )
    for name in ("outputs", "sample_id", "sample"):
        data = payload(emitted)
        data.pop(name)
        rejected(
            "emission-missing-" + name,
            lambda data=data: p.joint_root_emission_from_dict(data, decisions=decisions),
        )
    data = payload(emitted)
    data["sample"]["base"]["width"] = True
    rejected(
        "emission-boolean-width",
        lambda data=data: p.joint_root_emission_from_dict(data, decisions=decisions),
    )


def rollback(roots):
    plan, decisions = make_plan("covariance", "tree")
    streams = [np.random.default_rng(s) for s in (731, 953, 1171)]
    before = state(streams)
    module = "r0.modules.priors.lib.joint_root_realization"
    original = p.materialize_joint_function
    reached = []

    def fail_after_fit(*args, **kwargs):
        original(*args, **kwargs)
        reached.append((state([kwargs["rng"]]), state([kwargs["fit_rng"]])))
        raise ValueError("injected post-function-fit failure")

    with patch(module + ".materialize_joint_function", fail_after_fit):
        rejected("post-fit-rollback", lambda streams=streams: materialize(plan, decisions, streams))
    require(reached and state(streams) == before, "all three caller streams must roll back")
    require(
        reached[0][0] != state([np.random.default_rng(731)])
        and reached[0][1] != state([np.random.default_rng(1171)]),
        "failure must follow raw and fit draws",
    )
    require(
        materialize(plan, decisions, streams) == materialize(plan, decisions),
        "recovery exact replay",
    )
    for pair in ((0, 1), (0, 2), (1, 2)):
        streams = [np.random.default_rng(s) for s in (731, 953, 1171)]
        streams[pair[1]] = np.random.Generator(streams[pair[0]].bit_generator)
        before = state(streams)
        rejected(
            "aliased-" + str(pair), lambda streams=streams: materialize(plan, decisions, streams)
        )
        require(state(streams) == before, "alias preflight mutation")
    root, decisions = roots["gaussian", "lin"]
    rng = np.random.default_rng(77)
    before = state([rng])
    emodule = "r0.modules.priors.lib.joint_root_emission"
    visited = []

    def postdraw(*args):
        visited.append(True)
        raise ValueError("injected post-draw evaluation failure")

    with patch(emodule + ".evaluate_admitted_function", postdraw):
        rejected(
            "emission-postdraw",
            lambda: p.sample_joint_root_values(
                root, decisions, rows=7, rng=rng, sample_id="failure"
            ),
        )
    require(visited and state([rng]) == before, "emission rollback")
    recovered = p.sample_joint_root_values(root, decisions, rows=7, rng=rng, sample_id="recovered")
    fresh = p.sample_joint_root_values(
        root, decisions, rows=7, rng=np.random.default_rng(77), sample_id="recovered"
    )
    require(recovered == fresh, "emission recovery")
    record("rollback-recovery", status="PASS")


def main():
    torch.set_num_threads(1)
    verify_pins()
    roots = matrix_cases()
    corruptions(roots)
    rollback(roots)
    from adversarial import additional

    additional(roots, record, rejected, make_plan, materialize, state, payload)
    record("component", status="PASS")
    print(OUT)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        record("execution", status="REJECT_OR_HARNESS_FAILURE", error=repr(error))
        print(OUT)
        raise
