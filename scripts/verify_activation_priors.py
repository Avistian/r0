"""Independent activation-prior oracles, replay probes, and compatibility snapshots.

Run with an existing r0 development environment; no network or added dependencies.
Output defaults to a fresh temporary directory. Each run preserves its own evidence.
"""

import argparse
import copy
import hashlib
import importlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np


def verify(priors, root, head, base, repo):
    ROOT, HEAD, BASE = root, head, base
    TASK = priors.LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    GRAPH = priors.CausalGraphPrior(nodes=4, edge_probability=0)
    RESULTS = []

    def record(name, passed, details):
        RESULTS.append({"name": name, "pass": bool(passed), "details": details})
        print(json.dumps(RESULTS[-1]), flush=True)

    def sample(prior, task_id=0, graph=GRAPH, task=TASK):
        return priors.sample_causal_recipe(
            graph, task=task, seed=713431, task_id=task_id, activations=prior
        )

    def rejected(name, operation):
        try:
            operation()
        except (ValueError, TypeError, OverflowError) as error:
            record(name, True, {"error_type": type(error).__name__, "message": str(error)})
        else:
            record(name, False, "Accepted an invalid or unrepresentable record")

    def snapshot():
        rows = []
        for profile in ("causal-affine-v1", "causal-activated-v1"):
            for family in ("forward", "layered", "redirection"):
                for task_id in (0, 1, 17):
                    recipe = priors.sample_recipe(
                        family, task=TASK, seed=84, task_id=task_id, profile=profile
                    )
                    task = priors.generate_task(recipe, seed=84, task_id=task_id)
                    value = json.dumps(
                        {
                            "recipe": asdict(recipe),
                            "card": json.loads(task.mechanism_json),
                            "inputs": task.inputs.tolist(),
                            "targets": task.targets.tolist(),
                        },
                        sort_keys=True,
                    )
                    rows.append(hashlib.sha256(value.encode()).hexdigest())
        return rows

    def run_checks():
        choices = (
            priors.CausalActivation("tanh"),
            priors.CausalActivation("sin"),
            priors.CausalActivation("relu"),
        )
        n = 1800
        for mode in ("independent", "dirichlet", "homogeneous"):
            prior = priors.CausalActivationPrior(
                choices=choices,
                weights=(1, 2, 3),
                mode=mode,
                concentration=5 if mode == "dirichlet" else 1,
            )
            ps, labels, agreements = [], [], []
            for task_id in range(n):
                recipe = sample(prior, task_id)
                ps.append(recipe.activation_probabilities)
                kinds = [node.output_activation.kind for node in recipe.nodes]
                labels.append(kinds[0] == "tanh")
                agreements.append(kinds[0] == kinds[1])
                if mode == "homogeneous":
                    assert len(set(kinds)) == 1
            p = np.asarray(ps)
            w = np.array([1 / 6, 1 / 3, 1 / 2])
            second = (
                (5 * w * w + w) / 6
                if mode == "dirichlet"
                else (w if mode == "homogeneous" else w * w)
            )
            agreement = second.sum()
            actual = {
                "means": p.mean(0).tolist(),
                "second": (p * p).mean(0).tolist(),
                "first_choice_frequency": float(np.mean(labels)),
                "agreement": float(np.mean(agreements)),
                "tasks": n,
            }
            # Distribution-free Hoeffding bounds for bounded independent task-level observations.
            tolerance = math.sqrt(math.log(2 * 30 / 1e-6) / (2 * n))
            passed = (
                np.max(abs(p.mean(0) - w)) < tolerance
                and np.max(abs((p * p).mean(0) - second)) < tolerance
                and abs(np.mean(labels) - w[0]) < tolerance
                and abs(np.mean(agreements) - agreement) < tolerance
            )
            record(
                mode + "_law",
                passed,
                {
                    "observed": actual,
                    "expected_second": second.tolist(),
                    "expected_agreement": float(agreement),
                    "tolerance": tolerance,
                },
            )

        # Parameter laws use the actual sampled node descriptors; no sampler-internal helpers.
        parameters = (
            priors.ActivationParameterPrior("input_shift", -4, 8),
            priors.ActivationParameterPrior(
                "output_scale", math.exp(-2), math.exp(4), "log_uniform"
            ),
            priors.ActivationParameterPrior("exponent", -3, 4, "randint"),
        )
        prior = priors.CausalActivationPrior(
            choices=(priors.CausalActivation("integer_power"),),
            parameter_priors=(parameters,),
            placement="both",
            noise_after_probability=0,
        )
        draws = []
        for task_id in range(1000):
            recipe = sample(prior, task_id)
            draws.append(dict(recipe.nodes[0].output_activation.parameters))
        observed = [
            np.mean([v["input_shift"] for v in draws]),
            np.mean([math.log(v["output_scale"]) for v in draws]),
            np.mean([v["exponent"] for v in draws]),
        ]
        record(
            "numeric_parameter_laws",
            all(abs(a - b) < 0.4 for a, b in zip(observed, [2, 1, 0.5], strict=True)),
            {"means": observed, "expected": [2, 1, 0.5]},
        )
        record(
            "integer_type_and_inclusive_support",
            all(type(v["exponent"]) is int for v in draws)
            and {v["exponent"] for v in draws} == set(range(-3, 5)),
            "inclusive -3 through 4",
        )
        recipe = sample(prior, graph=replace(GRAPH, edge_probability=1))
        wire = json.loads(json.dumps(asdict(recipe)))
        record("json_roundtrip", priors.recipe_from_dict(wire) == recipe, "all frozen prior fields")
        longer = sample(
            prior, task=replace(TASK, rows=32), graph=replace(GRAPH, edge_probability=1)
        )
        record(
            "row_count_invariance",
            longer.nodes == recipe.nodes
            and longer.topology == recipe.topology
            and longer.observed == recipe.observed
            and longer.target == recipe.target,
            "16 versus 32 rows",
        )
        for mode in ("independent", "dirichlet", "homogeneous"):
            for placement in ("input", "output", "both"):
                r = sample(
                    replace(prior, mode=mode, placement=placement),
                    graph=replace(GRAPH, edge_probability=1),
                )
                active = []
                for node in r.nodes:
                    if node.parents and placement in ("input", "both"):
                        active.append(node.input_activation)
                    else:
                        assert node.input_activation == priors.CausalActivation()
                    if placement in ("output", "both"):
                        active.append(node.output_activation)
                    else:
                        assert node.output_activation == priors.CausalActivation()
                assert mode != "homogeneous" or len(set(active)) == 1
        record("placement_homogeneous_parameters", True, "all 9 mode/placement combinations")

        for name in ("outside_range", "fractional_integer", "inactive_stage", "noise_order"):
            bad = copy.deepcopy(wire)
            if name == "outside_range":
                bad["nodes"][0]["output_activation"]["parameters"][0][1] = -5
            elif name == "fractional_integer":
                bad["nodes"][0]["output_activation"]["parameters"][2][1] = 1.5
            elif name == "inactive_stage":
                bad["nodes"][0]["input_activation"] = {"kind": "tanh", "parameters": []}
            else:
                bad["nodes"][0]["noise_position"] = "after_output"
            rejected("decoder_" + name, lambda bad=bad: priors.recipe_from_dict(bad))

        fair = priors.CausalActivationPrior(choices=choices[:2], weights=(1, 1))
        rejected(
            "positive_weight_underflow", lambda: sample(replace(fair, weights=(1e-308, 1e308)))
        )
        huge = sample(replace(fair, weights=(1e308, 1e308)))
        record(
            "finite_huge_weights",
            huge.activation_probabilities == (0.5, 0.5),
            "scaled normalization",
        )
        rejected(
            "tiny_concentration",
            lambda: sample(replace(fair, mode="dirichlet", concentration=5e-324)),
        )
        p = priors.ActivationParameterPrior("exponent", -(2**63), 2**63 - 1, "randint")
        extreme = sample(
            priors.CausalActivationPrior(
                choices=(priors.CausalActivation("integer_power"),), parameter_priors=((p,),)
            )
        )
        record(
            "full_int64_support_sample",
            all(
                type(dict(node.output_activation.parameters)["exponent"]) is int
                for node in extreme.nodes
            ),
            "retained Python integers",
        )

        # Public generator numerical oracle: frozen identity makes both noise positions equivalent.
        before = sample(priors.CausalActivationPrior(noise_after_probability=0))
        after = sample(priors.CausalActivationPrior(noise_after_probability=1))
        left = priors.generate_task(before, seed=51, task_id=9)
        right = priors.generate_task(after, seed=51, task_id=9)
        record(
            "identity_noise_order",
            left.inputs.equal(right.inputs) and left.scores.equal(right.scores),
            "independently keyed innovations survive prior/order changes",
        )
        for mode in ("independent", "dirichlet", "homogeneous"):
            r = sample(replace(fair, mode=mode))
            task = priors.generate_task(r, seed=51, task_id=9)
            restored = priors.recipe_from_dict(json.loads(json.dumps(asdict(r))))
            replay = priors.generate_task(restored, seed=51, task_id=9)
            record(
                "replay_trace_" + mode,
                task.episode() == replay.episode()
                and priors.explain_generation(task, [0, 15])["target_max_error"] < 1e-12,
                "saved descriptors, original selected rows",
            )

        # Frozen parameter provenance rejects changed numeric descriptors even in homogeneous mode.
        homogeneous = sample(replace(fair, mode="homogeneous"))
        bad = json.loads(json.dumps(asdict(homogeneous)))
        selected = homogeneous.activation_probabilities.index(1)
        bad["activation_probabilities"][selected] = 1 - 5e-13
        rejected("homogeneous_exact_one_hot", lambda: priors.recipe_from_dict(bad))

        archive_root = ROOT / "base-source"
        archive_root.mkdir()
        archive = subprocess.check_output(["git", "-C", str(repo), "archive", BASE, "src"])
        with tarfile.open(fileobj=io.BytesIO(archive)) as saved:
            saved.extractall(archive_root, filter="data")
        old = subprocess.check_output(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--repo",
                str(archive_root),
                "--snapshot-only",
            ],
            text=True,
        )
        record(
            "old_profiles_preserved",
            json.loads(old) == snapshot(),
            "18 affine/activated recipe + mechanism + generated tensor SHA256 comparisons",
        )
        output = {
            "head": HEAD,
            "source_root": str(repo / "src"),
            "base": BASE,
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "results": RESULTS,
        }
        (ROOT / "results.json").write_text(json.dumps(output, indent=2) + "\n")
        return RESULTS

    return run_checks()


def old_snapshot(priors):
    TASK = priors.LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    rows = []
    for profile in ("causal-affine-v1", "causal-activated-v1"):
        for family in ("forward", "layered", "redirection"):
            for task_id in (0, 1, 17):
                recipe = priors.sample_recipe(
                    family, task=TASK, seed=84, task_id=task_id, profile=profile
                )
                task = priors.generate_task(recipe, seed=84, task_id=task_id)
                value = json.dumps(
                    {
                        "recipe": asdict(recipe),
                        "card": json.loads(task.mechanism_json),
                        "inputs": task.inputs.tolist(),
                        "targets": task.targets.tolist(),
                    },
                    sort_keys=True,
                )
                rows.append(hashlib.sha256(value.encode()).hexdigest())
    return rows


def workflow(root):
    """Exercise public saved explorer workflows; this does not claim browser execution."""
    explorer = importlib.import_module("r0.modules.explorer")
    results = []
    for family in ("forward", "layered", "redirection"):
        for task_id in (3, 10, 21):
            key = f"{family}-{task_id}"
            session = explorer.ExplorerSession(root / "archives" / key)
            generated = session.request(
                "generate",
                {
                    "record": {
                        "recipe": {"rows": 16, "context_rows": 8},
                        "seed": 28,
                        "task_id": task_id,
                    },
                    "sampling": {"family": family, "profile": "causal-preferences-v1"},
                },
            )
            recipe = json.loads(generated["recipe_json"])
            reopened = explorer.ExplorerSession(root / "archives" / key).request(
                "open",
                {
                    "archive": generated["archive"],
                    "task_id": task_id,
                },
            )
            replay_session = explorer.ExplorerSession(root / "replays" / key)
            replay = replay_session.request("generate", {"record_json": generated["recipe_json"]})
            trace = session.request("graph", {"indices": [0, 15]})["graph"]
            result = {
                "key": key,
                "mode": recipe["recipe"]["activation_prior"]["mode"],
                "reopened_view": reopened["view"] == generated["view"],
                "replayed_recipe": replay["recipe_json"] == generated["recipe_json"],
                "replayed_row": replay_session.request("row", {"index": 15})
                == session.request("row", {"index": 15}),
                "trace_error": trace["target_max_error"],
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    (root / "workflow-results.json").write_text(json.dumps(results, indent=2) + "\n")
    return all(
        all(r[k] for k in ("reopened_view", "replayed_recipe", "replayed_row"))
        and r["trace_error"] < 1e-10
        for r in results
    )


def command(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def imported_source(repo):
    """Pin imports to the selected checkout rather than any editable installation."""
    source = (repo / "src").resolve()
    sys.path.insert(0, str(source))
    os.environ["PYTHONPATH"] = str(source)
    priors = importlib.import_module("r0.modules.priors")
    paths = {}
    for name, module in sys.modules.copy().items():
        if name == "r0" or name.startswith("r0."):
            filename = getattr(module, "__file__", None)
            if filename is not None:
                path = Path(filename).resolve()
                if not path.is_relative_to(source):
                    raise RuntimeError(f"Imported {name} outside selected source: {path}")
                paths[name] = str(path.relative_to(source))
    return priors, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base", help="Fixed base for exact affine/activated compatibility checks")
    parser.add_argument("--expected-head", help="Fail if the selected checkout differs")
    parser.add_argument("--output", type=Path, help="New evidence directory; default is temporary")
    parser.add_argument("--snapshot-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo = args.repo.resolve()
    priors, imports = imported_source(repo)
    if args.snapshot_only:
        print(json.dumps(old_snapshot(priors)))
        return
    if not args.base:
        parser.error("--base is required")
    head = command(repo, "rev-parse", "HEAD")
    base = command(repo, "rev-parse", args.base + "^{commit}")
    if args.expected_head and head != command(repo, "rev-parse", args.expected_head + "^{commit}"):
        raise RuntimeError("Selected repository is not at --expected-head")
    subprocess.run(
        ["git", "-C", str(repo), "diff", "--exit-code", "HEAD", "--", "src"],
        check=True,
        capture_output=True,
    )
    root = (
        args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="r0-prior-audit-"))
    )
    if args.output:
        root.mkdir(parents=True, exist_ok=False)
    receipt = {
        "head": head,
        "base": base,
        "repo": str(repo),
        "output": str(root),
        "source_tree": command(repo, "rev-parse", "HEAD:src"),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "executable": sys.executable,
        "argv": sys.argv,
        "imported_sources": imports,
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"head": head, "base": base, "output": str(root)}), flush=True)
    results = verify(priors, root, head, base, repo)
    workflow_pass = workflow(root)
    # Recheck frozen source identity after execution; docs may change independently.
    if command(repo, "rev-parse", "HEAD:src") != receipt["source_tree"]:
        raise RuntimeError("Source tree changed during verification")
    subprocess.run(
        ["git", "-C", str(repo), "diff", "--exit-code", "HEAD", "--", "src"],
        check=True,
        capture_output=True,
    )
    imported_source(repo)
    passed = all(result["pass"] for result in results) and workflow_pass
    summary = {
        "pass": passed,
        "standalone_checks": len(results),
        "standalone_passed": sum(result["pass"] for result in results),
        "workflow_pass": workflow_pass,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    sys.exit(not passed)


if __name__ == "__main__":
    main()
