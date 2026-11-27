"""Audit named cross-selector sharing and the complete public profile surface."""

import argparse
import hashlib
import json
import pathlib
import sys
import traceback
from dataclasses import asdict

from r0.modules import priors as p

parser = argparse.ArgumentParser()
parser.add_argument("outputdir")
args = parser.parse_args()
out = pathlib.Path(args.outputdir)
out.mkdir(parents=True, exist_ok=True)


def save(name, value):
    (out / name).write_text(json.dumps(value, indent=2, allow_nan=False))


save(
    "execution.json",
    {
        "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "python": sys.executable,
        "imports": {
            str(pathlib.Path(m.__file__).resolve()): hashlib.sha256(
                pathlib.Path(m.__file__).read_bytes()
            ).hexdigest()
            for name, m in list(sys.modules.items())
            if name.startswith("r0.") and getattr(m, "__file__", None)
        },
    },
)
results = []
fixed_overlap = 0
power_overlap = 0
controls = p.LinearRecipe(rows=16, context_rows=8)
try:
    for selector in ("standard", "plain"):
        for suffix in ("", "-refreshed", "-local", "-global"):
            for family in ("forward", "layered", "redirection"):
                name = f"causal-source-{selector}{suffix}-matrices-v1-{family}"
                rec = p.sample_recipe(
                    family,
                    task=controls,
                    profile=f"causal-source-{selector}{suffix}-matrices-v1",
                    seed=28,
                    task_id=10,
                )
                save(name + ".json", asdict(rec))
                assert isinstance(rec, p.SourceMatrixCausalRecipe)
                assert p.recipe_from_dict(json.loads(json.dumps(asdict(rec)))) == rec
                results.append({"case": name, "status": "PASS"})
    base = p.sample_causal_recipe(
        p.CausalGraphPrior(family="forward", nodes=4, edge_probability=1.0),
        task=p.LinearRecipe(
            rows=16, context_rows=8, features=2, signal_features=2, active_features=2
        ),
        seed=28,
        task_id=0,
        widths=(2, 1, 3, 1),
    )
    for seed in range(16):
        name = f"global-standard-joint-{seed}"
        rec = p.with_source_matrices(
            base,
            prior=p.SourceActivationPrior(mode="global", placement="both", calibration_rows=32),
            matrix_prior=p.CausalMatrixPrior(
                mode="global", families=("transformed",), simple_families=("gaussian",)
            ),
            seed=seed,
            task_id=0,
        )
        save(name + ".json", asdict(rec))
        selected = rec.source_selections
        matrix = selected.selections[:3]
        stages = selected.selections[3:]
        assert all(s.selector == "plain" for s in matrix)
        assert all(s.selector == "standard" for s in stages)
        assert len({s.activation for s in matrix}) == 1
        assert len({s.activation for s in stages}) == 1
        fixed = [s for s in selected.selections if s.primitive_draw is not None]
        powers = [s for s in selected.selections if s.exponent_draw is not None]
        assert len({s.activation for s in fixed}) <= 1
        if {s.selector for s in fixed} == {"plain", "standard"}:
            fixed_overlap += 1
        by_kind = {}
        for s in powers:
            by_kind.setdefault(s.activation.kind, []).append(s)
        for group in by_kind.values():
            assert len({s.activation for s in group}) == 1
            if {s.selector for s in group} == {"plain", "standard"}:
                power_overlap += 1
        names = [v.request.name for v in selected.categorical.preferences]
        assert len(names) == len(set(names))
        assert "act_type_plain" in names
        assert "act_type" in names
        if fixed:
            assert names.count("acts_torch") == 1
        results.append({"case": name, "status": "PASS"})
        save("progress.json", results)
    assert fixed_overlap > 0
    save(
        "summary.json",
        {
            "status": "PASS",
            "checks": results,
            "fixed_overlap_cases": fixed_overlap,
            "power_overlap_cases": power_overlap,
        },
    )
except Exception:
    save(
        "summary.json",
        {
            "status": "STOPPED",
            "completed": results,
            "case": name,
            "traceback": traceback.format_exc(),
        },
    )
    raise
