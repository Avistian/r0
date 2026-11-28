"""CLI-compatible observation controls bind generated and resumed task archives."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.generation.lib.requests import record_matches


@pytest.mark.parametrize(
    "mode",
    [
        "mcar",
        "mar",
        "mnar-self",
        "mnar-target",
        "mnar-latent",
        "row",
        "block",
        "column",
        "rectangle",
        "monotone",
    ],
)
def test_guarded_missingness_json_and_resume(mode: str, tmp_path: Path) -> None:
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    plan = p.MissingnessPlan(
        mode,
        (0,),
        probability=0.4
        if mode in ("mcar", "row", "block", "column", "rectangle", "monotone")
        else None,
        drivers=(1,) if mode == "mar" else (),
        coefficients=(0.5,) if mode == "mar" else (),
        block_rows=3 if mode in ("block", "rectangle") else 1,
    )
    observation = g.ObservationRequest(
        plan,
        seed=42,
        latent=tuple(float(i) / 16 for i in range(16)) if mode == "mnar-latent" else (),
        latent_provenance="authored fixed row signal" if mode == "mnar-latent" else None,
    )
    request = g.RunRequest(
        task, seed=30, count=1, max_seconds=30, max_bytes=8000000, observation=observation
    )
    decoded = g.request_from_dict(json.loads(json.dumps(asdict(request))))
    assert decoded == request
    run = g.run_guarded(decoded, tmp_path)
    assert run.returncode == 0, run.stderr
    assert run.summary is not None
    assert run.summary["completed"] == 1
    saved = [t for path in tmp_path.rglob("data-*.zip") for t in g.load_group(path)]
    assert len(saved) == 1
    raw = json.loads(saved[0].recipe_json)
    assert record_matches(raw, request, 0)
    assert not record_matches(raw, replace(request, observation=None), 0)
    changed = replace(request, observation=replace(observation, seed=43))
    assert not record_matches(raw, changed, 0)
    with pytest.raises(ValueError, match="configuration changed"):
        g.run_generation(changed, tmp_path)
    resumed = g.run_guarded(request, tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["generated"] == 0


def test_latent_driver_bound_to_request() -> None:
    from r0.modules.generation.lib.requests import generate_requested

    task = p.LinearRecipe(rows=16, context_rows=8)
    observation = g.ObservationRequest(
        p.MissingnessPlan("mnar-latent", (0,)),
        latent=(0.0,) * 16,
        latent_provenance="authored zero signal",
    )
    request = g.RunRequest(
        task, seed=30, count=1, max_seconds=30, max_bytes=8000000, observation=observation
    )
    generated = generate_requested(request, 0)
    altered = replace(observation, latent=(1.0,) * 16)
    with pytest.raises(ValueError, match="latent driver"):
        altered.validate(generated, p.CausalLimits())
