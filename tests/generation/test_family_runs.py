"""Configured families reach the guarded process and persist replayable cards."""

import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest

from r0.modules.generation import load_request, run_guarded
from r0.modules.priors import FourierRecipe, LinearRecipe, generate_task, recipe_from_dict


@pytest.mark.parametrize(
    "family_fields",
    [
        'version = "fourier-v1"\nfamily = "rff"\nterms = 4',
        'version = "fourier-v1"\nfamily = "periodic"\nterms = 4',
        'version = "quadratic-v1"\nrank = 4',
        'version = "grouped-linear-v1"\ngroup_size = 2',
        'version = "smooth-v1"\nfamily = "radial"\nprojection = "ridge"\nterms = 4',
        'version = "exact-gp-v1"\ncalibration_rows = 32',
    ],
)
def test_configured_run_saves_exact_recipe_and_resumes(tmp_path: Path, family_fields: str) -> None:
    config = tmp_path / "request.toml"
    config.write_text(f"""
[recipe]
{family_fields}
[recipe.task]
task_type = "binary"
classes = 2
class_offsets = [0.5]
[run]
seed = 27
count = 2
group_size = 2
max_seconds = 30
max_bytes = 1048576
""")
    request = load_request(config)
    first = run_guarded(request, tmp_path / "run", workers=2)
    assert first.returncode == 0, first.stderr
    archive = next((tmp_path / "run").glob("groups/*/*/data-*.zip"))
    with zipfile.ZipFile(archive) as group:
        card = json.loads(group.read("0/card.json"))
    assert card["recipe"]["recipe"] == json.loads(json.dumps(asdict(request.recipe)))
    replay = generate_task(recipe_from_dict(card["recipe"]["recipe"]), seed=27, task_id=0)
    assert json.loads(replay.mechanism_json) == card["mechanism"]
    resumed = run_guarded(request, tmp_path / "run", workers=1)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["generated"] == 0
    assert resumed.summary["completed"] == 2


def test_decoder_preserves_input_and_rejects_unknown_versions() -> None:
    original = json.loads(json.dumps(asdict(FourierRecipe(task=LinearRecipe()))))
    snapshot = json.dumps(original, sort_keys=True)
    recipe_from_dict(original)
    assert json.dumps(original, sort_keys=True) == snapshot
    with pytest.raises(ValueError, match="version"):
        recipe_from_dict({"version": "unknown"})
