"""The public population-reference worker shares strict saved-record metadata rules."""

import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from typing import Any

import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession
from r0.modules.generation.worker import main


@pytest.fixture(scope="module")
def worker_record(tmp_path_factory: pytest.TempPathFactory):
    examples = ExplorerSession(tmp_path_factory.mktemp("worker-record")).request(
        "construction_examples", {}
    )["examples"]
    controls = next(
        v for k, v in examples.items() if k.startswith("Population DAG: functional linear")
    )
    source = g.GraphConstruction(json.dumps(controls), 17).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    assert isinstance(source, p.RecursivePopulationGraph)
    task = p.generate_task(source, seed=31, task_id=2**55 + 7)
    request = dict(
        validation="population-curve",
        record=json.loads(task.recipe_json),
        row=7,
        feature=1,
        values=[-2.0, 2.0],
        capture_location="capture",
        causal_limits=asdict(p.CausalLimits()),
    )
    return request, type(source.bindings[0].sample.initial)


def invoke(request: dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(dict(request=request, out="unused"))))
    output = io.StringIO()
    with redirect_stdout(output):
        assert main() == 0
    return json.loads(output.getvalue())


def test_empty_saved_python_rejects_before_restore(
    worker_record: Any, monkeypatch: pytest.MonkeyPatch
):
    request, snapshot = worker_record
    request = json.loads(json.dumps(request))
    request["record"]["execution"]["python"] = ""

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("malformed saved execution metadata reached source replay")

    monkeypatch.setattr(snapshot, "restore", forbidden)
    with pytest.raises(ValueError, match="execution|metadata"):
        invoke(request, monkeypatch)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("numpy", ""),
        ("torch", ""),
        ("python", None),
        ("numpy", 3),
        ("cpu_threads", 0),
        ("cpu_threads", -1),
        ("cpu_threads", True),
        ("cpu_threads", 1.0),
        ("cpu_threads", 2**63),
        ("cuda", ""),
        ("gpu", ""),
        ("gpu", False),
        ("implementation", "wrong"),
        ("implementation", ""),
        ("device", "cuda"),
        ("extra", "unused"),
        ("missing", None),
        ("execution", []),
        ("seed", True),
        ("seed", 32),
        ("task_id", 1.0),
        ("task_id", 2**63),
        ("count", 1),
    ],
)
def test_complete_record_schema_precedes_worker_restore(
    worker_record: Any, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
):
    request, snapshot = worker_record
    request = json.loads(json.dumps(request))
    record = request["record"]
    if field == "missing":
        del record["execution"]["torch"]
    elif field in ("execution", "seed", "task_id", "count"):
        record[field] = value
    else:
        record["execution"][field] = value

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("invalid complete saved record reached restoration")

    monkeypatch.setattr(snapshot, "restore", forbidden)
    with pytest.raises(ValueError, match="execution|metadata|identity|integer|exactly"):
        invoke(request, monkeypatch)


def test_valid_historical_execution_preserves_complete_reference(
    worker_record: Any, monkeypatch: pytest.MonkeyPatch
):
    request, _ = worker_record
    baseline = invoke(request, monkeypatch)
    historical = json.loads(json.dumps(request))
    historical["record"]["execution"].update(
        python="historical-python",
        numpy="historical-numpy",
        torch="historical-torch",
        cpu_threads=2**63 - 1,
        cuda="historical-cuda",
        gpu="historical-device",
    )
    assert invoke(historical, monkeypatch) == baseline
    historical["record"]["execution"]["cpu_threads"] = 1
    assert invoke(historical, monkeypatch) == baseline
