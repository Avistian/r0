"""Notebook transport uses the same session actions and charts as the local UI."""

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from r0.modules.explorer import show_colab_explorer


@pytest.mark.parametrize(
    "example",
    [
        "Native mixed graph (authored)",
        "Numerical pipeline: readout-only",
        "Explicit causal graph",
        "Pipeline innovations: uniform (r0 extension)",
        "Attachment geometry (r0 extension)",
        "Motif geometry (r0 adaptation)",
        "Typed categorical converter pipelines",
    ],
)
def test_notebook_registers_callback_and_returns_structured_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, example: str
) -> None:
    callbacks: dict[str, Any] = {}
    displayed: list[str] = []
    output = ModuleType("google.colab.output")

    def register(name: str, callback: Any) -> None:
        callbacks[name] = callback

    monkeypatch.setattr(output, "register_callback", register, raising=False)
    display = ModuleType("IPython.display")

    def wrap_json(value: Any) -> dict[str, Any]:
        return {"application/json": value}

    def wrap_html(value: str) -> str:
        return value

    monkeypatch.setattr(display, "JSON", wrap_json, raising=False)
    monkeypatch.setattr(display, "HTML", wrap_html, raising=False)
    monkeypatch.setattr(display, "display", displayed.append, raising=False)
    monkeypatch.setitem(sys.modules, "google.colab.output", output)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    name = show_colab_explorer(tmp_path)
    assert name in callbacks
    assert len(displayed) == 1
    assert name in displayed[0]
    assert "google.colab.kernel.invokeFunction" in displayed[0]
    assert 'id="feature-hist"' in displayed[0]
    assert "window.r0token =" not in displayed[0]
    response = callbacks[name]("examples", {})["application/json"]
    assert len(response["examples"]) == 10
    assert response["output_dir"] == str(tmp_path)
    failure = callbacks[name]("row", {"index": 0})["application/json"]
    assert failure == {"error": "generate or open a dataset first"}

    assert 'id="construction-panel"' in displayed[0]
    assert 'id="workbench-panel"' in displayed[0]
    controls = callbacks[name]("construction_examples", {})["application/json"]["examples"][example]
    if controls["topology"] == "sampled-converted":
        controls["request"].update(rows=16, context_rows=8, calibration_rows=8)
    else:
        controls["task"].update(rows=16, context_rows=8)
    if controls["topology"] not in ("native", "sampled-converted"):
        controls["calibration_rows"] = 4
    if "width_prior" in controls:
        controls["width_prior"].update(low=1, high=3)
    created = callbacks[name](
        "construct", {"controls": controls, "construction_seed": 28, "seed": 19, "task_id": 0}
    )["application/json"]
    assert "error" not in created
    editor = callbacks[name]("mechanism", {})["application/json"]
    node = editor["editable_recipe"]["nodes"][0]
    equation = node.get("equation", node)
    equation["noise_scale"] = [0.1] * editor["nodes"][0]["width"]
    changed = callbacks[name]("edit", {"recipe": editor["editable_recipe"]})["application/json"]
    assert changed["edit"]["card"]["kind"] == "mechanism-edit"
    reopened = callbacks[name]("open_edit", {"path": changed["edit"]["path"]})["application/json"]
    assert reopened["edit"]["card"] == changed["edit"]["card"]
    if example == "Explicit causal graph":
        assert 'id="intervention-controls"' in displayed[0]
        baseline = reopened["recipe_json"]
        pair = callbacks[name](
            "intervene",
            {
                "interventions_json": '[{"version":"hard-causal-intervention-v1",'
                '"node":2,"channels":[0],"values":[4],'
                '"category":null,"location":"state_and_readout"}]',
                "mode": "matched",
                "population": "1",
            },
        )["application/json"]
        assert "error" not in pair
        assert pair["recipe_json"] == baseline
        assert all(row["target"] == 4 for row in pair["intervention"]["after"]["rows"])
        loaded_pair = callbacks[name]("open_intervention", {"path": pair["intervention"]["path"]})[
            "application/json"
        ]
        assert loaded_pair["intervention"]["card"] == pair["intervention"]["card"]

    if example == "Typed categorical converter pipelines":
        info = callbacks[name]("intervention_controls", {})["application/json"]
        node = next(n for n in info["nodes"] if n["converters"])
        converter = node["converters"][0]
        pair = callbacks[name](
            "intervene",
            {
                "interventions_json": json.dumps(
                    [
                        {
                            "version": "converter-hard-intervention-v1",
                            "node": node["index"],
                            "capture": converter["name"],
                            "category": converter["vocabulary"][0],
                            "location": "capture",
                        }
                    ]
                )
            },
        )["application/json"]
        assert "error" not in pair
        assert pair["recipe_json"] == created["recipe_json"]
        loaded = callbacks[name]("open_intervention", {"path": pair["intervention"]["path"]})[
            "application/json"
        ]
        assert loaded["intervention"]["card"] == pair["intervention"]["card"]
