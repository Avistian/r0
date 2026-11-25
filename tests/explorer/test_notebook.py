"""Notebook transport uses the same session actions and charts as the local UI."""

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from r0.modules.explorer import show_colab_explorer


def test_notebook_registers_callback_and_returns_structured_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
