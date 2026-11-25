"""Embed the existing explorer in a trusted, executed Colab output cell."""

import importlib
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from r0.modules.explorer.lib.page import explorer_page
from r0.modules.explorer.lib.session import ExplorerSession


def show_colab_explorer(output_dir: Path) -> str:
    """Display notebook controls; return the callback name for explicit cleanup.

    Colab and IPython are provided by the notebook runtime, not project dependencies.
    Use google.colab.output.unregister_callback(name) before replacing this output
    to release its session and make the old controls inactive.
    """
    try:
        output = importlib.import_module("google.colab.output")
        display = importlib.import_module("IPython.display")
    except ModuleNotFoundError as error:
        raise RuntimeError("Run show_colab_explorer inside a Google Colab notebook") from error
    session = ExplorerSession(output_dir)
    actions = Lock()

    def invoke(action: str, data: dict[str, Any]) -> Any:
        try:
            with actions:
                result = session.request(action, data)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
            result = {"error": str(error)}
        return display.JSON(result)

    callback_name = "r0.explorer." + uuid.uuid4().hex
    output.register_callback(callback_name, invoke)
    try:
        display.display(display.HTML(explorer_page(callback_name=callback_name)))
    except Exception:
        output.unregister_callback(callback_name)
        raise
    return callback_name
