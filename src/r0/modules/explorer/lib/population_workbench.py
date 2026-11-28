"""Dispatch explicit finite and row-owned population workbench actions."""

from typing import TYPE_CHECKING, Any

from r0.modules.explorer.lib.gp_root_workbench import gp_root_action
from r0.modules.explorer.lib.population_dag_workbench import population_dag_action

if TYPE_CHECKING:
    from r0.modules.explorer.lib.session import ExplorerSession


def population_action(
    session: "ExplorerSession",
    action: str,
    data: dict[str, Any],
) -> dict[str, Any] | None:
    result = gp_root_action(session, action, data)
    return result if result is not None else population_dag_action(session, action, data)
