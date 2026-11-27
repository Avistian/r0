"""Explicit runtime mechanism context; ordinary frozen evaluation remains the default."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import torch

from r0.modules.priors.lib.frozen_function_evaluation import FrozenFunction

if TYPE_CHECKING:
    from r0.modules.priors.lib.functional_analysis import ChannelCertificate


class NodeMechanisms(Protocol):
    """An enclosing population envelope admits and owns every node-specific replacement."""

    def evaluate(
        self,
        index: int,
        function: FrozenFunction,
        inputs: torch.Tensor,
        diagnostics: list[dict[str, Any]] | None = None,
    ) -> torch.Tensor: ...

    def replaces(self, index: int) -> bool: ...

    def steps(
        self,
        index: int,
        *,
        parents: list[str],
        indices: Sequence[int],
        add: Callable[..., None],
    ) -> list[str]: ...

    def metadata(self, index: int) -> dict[str, Any]: ...

    def certify(
        self, index: int, function: FrozenFunction, incoming: list["ChannelCertificate"]
    ) -> list["ChannelCertificate"]: ...

    def record(self, index: int, **values: torch.Tensor) -> None: ...
