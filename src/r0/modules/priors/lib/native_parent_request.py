"""Native variable domains with source-specific per-parent structural mechanisms."""

from dataclasses import dataclass

from r0.modules.priors.lib.native_graph_request import NativeGraphRequest
from r0.modules.priors.lib.parent_composition_plans import ParentCompositionPrior


@dataclass(frozen=True, slots=True)
class NativeParentGraphRequest(NativeGraphRequest):
    composition: ParentCompositionPrior = ParentCompositionPrior()
    version: str = "native-parent-graph-request-v1"

    def __post_init__(self) -> None:
        NativeGraphRequest.__post_init__(self)
        if type(self.composition) is not ParentCompositionPrior:
            raise ValueError("native parent construction requires its composition prior")
