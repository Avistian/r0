"""Versioned soft replacements of complete frozen root, pipeline and emission stages."""

from dataclasses import dataclass

from r0.modules.priors.lib.joint_root_base import JointRootBase
from r0.modules.priors.lib.native_causal import FrequencyVariableDomain, VariableDomain
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline


@dataclass(frozen=True, slots=True)
class StageIntervention:
    node: int
    root: JointRootBase | None = None
    replace_pipeline: bool = False
    pipeline: CausalNodePipeline | None = None
    domain: VariableDomain | None = None
    version: str = "stage-causal-intervention-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "stage-causal-intervention-v1"
            or type(self.node) is not int
            or self.node < 0
            or type(self.replace_pipeline) is not bool
            or self.root is not None
            and type(self.root) is not JointRootBase
            or self.pipeline is not None
            and type(self.pipeline) is not CausalNodePipeline
            or self.pipeline is not None
            and not self.replace_pipeline
            or self.domain is not None
            and type(self.domain) not in (VariableDomain, FrequencyVariableDomain)
            or self.root is None
            and not self.replace_pipeline
            and self.domain is None
        ):
            raise ValueError("stage intervention requires explicit frozen stage replacements")
