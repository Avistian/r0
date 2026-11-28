"""Typed feature specifications and source geometry for sampled converter graphs."""

from dataclasses import dataclass

from r0.modules.priors.lib.categorical_converter_plans import CategoricalConverterPrior
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.geometry_graph_request import GeometryGraphRequest
from r0.modules.priors.lib.geometry_topology import GeometryTopologyPrior
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_pipeline_plans import (
    JointInnovationPipelinePrior,
    JointPipelinePrior,
)
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.joint_width_plans import JointWidthPrior
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.parent_composition_plans import ParentCompositionPrior
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


@dataclass(frozen=True, slots=True)
class ConvertedGraphRequest:
    topology: GeometryTopologyPrior
    vocabularies: tuple[tuple[str, ...], ...]
    target_vocabulary: tuple[str, ...] = ()
    rows: int = 64
    context_rows: int = 32
    split: int = 0
    subsample: bool = True
    width_prior: JointWidthPrior = JointWidthPrior()
    calibration_rows: int = 256
    root_prior: JointRootPrior = JointRootPrior()
    function_prior: JointFunctionPrior = JointFunctionPrior()
    pipeline_prior: JointPipelinePrior = JointPipelinePrior()
    converter_prior: CategoricalConverterPrior = CategoricalConverterPrior()
    composition: ParentCompositionPrior | None = None
    mode: str = "meta"
    categorical_policy: str = "shared"
    global_policy: str = "scalar-quantile-v1"
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "converted-graph-request-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "converted-graph-request-v1"
            or type(self.vocabularies) is not tuple
            or not self.vocabularies
            or type(self.converter_prior) is not CategoricalConverterPrior
            or type(self.pipeline_prior) not in (JointPipelinePrior, JointInnovationPipelinePrior)
            or self.composition is not None
            and type(self.composition) is not ParentCompositionPrior
        ):
            raise ValueError(
                "converted graph requires typed feature specifications and frozen priors"
            )
        for vocabulary in (*self.vocabularies, self.target_vocabulary):
            if (
                type(vocabulary) is not tuple
                or len(vocabulary) == 1
                or any(type(v) is not str or not v for v in vocabulary)
                or len(set(vocabulary)) != len(vocabulary)
            ):
                raise ValueError(
                    "feature vocabulary must be empty continuous or at least two unique labels"
                )
        self.geometry()
        self.task()

    def task(self) -> LinearRecipe:
        cardinality = len(self.target_vocabulary)
        return LinearRecipe(
            rows=self.rows,
            context_rows=self.context_rows,
            features=len(self.vocabularies),
            signal_features=len(self.vocabularies),
            active_features=len(self.vocabularies),
            task_type="regression"
            if cardinality == 0
            else "binary"
            if cardinality == 2
            else "multiclass",
            classes=cardinality,
            split=self.split,
        )

    def geometry(self) -> GeometryGraphRequest:
        # Geometry reserves one logical target, even when its capture has K categories.
        task = LinearRecipe(
            rows=self.rows,
            context_rows=self.context_rows,
            features=len(self.vocabularies),
            signal_features=len(self.vocabularies),
            active_features=len(self.vocabularies),
            split=self.split,
        )
        return GeometryGraphRequest(
            self.topology,
            task,
            self.subsample,
            self.width_prior,
            self.calibration_rows,
            self.root_prior,
            self.function_prior,
            0.0,
            self.mode,
            self.categorical_policy,
            self.global_policy,
            self.limits,
            self.raw_limits,
        )
