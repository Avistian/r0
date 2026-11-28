"""Complete native construction archives with four-stream replay custody."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_operation_registry import JointOperationRegistry
from r0.modules.priors.lib.joint_root_plans import JointRootPlan
from r0.modules.priors.lib.joint_root_realization import (
    JointRootRealization,
    RootFunctionRealization,
)
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.native_vector_request import (
    NativeVectorDomainPlan,
    NativeVectorGraphRequest,
)
from r0.modules.priors.lib.parent_composition_plans import ParentFunctionPlan
from r0.modules.priors.lib.parent_function_realization import (
    ParentCalibrationReceipt,
    ParentFunctionRealization,
)
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord


@dataclass(frozen=True, slots=True)
class SampledNativeVectorGraph:
    request: NativeVectorGraphRequest
    domains: tuple[NativeVectorDomainPlan, ...]
    mechanisms: tuple[JointRootPlan | JointFunctionPlan | ParentFunctionPlan, ...]
    registry: JointOperationRegistry
    realizations: tuple[
        JointRootRealization | RootFunctionRealization | ParentFunctionRealization, ...
    ]
    raw: tuple[RawDrawRecord, ...]
    recipe: NativeVectorRecipe
    initial: tuple[GeneratorSnapshot, ...]
    final: tuple[GeneratorSnapshot, ...]
    raw_limits: RawDrawLimits
    calibrations: tuple[ParentCalibrationReceipt, ...] = ()
    version: str = "sampled-native-vector-graph-v1"

    @property
    def task(self) -> LinearRecipe:
        return self.recipe.task

    def validate_replay(self) -> None:
        from r0.modules.priors.lib.native_vector_sampling import sample_native_vector_graph

        if self.version != "sampled-native-vector-graph-v1" or len(self.initial) != 4:
            raise ValueError("native archive requires its version and four initial streams")
        streams = tuple(
            s.restore(byte_limit=self.request.raw_limits.state_bytes) for s in self.initial
        )
        replay = sample_native_vector_graph(
            self.request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
        if replay != self:
            raise ValueError("native construction differs from complete replay")

    def with_task(self, task: LinearRecipe) -> "SampledNativeVectorGraph":
        if replace(self.task, context_rows=task.context_rows, split=task.split) != task:
            raise ValueError("native archive task replacement supports episode resplits only")
        return replace(
            self,
            request=replace(self.request, context_rows=task.context_rows, split=task.split),
            recipe=replace(self.recipe, task=task),
        )
