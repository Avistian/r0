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
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_domain_prior import NativeDomainPlan
from r0.modules.priors.lib.native_graph_request import NativeGraphRequest
from r0.modules.priors.lib.parent_composition_plans import ParentFunctionPlan
from r0.modules.priors.lib.parent_function_realization import ParentFunctionRealization
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord


@dataclass(frozen=True, slots=True)
class SampledNativeGraph:
    request: NativeGraphRequest
    domains: tuple[NativeDomainPlan, ...]
    mechanisms: tuple[JointRootPlan | JointFunctionPlan | ParentFunctionPlan, ...]
    registry: JointOperationRegistry
    realizations: tuple[
        JointRootRealization | RootFunctionRealization | ParentFunctionRealization, ...
    ]
    raw: tuple[RawDrawRecord, ...]
    recipe: NativeCausalRecipe
    initial: tuple[GeneratorSnapshot, ...]
    final: tuple[GeneratorSnapshot, ...]
    raw_limits: RawDrawLimits
    version: str = "sampled-native-graph-v1"

    @property
    def task(self) -> LinearRecipe:
        return self.recipe.task

    def validate_replay(self) -> None:
        from r0.modules.priors.lib.native_graph_sampling import sample_native_graph

        if self.version != "sampled-native-graph-v1" or len(self.initial) != 4:
            raise ValueError("native archive requires its version and four initial streams")
        streams = tuple(
            s.restore(byte_limit=self.request.raw_limits.state_bytes) for s in self.initial
        )
        replay = sample_native_graph(
            self.request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
        if replay != self:
            raise ValueError("native construction differs from complete replay")

    def with_task(self, task: LinearRecipe) -> "SampledNativeGraph":
        if replace(self.task, context_rows=task.context_rows, split=task.split) != task:
            raise ValueError("native archive task replacement supports episode resplits only")
        return replace(
            self,
            request=replace(self.request, context_rows=task.context_rows, split=task.split),
            recipe=replace(self.recipe, task=task),
        )
