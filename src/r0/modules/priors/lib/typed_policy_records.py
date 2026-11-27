"""Versioned typed receipts preserve full original construction and fit custody."""

from dataclasses import asdict, dataclass, replace
from typing import Any

from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord
from r0.modules.priors.lib.sampled_graph_decoding import construction_matches
from r0.modules.priors.lib.typed_policy_request import TypedInterventionPolicyRequest


@dataclass(frozen=True, slots=True)
class TypedInterventionBounds:
    location: int
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    version: str = "typed-intervention-bounds-v1"


@dataclass(frozen=True, slots=True)
class SampledTypedInterventionPolicy:
    request: TypedInterventionPolicyRequest
    selected: tuple[int, ...]
    bounds: tuple[TypedInterventionBounds, ...]
    reference_seed: int
    runtime: CausalInterventionRecipe
    initial: tuple[GeneratorSnapshot, ...]
    final: tuple[GeneratorSnapshot, ...]
    raw: tuple[RawDrawRecord, ...]
    reference_raw: tuple[RawDrawRecord, ...]
    version: str = "sampled-typed-intervention-policy-v1"

    @property
    def task(self):
        return self.request.base.task

    def validate_replay(self) -> None:
        from r0.modules.priors.lib.typed_policy_admission import admit_typed_policy
        from r0.modules.priors.lib.typed_policy_sampling import sample_typed_intervention_policy

        admit_typed_policy(self.request)
        if self.version != "sampled-typed-intervention-policy-v1" or len(self.initial) != 3:
            raise ValueError("typed policy requires its exact version and three initial snapshots")
        streams = [s.restore(byte_limit=self.request.raw_limits.state_bytes) for s in self.initial]
        replay = sample_typed_intervention_policy(
            self.request, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
        )
        if not construction_matches(replay, asdict(self)):
            raise ValueError("typed policy differs from exact complete replay")

    def with_task(self, task: LinearRecipe) -> "SampledTypedInterventionPolicy":
        from r0.modules.priors.lib.typed_policy_admission import admit_typed_policy
        from r0.modules.priors.lib.typed_policy_sampling import sample_typed_intervention_policy

        rows = (
            task.rows
            if self.request.bounds_lifecycle == "source-reference-population"
            else self.request.reference_rows
        )
        # Prospective geometry is admitted before any envelope-specific replay.
        admit_typed_policy(
            replace(self.request, base=replace(self.request.graph, task=task), reference_rows=rows)
        )
        self.validate_replay()
        base: Any = self.request.base
        revised = (
            base.with_task(task)
            if callable(getattr(base, "with_task", None))
            else replace(base, task=task)
        )
        request = replace(self.request, base=revised, reference_rows=rows)
        admit_typed_policy(request)
        streams = [s.restore(byte_limit=request.raw_limits.state_bytes) for s in self.initial]
        return sample_typed_intervention_policy(
            request, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
        )
