"""Complete sampled intervention receipts, with exact replay independent of query rows."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.intervention_policy_request import InterventionPolicyRequest
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord


@dataclass(frozen=True, slots=True)
class InterventionBounds:
    node: int
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    version: str = "intervention-bounds-v1"


@dataclass(frozen=True, slots=True)
class SampledInterventionPolicy:
    request: InterventionPolicyRequest
    selected: tuple[int, ...]
    bounds: tuple[InterventionBounds, ...]
    reference_seed: int
    runtime: CausalInterventionRecipe
    initial: tuple[GeneratorSnapshot, ...]
    final: tuple[GeneratorSnapshot, ...]
    raw: tuple[RawDrawRecord, ...]
    reference_raw: tuple[RawDrawRecord, ...]
    version: str = "sampled-intervention-policy-v1"

    @property
    def task(self):
        return self.request.base.task

    def with_task(self, task: LinearRecipe) -> "SampledInterventionPolicy":
        from r0.modules.priors.lib.intervention_policy_sampling import (
            admit_intervention_policy,
            sample_intervention_policy,
        )

        request = replace(
            self.request,
            base=replace(self.request.base, task=task),
            reference_rows=task.rows
            if self.request.bounds_lifecycle == "source-reference-population"
            else self.request.reference_rows,
        )
        admit_intervention_policy(request)
        self.validate_replay()
        streams = tuple(s.restore(byte_limit=request.raw_limits.state_bytes) for s in self.initial)
        return sample_intervention_policy(
            request, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
        )

    def validate_replay(self) -> None:
        from dataclasses import asdict

        from r0.modules.priors.lib.intervention_policy_sampling import (
            admit_intervention_policy,
            sample_intervention_policy,
        )
        from r0.modules.priors.lib.sampled_graph_decoding import construction_matches

        admit_intervention_policy(self.request)
        if self.version != "sampled-intervention-policy-v1" or len(self.initial) != 3:
            raise ValueError("sampled intervention requires its complete stream custody")
        streams = tuple(
            s.restore(byte_limit=self.request.raw_limits.state_bytes) for s in self.initial
        )
        replay = sample_intervention_policy(
            self.request,
            selection_rng=streams[0],
            value_rng=streams[1],
            calibration_rng=streams[2],
        )
        if not construction_matches(replay, asdict(self)):
            raise ValueError("sampled intervention policy differs from complete replay")
