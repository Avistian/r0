"""Retained raw permutations and with-replacement feature-node assignments."""

from dataclasses import dataclass, replace

import numpy as np

from r0.modules.priors.lib.joint_observation_plans import JointObservationPlan
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord

_RAW_LIMITS = RawDrawLimits()


def _assignments(
    plan: JointObservationPlan, raw: tuple[RawDrawRecord, ...]
) -> tuple[tuple[int, ...], ...]:
    per_group = 2 if plan.subsample else 1
    if type(raw) is not tuple or len(raw) != per_group * len(plan.groups):
        raise ValueError("observation assignments require complete raw records")
    assignments: list[tuple[int, ...]] = []
    for i, (group, count) in enumerate(zip(plan.groups, plan.eligible_counts, strict=True)):
        nodes = tuple(range(plan.nodes))
        for j in range(per_group):
            index = i * per_group + j
            record = raw[index]
            permutation = plan.subsample and j == 0
            if (
                type(record) is not RawDrawRecord
                or record.sequence != index
                or record.owner != (*plan.operation, f"group:{i}")
                or record.role != ("eligible-permutation" if permutation else "assignments")
                or record.stream != "raw"
                or record.distribution != ("permutation" if permutation else "integers")
                or record.shape != ((plan.nodes,) if permutation else (group.count,))
            ):
                raise ValueError("observation raw binding differs")
            expected_parameters = (("count", plan.nodes),) if permutation else (("high", count),)
            if record.parameters != expected_parameters or record.dtype != "<i8":
                raise ValueError("observation raw law differs")
            if index and (
                record.generator != raw[index - 1].generator
                or record.state_before != raw[index - 1].state_after
            ):
                raise ValueError("observation raw stream chain differs")
            values = tuple(int(v) for v in record.values)
            if permutation:
                if tuple(sorted(values)) != tuple(range(plan.nodes)):
                    raise ValueError("observation permutation is not a node permutation")
                nodes = values[:count]
            else:
                if any(not 0 <= v < count for v in values):
                    raise ValueError("observation assignments exceed eligible nodes")
                assignments.append(tuple(nodes[v] for v in values))
    return tuple(assignments)


@dataclass(frozen=True, slots=True)
class JointObservationRealization:
    plan: JointObservationPlan
    assignments: tuple[tuple[int, ...], ...]
    raw: tuple[RawDrawRecord, ...]
    version: str = "joint-observation-realization-v1"

    def validate(self, decisions: JointSamplingRecord) -> None:
        if (
            self.version != "joint-observation-realization-v1"
            or type(self.plan) is not JointObservationPlan
        ):
            raise ValueError("observations require a frozen plan and supported version")
        self.plan.validate(decisions)
        if (
            type(self.assignments) is not tuple
            or any(
                type(row) is not tuple or any(type(n) is not int for n in row)
                for row in self.assignments
            )
            or self.assignments != _assignments(self.plan, self.raw)
        ):
            raise ValueError("observation assignments differ from retained raw choices")

    def replay(
        self, decisions: JointSamplingRecord, rng: np.random.Generator
    ) -> "JointObservationRealization":
        self.validate(decisions)
        with raw_transactions({"raw": rng}, None) as (streams, _):
            for record in self.raw:
                record.replay(streams["raw"])
            return self


def materialize_joint_observations(
    plan: JointObservationPlan,
    decisions: JointSamplingRecord,
    *,
    rng: np.random.Generator,
    raw_limits: RawDrawLimits = _RAW_LIMITS,
) -> JointObservationRealization:
    if type(plan) is not JointObservationPlan or type(raw_limits) is not RawDrawLimits:
        raise ValueError("observations require a frozen plan and raw limits")
    plan.validate(decisions)
    limits = replace(
        raw_limits, working_bytes=min(raw_limits.working_bytes, plan.limits.working_bytes)
    )
    ledger = RawDrawLedger(limits=limits)
    with ledger.transaction(rng) as (staged, capture):
        for i, (group, count) in enumerate(zip(plan.groups, plan.eligible_counts, strict=True)):
            owner = (*plan.operation, f"group:{i}")
            if plan.subsample:
                capture.draw(
                    staged,
                    owner=owner,
                    role="eligible-permutation",
                    distribution="permutation",
                    shape=(plan.nodes,),
                    parameters=(("count", plan.nodes),),
                )
            capture.draw(
                staged,
                owner=owner,
                role="assignments",
                distribution="integers",
                shape=(group.count,),
                parameters=(("high", count),),
            )
        raw = capture.freeze()
        result = JointObservationRealization(plan, _assignments(plan, raw), raw)
        result.validate(decisions)
        return result
