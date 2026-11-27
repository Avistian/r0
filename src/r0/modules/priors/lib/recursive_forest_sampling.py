"""Transactional recursive-tree sampling with exact raw replay and per-row ownership."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.recursive_forest_plan import (
    RecursiveForestPlan,
    recursive_effective_limits,
)
from r0.modules.priors.lib.recursive_forest_records import (
    RecursiveForest,
    RecursiveTree,
    RecursiveTreeNode,
)

_DEFAULT_LIMITS = CausalLimits()
_DEFAULT_RAW_LIMITS = RawDrawLimits()


def _tree(
    plan: RecursiveForestPlan,
    rng: np.random.Generator,
    ledger: RawDrawLedger,
    population: int,
    tree: int,
) -> RecursiveTree:
    owner = ("recursive-forest", str(population), str(tree))
    start = 0
    if plan.task == "classification" and plan.class_assignment != "iid":
        drawn = int(
            ledger.draw(
                rng,
                owner=owner,
                role="class-start",
                distribution="integers",
                shape=(1,),
                parameters=(("high", plan.classes),),
            )[0]
        )
        start = drawn if plan.class_assignment == "random-start-cycle" else 0
    records: list[dict[str, Any]] = []
    stack = [(0, tuple((float(a), float(b)) for a, b in plan.intervals), -1, "")]
    counter = 0
    while stack:
        depth, intervals, parent, side = stack.pop()
        index = len(records)
        node_owner = (*owner, str(index))
        if plan.split_probabilities:
            distribution = "choice"
            parameters: tuple[tuple[str, Any], ...] = (
                ("count", plan.input_width),
                ("probabilities", plan.split_probabilities),
            )
        else:
            distribution, parameters = "integers", (("high", plan.input_width),)
        feature = int(
            ledger.draw(
                rng,
                owner=node_owner,
                role="split-index",
                distribution=distribution,
                shape=(1,),
                parameters=parameters,
            )[0]
        )
        quantile = float(
            ledger.draw(
                rng, owner=node_owner, role="threshold-quantile", distribution="random", shape=(1,)
            )[0]
        )
        low, high = intervals[feature]
        threshold = float(low + (high - low) * quantile)
        # Algorithm 4 evaluates its left Boolean operand before the minimum-depth override.
        split = False
        if depth < plan.max_depth:
            terminal = float(
                ledger.draw(
                    rng,
                    owner=node_owner,
                    role="termination-quantile",
                    distribution="random",
                    shape=(1,),
                )[0]
            )
            split = terminal >= plan.terminal_probability
        split = split or depth < plan.min_depth
        record: dict[str, Any] = dict(
            feature=feature,
            threshold=threshold,
            interval=(low, high),
            depth=depth,
            left=-1,
            right=-1,
            value=(),
            label=None,
        )
        records.append(record)
        if parent >= 0:
            records[parent][side] = index
        if split:
            for side, interval in (("right", (threshold, high)), ("left", (low, threshold))):
                child_intervals = list(intervals)
                child_intervals[feature] = interval
                stack.append((depth + 1, tuple(child_intervals), index, side))
        elif plan.task == "regression":
            z = ledger.draw(
                rng,
                owner=node_owner,
                role="leaf-normal",
                distribution="normal",
                shape=(plan.output_width,),
            )
            record["value"] = tuple(
                float(m + s * float(v))
                for m, s, v in zip(plan.leaf_mean, plan.leaf_scale, z, strict=True)
            )
        else:
            if plan.class_assignment == "iid":
                record["label"] = int(
                    ledger.draw(
                        rng,
                        owner=node_owner,
                        role="leaf-label",
                        distribution="integers",
                        shape=(1,),
                        parameters=(("high", plan.classes),),
                    )[0]
                )
            else:
                record["label"] = (start + counter) % plan.classes
            counter += 1
    return RecursiveTree(tuple(RecursiveTreeNode(**record) for record in records))


def _realize(
    plan: RecursiveForestPlan,
    rng: np.random.Generator,
    ledger: RawDrawLedger,
) -> tuple[RecursiveForest, ...]:
    return tuple(
        RecursiveForest(
            plan.input_width,
            tuple(_tree(plan, rng, ledger, row, tree) for tree in range(plan.estimators)),
            plan.task,
            plan.output_width,
            plan.classes,
            plan.tie_break,
        )
        for row in range(plan.populations)
    )


@dataclass(frozen=True, slots=True)
class SampledRecursiveForest:
    plan: RecursiveForestPlan
    forests: tuple[RecursiveForest, ...]
    initial: GeneratorSnapshot
    final: GeneratorSnapshot
    raw: tuple[RawDrawRecord, ...]
    limits: CausalLimits = _DEFAULT_LIMITS
    raw_limits: RawDrawLimits = _DEFAULT_RAW_LIMITS
    version: str = "sampled-recursive-forest-v1"
    reconstruction: str = "path-local-depth-first-left-first-tree-local-cycle-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "sampled-recursive-forest-v1"
            or self.reconstruction != "path-local-depth-first-left-first-tree-local-cycle-v1"
            or type(self.plan) is not RecursiveForestPlan
            or type(self.forests) is not tuple
            or len(self.forests) != self.plan.populations
            or any(type(f) is not RecursiveForest for f in self.forests)
            or type(self.initial) is not GeneratorSnapshot
            or type(self.final) is not GeneratorSnapshot
            or type(self.raw) is not tuple
            or any(type(r) is not RawDrawRecord for r in self.raw)
        ):
            raise ValueError("invalid recursive forest construction custody")
        self.validate_replay(limits=self.limits)

    def validate_replay(self, *, limits: CausalLimits = _DEFAULT_LIMITS) -> None:
        limits = recursive_effective_limits(self.limits, limits)
        budget = self.plan.admit(limits, self.raw_limits)
        function_record_cost(self, limits)
        if (
            len(self.raw) > budget["draws"]
            or sum(len(record.values) for record in self.raw) > budget["numbers"]
        ):
            raise ValueError("recursive raw geometry exceeds plan")
        for forest in self.forests:
            if (
                forest.input_width != self.plan.input_width
                or forest.output_width != self.plan.output_width
                or forest.task != self.plan.task
                or forest.classes != self.plan.classes
                or forest.tie_break != self.plan.tie_break
                or len(forest.trees) != self.plan.estimators
                or any(
                    len(tree.nodes) > 2 ** (self.plan.max_depth + 1) - 1 for tree in forest.trees
                )
            ):
                raise ValueError("recursive forest geometry differs from retained plan")
            forest.admit(1, limits)
        rng = self.initial.restore(byte_limit=self.raw_limits.state_bytes)
        ledger = RawDrawLedger(limits=self.raw_limits)
        with ledger.transaction(rng) as (owned, capture):
            forests = _realize(self.plan, owned, capture)
            final = snapshot_generator(owned, byte_limit=self.raw_limits.state_bytes)
            raw = capture.freeze()
        if (
            not exact_frozen_equal(forests, self.forests)
            or not exact_frozen_equal(raw, self.raw)
            or not exact_frozen_equal(final, self.final)
        ):
            raise ValueError("recursive forest differs from exact raw construction replay")


def sample_recursive_forest(
    plan: RecursiveForestPlan,
    *,
    rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT_LIMITS,
    raw_limits: RawDrawLimits = _DEFAULT_RAW_LIMITS,
) -> SampledRecursiveForest:
    if type(plan) is not RecursiveForestPlan:
        raise ValueError("recursive sampling requires its explicit plan")
    plan.admit(limits, raw_limits)
    initial = snapshot_generator(rng, byte_limit=raw_limits.state_bytes)
    ledger = RawDrawLedger(limits=raw_limits)
    with ledger.transaction(rng) as (owned, capture):
        forests = _realize(plan, owned, capture)
        result = SampledRecursiveForest(
            plan,
            forests,
            initial,
            snapshot_generator(owned, byte_limit=raw_limits.state_bytes),
            capture.freeze(),
            limits,
            raw_limits,
        )
    return result
