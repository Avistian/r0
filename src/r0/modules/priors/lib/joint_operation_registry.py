"""Completed operation ownership, including scopes without named decisions."""

from dataclasses import dataclass

from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_path


@dataclass(frozen=True, slots=True)
class JointRegistryLimits:
    operations: int = 100_000
    path_bytes: int = 16 * 1024**2

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or not 1 <= value < 2**63
            for value in (self.operations, self.path_bytes)
        ):
            raise ValueError("joint registry limits must be positive bounded integers")

    def admit(self, count: int, path_bytes: int) -> None:
        if any(type(value) is not int or value < 0 for value in (count, path_bytes)):
            raise ValueError("joint registry admission counts must be nonnegative integers")
        if count > self.operations or path_bytes > self.path_bytes:
            raise ValueError("joint operation registry limit exceeded")


@dataclass(frozen=True, slots=True)
class JointOperationSpan:
    operation: tuple[str, ...]
    event_start: int
    event_stop: int

    def __post_init__(self) -> None:
        validate_path(self.operation)
        if (
            type(self.event_start) is not int
            or type(self.event_stop) is not int
            or not 0 <= self.event_start <= self.event_stop < 2**63
        ):
            raise ValueError("joint operation spans require ordered bounded integer event indices")

    @property
    def path_bytes(self) -> int:
        return sum(len(name.encode("utf-8")) for name in self.operation)


@dataclass(frozen=True, slots=True)
class JointOperationRegistry:
    decisions: JointSamplingRecord
    operations: tuple[JointOperationSpan, ...]
    limits: JointRegistryLimits = JointRegistryLimits()
    version: str = "joint-operation-registry-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "joint-operation-registry-v1"
            or type(self.decisions) is not JointSamplingRecord
            or type(self.operations) is not tuple
            or type(self.limits) is not JointRegistryLimits
            or any(type(span) is not JointOperationSpan for span in self.operations)
        ):
            raise ValueError("joint registry requires frozen decisions, operation spans and limits")
        self.limits.admit(len(self.operations), sum(span.path_bytes for span in self.operations))
        by_path: dict[tuple[str, ...], JointOperationSpan] = {}
        stack: list[JointOperationSpan] = []
        for span in self.operations:
            if span.operation in by_path:
                raise ValueError("joint registry operation paths must be unique")
            if span.event_stop > len(self.decisions.events):
                raise ValueError("joint registry span exceeds its decision ledger")
            earliest = 0
            parent = span.operation[:-1]
            while stack and stack[-1].operation != parent:
                earliest = max(earliest, stack.pop().event_stop)
            if len(parent) and (not stack or stack[-1].operation != parent):
                raise ValueError("joint registry requires parents before children in preorder")
            if span.event_start < earliest:
                raise ValueError("joint registry sibling event spans must not overlap")
            if stack and not (
                stack[-1].event_start <= span.event_start <= span.event_stop <= stack[-1].event_stop
            ):
                raise ValueError("joint registry child span must lie within its parent")
            by_path[span.operation] = span
            stack.append(span)
        # At most16 prefixes per event; never scan all events separately for every scope.
        bounds: dict[tuple[str, ...], tuple[int, int, int]] = {}
        for index, event in enumerate(self.decisions.events):
            if event.operation not in by_path:
                raise ValueError("joint decision has no registered operation owner")
            for length in range(1, len(event.operation) + 1):
                path = event.operation[:length]
                first, _, count = bounds.get(path, (index, index, 0))
                bounds[path] = (first, index + 1, count + 1)
        for path, span in by_path.items():
            expected = bounds.get(path)
            if expected is None:
                if span.event_start != span.event_stop:
                    raise ValueError("event-free joint operations must have empty spans")
            elif expected != (
                span.event_start,
                span.event_stop,
                span.event_stop - span.event_start,
            ):
                raise ValueError(
                    "joint registry span must contain exactly its owned subtree events"
                )
