"""Forward Cauchy-logistic DAGs with retained full proposal matrices."""

import math
from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord

_DEFAULT_LIMITS = CausalLimits()
_OWNER = ("cauchy-topology",)
_ROLES = ("global-offset", "receiver-propensity", "source-propensity", "edge-proposals")


@dataclass(frozen=True, slots=True)
class CauchyGraphPrior:
    nodes: int
    offset: float
    version: str = "cauchy-graph-prior-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "cauchy-graph-prior-v1"
            or type(self.nodes) is not int
            or not 1 <= self.nodes < 2**63
            or not finite_number(self.offset)
        ):
            raise ValueError("Cauchy graph requires bounded node count and finite explicit offset")


def _admit(prior: CauchyGraphPrior, limits: CausalLimits) -> None:
    if type(prior) is not CauchyGraphPrior or type(limits) is not CausalLimits:
        raise ValueError("Cauchy graph requires frozen prior and limits")
    n = prior.nodes
    if n > limits.nodes or n > limits.total_channels:
        raise ValueError("Cauchy graph node limit exceeded")
    # Full raw matrix, derived parent indices, retained/replay copies, snapshots.
    # Conservative estimates, not measured RSS. No n-by-n array precedes this gate.
    if 3 * n * n + 4 * n + 512 > limits.parameters:
        raise ValueError("Cauchy graph parameter limit exceeded")
    if 512 * (n * n + 3 * n + 1) + 8 * 1024**2 > limits.working_bytes:
        raise ValueError("Cauchy graph estimated working memory limit exceeded")


def _parents(
    prior: CauchyGraphPrior, raw: tuple[RawDrawRecord, ...]
) -> tuple[tuple[int, ...], ...]:
    n = prior.nodes
    offset = float(raw[0].values[0]) + prior.offset
    if not math.isfinite(offset):
        raise ValueError("nonfinite Cauchy offset; graph was not retried")
    receiver, source, proposals = raw[1].values, raw[2].values, raw[3].values
    parents: list[tuple[int, ...]] = []
    for target in range(n):
        incoming: list[int] = []
        for parent in range(target):
            logit = offset + receiver[target] + source[parent]
            if not math.isfinite(logit):
                raise ValueError("nonfinite Cauchy connectivity; graph was not retried")
            # Stable sigmoid preserves valid extremes without clipping logits.
            exponential = math.exp(-abs(logit))
            probability = 1 / (1 + exponential) if logit >= 0 else exponential / (1 + exponential)
            if proposals[parent * n + target] <= probability:
                incoming.append(parent)
        parents.append(tuple(incoming))
    return tuple(parents)


@dataclass(frozen=True, slots=True)
class CauchyGraph:
    prior: CauchyGraphPrior
    parents: tuple[tuple[int, ...], ...]
    raw: tuple[RawDrawRecord, ...]
    limits: CausalLimits = _DEFAULT_LIMITS
    version: str = "cauchy-graph-v1"

    def validate(self) -> None:
        _admit(self.prior, self.limits)
        n = self.prior.nodes
        if (
            type(self.parents) is not tuple
            or len(self.parents) != n
            or any(
                type(incoming) is not tuple
                or len(incoming) > target
                or any(type(parent) is not int or not 0 <= parent < target for parent in incoming)
                or tuple(sorted(set(incoming))) != incoming
                for target, incoming in enumerate(self.parents)
            )
        ):
            raise ValueError("Cauchy graph parents must be ordered distinct earlier integers")
        if self.version != "cauchy-graph-v1" or type(self.raw) is not tuple or len(self.raw) != 4:
            raise ValueError("Cauchy graph requires its complete versioned raw transcript")
        for i, (record, shape, role) in enumerate(
            zip(self.raw, ((1,), (n,), (n,), (n, n)), _ROLES, strict=True)
        ):
            if (
                type(record) is not RawDrawRecord
                or record.owner != _OWNER
                or record.role != role
                or record.sequence != i
                or record.stream != "raw"
                or record.shape != shape
                or record.distribution != ("random" if i == 3 else "cauchy")
                or record.parameters
            ):
                raise ValueError("Cauchy graph raw binding differs from topology construction")
            if i and (
                record.generator != self.raw[i - 1].generator
                or record.state_before != self.raw[i - 1].state_after
            ):
                raise ValueError("Cauchy graph raw stream chain differs")
        if any(not 0 <= value < 1 for value in self.raw[-1].values):
            raise ValueError("Cauchy graph proposals must be uniform values in [0, 1)")
        if self.parents != _parents(self.prior, self.raw):
            raise ValueError("Cauchy graph parents differ from retained connectivity draws")

    def replay(self, rng: np.random.Generator) -> "CauchyGraph":
        """Verify every draw from its injected pre-state, publishing only on success."""
        self.validate()
        with raw_transactions({"raw": rng}, None) as (streams, _):
            for record in self.raw:
                record.replay(streams["raw"])
            return self


def sample_cauchy_graph(
    prior: CauchyGraphPrior, *, rng: np.random.Generator, limits: CausalLimits = _DEFAULT_LIMITS
) -> CauchyGraph:
    """NumPy float64 adaptation; explicit offset, no connectivity-conditioned retries."""
    _admit(prior, limits)
    n = prior.nodes
    ledger = RawDrawLedger(
        limits=RawDrawLimits(
            draws=4, numbers=n * n + 2 * n + 1, working_bytes=limits.working_bytes // 2
        )
    )
    with ledger.transaction(rng) as (staged, capture):
        for i, (role, shape) in enumerate(zip(_ROLES, ((1,), (n,), (n,), (n, n)), strict=True)):
            capture.draw(
                staged,
                owner=_OWNER,
                role=role,
                distribution="random" if i == 3 else "cauchy",
                shape=shape,
            )
        raw = capture.freeze()
        result = CauchyGraph(prior, _parents(prior, raw), raw, limits)
        result.validate()
        return result
