"""Versioned controls for sampling topology before frozen graph mechanisms."""

from dataclasses import dataclass, fields, replace

from r0.modules.priors.lib.cauchy_graph import CauchyGraphPrior
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_graph_request import JointGraphNoise, JointGraphRequest
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


@dataclass(frozen=True, slots=True)
class TopologyGraphRequest:
    topology: CauchyGraphPrior
    task: LinearRecipe
    widths: tuple[int, ...]
    observed: tuple[tuple[int, int], ...]
    target: tuple[tuple[int, int], ...]
    calibration_rows: int = 256
    root_prior: JointRootPrior = JointRootPrior()
    function_prior: JointFunctionPrior = JointFunctionPrior()
    noise: tuple[JointGraphNoise, ...] = ()
    authored_pipelines: tuple[CausalNodePipeline | None, ...] = ()
    mode: str = "meta"
    categorical_policy: str = "shared"
    global_policy: str = "scalar-quantile-v1"
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "topology-graph-request-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "topology-graph-request-v1"
            or type(self.topology) is not CauchyGraphPrior
            or type(self.limits) is not CausalLimits
        ):
            raise ValueError("topology graph requires its version, prior and limits")
        if (
            type(self.widths) is not tuple
            or len(self.widths) != self.topology.nodes
            or len(self.widths) > self.limits.nodes
        ):
            raise ValueError("topology graph widths must match its bounded node count")
        # Validate shared controls before draws. This empty-edge geometry is only
        # a preflight check, never retained or used as the realized topology.
        self.for_parents(tuple(() for _ in self.widths), limits=self.limits)
        self.mechanism_limits()

    def for_parents(
        self,
        parents: tuple[tuple[int, ...], ...],
        *,
        limits: CausalLimits,
        raw_limits: RawDrawLimits | None = None,
    ) -> JointGraphRequest:
        controls = {
            f.name: getattr(self, f.name)
            for f in fields(JointGraphRequest)
            if f.name not in ("parents", "limits", "raw_limits", "version")
        }
        return JointGraphRequest(
            parents=parents,
            limits=limits,
            raw_limits=self.raw_limits if raw_limits is None else raw_limits,
            **controls,
        )

    def mechanism_limits(self) -> CausalLimits:
        """Reserve topology, envelope and snapshot copies before either sampler runs."""
        numbers, storage = function_record_cost(self, self.limits)
        n = self.topology.nodes
        reserved_numbers = 2 * (3 * n * n + 4 * n + 512) + 4 * numbers
        reserved_bytes = (
            1024 * (n * n + 3 * n + 1)
            + 4 * storage
            + 8 * 1024**2
            + 16 * min(self.raw_limits.state_bytes, 1024**2)
        )
        remaining_numbers = self.limits.parameters - reserved_numbers
        remaining_bytes = self.limits.working_bytes - reserved_bytes
        if remaining_numbers <= 0 or remaining_bytes <= 0:
            raise ValueError("aggregate topology graph construction limit exceeded")
        return replace(self.limits, parameters=remaining_numbers, working_bytes=remaining_bytes)

    def mechanism_raw_limits(self) -> RawDrawLimits:
        """Charge the full square topology transcript against the combined raw cap."""
        n = self.topology.nodes
        numbers = n * n + 2 * n + 1
        # Four record headers/roles/shapes and a separate topology transaction's
        # snapshots, in addition to its retained numerical values.
        storage = 96 * numbers + 16 * 1024 + 2 * self.raw_limits.state_bytes
        remaining = (
            self.raw_limits.draws - 4,
            self.raw_limits.numbers - numbers,
            self.raw_limits.working_bytes - storage,
        )
        if min(remaining) <= 0:
            raise ValueError("aggregate topology raw draw limit exceeded")
        return RawDrawLimits(*remaining, state_bytes=self.raw_limits.state_bytes)
