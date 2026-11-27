"""Controls for sampled topology, numerical observation geometry and mechanisms."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.cauchy_graph import CauchyGraphPrior
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.geometry_topology import (
    GeometryTopologyPrior,
    geometry_topology_raw_reservation,
)
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_graph_request import JointGraphNoise, JointGraphRequest
from r0.modules.priors.lib.joint_observation_plans import FeatureGroup
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.joint_width_plans import JointWidthPlan, JointWidthPrior
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline, NumericalConverter
from r0.modules.priors.lib.observation_channel_layout import ObservationChannelLayout
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.structural_graph_prior import AttachmentGraphPrior, MotifGraphPrior


@dataclass(frozen=True, slots=True)
class GeometryGraphRequest:
    topology: GeometryTopologyPrior
    task: LinearRecipe
    subsample: bool = True
    width_prior: JointWidthPrior = JointWidthPrior()
    calibration_rows: int = 256
    root_prior: JointRootPrior = JointRootPrior()
    function_prior: JointFunctionPrior = JointFunctionPrior()
    noise_scale: float = 0.0
    mode: str = "meta"
    categorical_policy: str = "shared"
    global_policy: str = "scalar-quantile-v1"
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "geometry-graph-request-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "geometry-graph-request-v1"
            or type(self.topology) not in (CauchyGraphPrior, AttachmentGraphPrior, MotifGraphPrior)
            or type(self.task) is not LinearRecipe
            or type(self.subsample) is not bool
            or type(self.width_prior) is not JointWidthPrior
            or type(self.root_prior) is not JointRootPrior
            or type(self.function_prior) is not JointFunctionPrior
            or type(self.limits) is not CausalLimits
            or type(self.raw_limits) is not RawDrawLimits
        ):
            raise ValueError("geometry graph requires its version and frozen controls")
        if (
            type(self.calibration_rows) is not int
            or not 1 <= self.calibration_rows < 2**63
            or not finite_number(self.noise_scale)
            or self.noise_scale < 0
            or self.mode not in ("meta", "local", "global")
            or self.categorical_policy not in ("shared", "refresh")
            or self.global_policy not in ("scalar-quantile-v1", "strict-contract-v1")
        ):
            raise ValueError("invalid geometry calibration, structural noise or sampling policy")
        if (
            self.topology.nodes > self.limits.nodes
            or self.topology.nodes + sum(g.count for g in self.groups) > self.limits.total_channels
        ):
            raise ValueError("geometry graph node or minimum channel limit exceeded")
        function_record_cost(self, self.limits)

    @property
    def groups(self) -> tuple[FeatureGroup, ...]:
        outputs = self.task.classes if self.task.task_type == "multiclass" else 1
        return (FeatureGroup("x", self.task.features), FeatureGroup("y", outputs))

    def mechanism_raw_limits(self) -> RawDrawLimits:
        n = self.topology.nodes
        topology_draws, topology_numbers, parameter_bytes = geometry_topology_raw_reservation(
            self.topology
        )
        draws = topology_draws + (4 if self.subsample else 2)
        numbers = topology_numbers + sum(g.count for g in self.groups)
        if self.subsample:
            numbers += 2 * n
        storage = 96 * numbers + 4096 * draws + parameter_bytes + 4 * self.raw_limits.state_bytes
        remaining = (
            self.raw_limits.draws - draws,
            self.raw_limits.numbers - numbers,
            self.raw_limits.working_bytes - storage,
        )
        if min(remaining) <= 0:
            raise ValueError("aggregate geometry raw draw limit exceeded")
        return RawDrawLimits(*remaining, state_bytes=self.raw_limits.state_bytes)

    def mechanism_limits(self, retained: tuple[object, ...]) -> CausalLimits:
        numbers, storage = function_record_cost((self, retained), self.limits)
        n = self.topology.nodes
        reserve_numbers = 4 * numbers + 3 * n * n + 512
        reserve_bytes = 4 * storage + 1024 * n * n + 8 * 1024**2
        reserve_bytes += 16 * min(self.raw_limits.state_bytes, 1024**2)
        if self.limits.parameters <= reserve_numbers or self.limits.working_bytes <= reserve_bytes:
            raise ValueError("aggregate geometry construction limit exceeded")
        return replace(
            self.limits,
            parameters=self.limits.parameters - reserve_numbers,
            working_bytes=self.limits.working_bytes - reserve_bytes,
        )

    def graph_request(
        self,
        parents: tuple[tuple[int, ...], ...],
        layout: ObservationChannelLayout,
        widths: JointWidthPlan,
        *,
        limits: CausalLimits,
    ) -> JointGraphRequest:
        pipelines = tuple(
            CausalNodePipeline(width, converters=tuple(NumericalConverter(i) for i in range(count)))
            for width, count in zip(widths.widths, layout.converter_channels, strict=True)
        )
        return JointGraphRequest(
            self.task,
            parents,
            widths.widths,
            tuple((node, start) for node, start, _ in layout.spans[0]),
            tuple((node, start) for node, start, _ in layout.spans[1]),
            self.calibration_rows,
            self.root_prior,
            self.function_prior,
            tuple(JointGraphNoise((float(self.noise_scale),) * width) for width in widths.widths),
            pipelines,
            self.mode,
            self.categorical_policy,
            self.global_policy,
            limits,
            self.mechanism_raw_limits(),
        )
