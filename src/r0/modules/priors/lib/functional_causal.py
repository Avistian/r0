"""Authored DAGs whose structural equations are frozen, dimension-changing functions."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.frozen_function_evaluation import FrozenFunction, function_geometry
from r0.modules.priors.lib.functional_analysis import functional_certificates
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.joint_root_base import JointRootBase
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline


@dataclass(frozen=True, slots=True)
class FunctionalCausalNode:
    parents: tuple[int, ...]
    function: FrozenFunction
    root: JointRootBase | None = None
    noise_scale: tuple[float, ...] = ()
    exogenous: ExogenousDistribution = ExogenousDistribution()
    noise_response: NoiseResponse = NoiseResponse()
    pipeline: CausalNodePipeline | None = None
    version: str = "functional-causal-node-v1"

    def __post_init__(self) -> None:
        if self.version != "functional-causal-node-v1":
            raise ValueError("unsupported functional node version")
        function_geometry(self.function)
        width = self.function.output_width
        if (
            type(self.noise_scale) is not tuple
            or self.noise_scale
            and (
                len(self.noise_scale) != width
                or any(not finite_number(v) or v < 0 for v in self.noise_scale)
            )
        ):
            raise ValueError("noise scales must be nonnegative finite output-channel factors")
        if (
            type(self.exogenous) is not ExogenousDistribution
            or type(self.noise_response) is not NoiseResponse
        ):
            raise ValueError("functional noise requires frozen exogenous and response descriptors")
        self.exogenous.validate_width(width)
        if self.pipeline is not None and (
            type(self.pipeline) is not CausalNodePipeline or self.pipeline.channels != width
        ):
            raise ValueError("functional pipeline width must match function output")
        if (
            type(self.parents) is not tuple
            or any(type(p) is not int or p < 0 for p in self.parents)
            or len(set(self.parents)) != len(self.parents)
        ):
            raise ValueError("functional parents must be distinct nonnegative integer indices")
        if self.parents:
            if self.root is not None:
                raise ValueError("only parentless functions have a root base")
        elif type(self.root) is not JointRootBase or self.root.width != self.function.input_width:
            raise ValueError("root base width must match the root function input")


@dataclass(frozen=True, slots=True)
class FunctionalCausalRecipe:
    task: LinearRecipe
    nodes: tuple[FunctionalCausalNode, ...]
    observed: tuple[tuple[int, int], ...]
    target: tuple[tuple[int, int], ...]
    limits: CausalLimits = CausalLimits()
    version: str = "functional-causal-v1"

    @property
    def target_calibration_rows(self) -> int:
        return 0

    def __post_init__(self) -> None:
        if (
            self.version not in ("functional-causal-v1", "functional-causal-v2")
            or self.version == "functional-causal-v2"
            and not self.target_calibration_rows
            or type(self.task) is not LinearRecipe
        ):
            raise ValueError("functional graph requires its version and linear-v1 task controls")
        if type(self.limits) is not CausalLimits:
            raise ValueError("functional graph requires CausalLimits")
        if type(self.nodes) is not tuple or not self.nodes:
            raise ValueError("functional graph requires a nonempty node tuple")
        if len(self.nodes) > self.limits.nodes:
            raise ValueError("functional graph node count limit exceeded")
        for index, node in enumerate(self.nodes):
            if type(node) is not FunctionalCausalNode or any(p >= index for p in node.parents):
                raise ValueError("functional nodes must be in acyclic parent-before-child order")
            if node.parents and node.function.input_width != sum(
                self.nodes[p].function.output_width for p in node.parents
            ):
                raise ValueError("function input width must match concatenated parent state widths")
        for refs in (self.observed, self.target):
            if type(refs) is not tuple or any(
                type(ref) is not tuple
                or len(ref) != 2
                or any(type(v) is not int for v in ref)
                or not 0 <= ref[0] < len(self.nodes)
                or not 0 <= ref[1] < self.nodes[ref[0]].function.output_width
                for ref in refs
            ):
                raise ValueError("functional readouts must identify valid node/channel pairs")
        outputs = self.task.classes if self.task.task_type == "multiclass" else 1
        if len(self.observed) != self.task.features or len(self.target) != outputs:
            raise ValueError("functional readout widths must match task controls")
        if set(self.observed) & set(self.target):
            raise ValueError("a declared target channel cannot be observed as a feature")
        self.admit()
        _, readouts = functional_certificates(self.nodes)
        if self.target_calibration_rows and any(readouts[i][j].order <= 2 for i, j in self.target):
            raise ValueError("target normalization requires certified finite second moments")

    def admit(self, *, evaluation_rows: int | None = None) -> dict[str, int]:
        """Aggregate frozen records and all live graph/function/root buffers before arrays.

        Numeric metadata includes all nodes, root factors and retained activation fits.
        Working bytes are a conservative estimate, not a measured RSS or OS memory cap.
        """
        if evaluation_rows is not None and (
            type(evaluation_rows) is not int or not 1 <= evaluation_rows < 2**63
        ):
            raise ValueError("functional evaluation rows must be positive bounded integers")
        parameters, storage = function_record_cost(
            (self.task, self.nodes, self.observed, self.target, self.limits), self.limits
        )
        if self.target_calibration_rows:
            extra = 3 * len(self.target) + 1
            parameters += extra
            storage += 64 * extra + 1024
            if parameters > self.limits.parameters:
                raise ValueError("target calibration parameter limit exceeded")
        geometries = [function_geometry(node.function) for node in self.nodes]
        widest = max(max(widths) for widths, _, _ in geometries)
        total = sum(node.function.output_width for node in self.nodes)
        if widest > self.limits.node_channels or total > self.limits.total_channels:
            raise ValueError("functional graph channel limit exceeded")
        fan_in = max(node.function.input_width for node in self.nodes)
        roots = max((node.root.width for node in self.nodes if node.root), default=0)
        rows = max(
            self.task.rows,
            self.target_calibration_rows,
            evaluation_rows or 0,
            max(floor for _, _, floor in geometries),
            256 if any(n.exogenous.inputs.bound is not None for n in self.nodes) else 0,
        )
        pipeline_work = max(
            (n.pipeline.working_extra_channels for n in self.nodes if n.pipeline), default=0
        )
        source_bound = sum(
            (n.root.width if n.root else 0)
            + (
                n.function.output_width
                if any(n.noise_scale) and (n.noise_response.offset or n.noise_response.strength)
                else 0
            )
            for n in self.nodes
        )
        # State and readout certificates can each retain every preceding source.
        # Charge Python set entries separately from numeric evaluation buffers.
        analysis_bytes = 256 * total * source_bound + 256 * total
        working = (
            8
            * rows
            * (
                6 * total
                + 2 * fan_in
                + 2 * self.task.features
                + (8 if self.target_calibration_rows else 4) * len(self.target)
                + max(work for _, work, _ in geometries)
                + 40 * roots
                + pipeline_work
                + 4
            )
            + analysis_bytes
            + 128 * parameters
            + 2 * (storage - 64 * parameters)
        )
        if working > self.limits.working_bytes:
            raise ValueError("estimated functional graph working memory exceeds declared limit")
        return {
            "nodes": len(self.nodes),
            "largest_node_channels": widest,
            "total_channels": total,
            "parameters": parameters,
            "estimated_analysis_bytes": analysis_bytes,
            "estimated_working_bytes": working,
        }


@dataclass(frozen=True, slots=True)
class CalibratedFunctionalCausalRecipe(FunctionalCausalRecipe):
    """Version two adds fixed target calibration without changing version-one records."""

    calibration_rows: int = 256
    version: str = "functional-causal-v2"

    @property
    def target_calibration_rows(self) -> int:
        return self.calibration_rows

    def __post_init__(self) -> None:
        if (
            self.version != "functional-causal-v2"
            or type(self.calibration_rows) is not int
            or not 32 <= self.calibration_rows <= 65536
        ):
            raise ValueError(
                "functional target calibration requires version two and 32..65536 rows"
            )
        FunctionalCausalRecipe.__post_init__(self)
