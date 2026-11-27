"""Authored DAGs whose structural equations are frozen, dimension-changing functions."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_analysis import functional_certificates
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.functional_graph_admission import (
    admit_functional_graph,
    validate_functional_nodes,
)
from r0.modules.priors.lib.linear import LinearRecipe


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
            self.version
            not in (
                "functional-causal-v1",
                "functional-causal-v2",
                "fitted-functional-causal-v1",
                "sampled-fitted-functional-causal-v1",
            )
            or self.version == "functional-causal-v2"
            and not self.target_calibration_rows
            or self.version
            in ("fitted-functional-causal-v1", "sampled-fitted-functional-causal-v1")
            and not getattr(self, "fits", ())
            or type(self.task) is not LinearRecipe
        ):
            raise ValueError("functional graph requires its version and linear-v1 task controls")
        validate_functional_nodes(self.nodes, self.limits)
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
        return admit_functional_graph(
            self.nodes,
            self.limits,
            rows=max(self.task.rows, self.target_calibration_rows, evaluation_rows or 0),
            metadata=(self.task, self.nodes, self.observed, self.target, self.limits),
            features=self.task.features,
            outputs=len(self.target),
            target_calibration=bool(self.target_calibration_rows),
        )


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
