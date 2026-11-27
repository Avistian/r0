"""Admitted, atomic matrix batches with source-order raw realizations."""

from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_matrix_plans import JointMatrixPlan
from r0.modules.priors.lib.joint_point_realization import frozen_matrix, materialize_joint_points
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord
from r0.modules.priors.lib.joint_weight_plans import JointWeightPlan, materialize_joint_weights
from r0.modules.priors.lib.kernel_points import KernelPointCloud
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transaction, recorded_draw
from r0.modules.priors.lib.transformed_weights import (
    GaussianWeights,
    NormalizedWeights,
    SimpleWeights,
    TransformedWeights,
)
from r0.modules.priors.lib.weight_geometry import Matrix
from r0.modules.priors.lib.weight_matrices import (
    ImportanceFactorizedWeights,
    ImportanceWeights,
    LaplaceKernelWeights,
)
from r0.modules.priors.lib.weight_vectors import (
    PolynomialWeightVector,
    UniformWeightVector,
    WeightVector,
)

_DEFAULT_LIMITS = CausalLimits()


def bind_weight_batch(plan: JointWeightPlan, vectors: tuple[WeightVector, ...]) -> None:
    if len(vectors) != plan.batch or any(len(v.draws) != plan.width for v in vectors):
        raise ValueError("realized weight batch geometry differs from its joint plan")
    for i, vector in enumerate(vectors):
        if plan.family == "uniform":
            if type(vector) is not UniformWeightVector:
                raise ValueError("uniform weight plan requires uniform realizations")
        elif (
            type(vector) is not PolynomialWeightVector
            or vector.decay != plan.decays[i]
            or vector.log_noise_scale != plan.scales[i]
        ):
            raise ValueError("realized polynomial weight differs from its frozen parameters")


def matrix_realization_parameters(plan: JointMatrixPlan) -> int:
    b, n, m = plan.batch, plan.rows, plan.columns
    if plan.family == "gaussian":
        raw = b * n * m
    elif plan.family == "importance":
        assert plan.weights is not None
        raw = b * (n * m + 1) + plan.weights.realization_parameter_count
    elif plan.family == "factorized":
        assert plan.weights is not None
        k = min(n, m)
        raw = b * (n * k + k * m) + plan.weights.realization_parameter_count
    elif plan.family == "kernel":
        assert plan.points is not None
        raw = plan.points.realization_parameter_count + b * ((n + m) * 3 + n * m + 1)
    elif plan.family == "transformed":
        assert plan.child is not None
        assert plan.activation is not None
        raw = matrix_realization_parameters(plan.child) + b * (
            plan.activation.activation.parameter_count + n * m + 1
        )
    else:
        raise ValueError("matrix realization branch is unavailable")
    return raw + b * (n * m + 2)


@dataclass(frozen=True, slots=True)
class JointMatrixRealization:
    plan: JointMatrixPlan
    matrices: tuple[NormalizedWeights, ...]
    point_cloud: KernelPointCloud | None = None
    version: str = "joint-matrix-realization-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "joint-matrix-realization-v1"
            or type(self.plan) is not JointMatrixPlan
            or type(self.matrices) is not tuple
            or len(self.matrices) != self.plan.batch
        ):
            raise ValueError("joint matrix realization requires its frozen plan and complete batch")
        for matrix in self.matrices:
            if (
                type(matrix) is not NormalizedWeights
                or matrix.shape != (self.plan.rows, self.plan.columns)
                or matrix.jitter_scale != 1e-6
                or matrix.epsilon != 1e-6
            ):
                raise ValueError("joint matrix wrapper must match source geometry and constants")
        kernel = self.plan.family == "kernel" or (
            self.plan.child is not None and self.plan.child.family == "kernel"
        )
        if (self.point_cloud is not None) != kernel:
            raise ValueError("only a kernel matrix batch owns a shared point cloud")
        if self.plan.family == "gaussian":
            if any(type(m.source) is not GaussianWeights for m in self.matrices):
                raise ValueError("Gaussian matrix plan requires Gaussian constructions")
        elif self.plan.family == "importance":
            vectors: list[WeightVector] = []
            for matrix in self.matrices:
                if type(matrix.source) is not ImportanceWeights or matrix.source.epsilon != 1e-6:
                    raise ValueError("importance matrix requires its inner source normalization")
                vectors.extend(matrix.source.importance)
            assert self.plan.weights is not None
            bind_weight_batch(self.plan.weights, tuple(vectors))
        elif self.plan.family == "factorized":
            vectors = []
            for matrix in self.matrices:
                if type(matrix.source) is not ImportanceFactorizedWeights:
                    raise ValueError("factorized matrix requires its Gaussian factors and vector")
                if len(matrix.source.diagonal.draws) != min(self.plan.rows, self.plan.columns):
                    raise ValueError("factorized matrix inner dimension differs from its plan")
                vectors.append(matrix.source.diagonal)
            assert self.plan.weights is not None
            bind_weight_batch(self.plan.weights, tuple(vectors))
        elif self.plan.family == "kernel":
            self._bind_cloud()
        elif self.plan.family == "transformed":
            assert self.plan.child is not None
            assert self.plan.activation is not None
            children: list[NormalizedWeights] = []
            for matrix in self.matrices:
                source = matrix.source
                if (
                    type(source) is not TransformedWeights
                    or source.jitter_scale != 1e-3
                    or source.activation != self.plan.activation.activation
                ):
                    raise ValueError(
                        "transformed matrix differs from its plain activation or jitter"
                    )
                children.append(source.child)
            JointMatrixRealization(self.plan.child, tuple(children), point_cloud=self.point_cloud)
        else:
            raise ValueError("matrix realization branch is unavailable")

    def _bind_cloud(self) -> None:
        plan = self.plan.points
        cloud = self.point_cloud
        assert plan is not None
        if (
            type(cloud) is not KernelPointCloud
            or cloud.shape != (plan.count, plan.dimension)
            or len(cloud.centers) != plan.components
            or cloud.law != "gaussian-mixture"
        ):
            raise ValueError("kernel cloud geometry or law differs from its point plan")
        bind_weight_batch(plan.mixture_weights, (cloud.component_weights,))
        bind_weight_batch(plan.factor_weights, cloud.factor_weights)
        points = cloud.materialize()
        n, m = self.plan.rows, self.plan.columns
        for i, matrix in enumerate(self.matrices):
            source = matrix.source
            segment = points[i * (n + m) : (i + 1) * (n + m)]
            if (
                type(source) is not LaplaceKernelWeights
                or source.point_construction is not None
                or source.left_points != segment[:n]
                or source.right_points != segment[n:]
                or source.distance_scale != self.plan.distance
            ):
                raise ValueError("kernel member differs from its shared cloud slice or distance")

    @property
    def parameter_count(self) -> int:
        return sum(m.parameter_count for m in self.matrices) + (
            self.point_cloud.parameter_count if self.point_cloud is not None else 0
        )

    def validate(self, decisions: JointSamplingRecord) -> None:
        self.plan.validate(decisions)

    def materialize(self) -> tuple[Matrix, ...]:
        return tuple(m.materialize() for m in self.matrices)


def admit_matrix_realization(plan: JointMatrixPlan, limits: CausalLimits) -> dict[str, int]:
    b, n, m = plan.batch, plan.rows, plan.columns
    return limits.admit_dimensions(
        (max(n, m),),
        ((),),
        rows=b * n,
        calibration_rows=0,
        features=m,
        outputs=m,
        extra_parameters=matrix_realization_parameters(plan),
        extra_working_channels=8 * (n + m),
    )


def materialize_joint_matrix(
    plan: JointMatrixPlan,
    decisions: JointSamplingRecord,
    *,
    rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT_LIMITS,
    capture: RawDrawLedger | None = None,
) -> JointMatrixRealization:
    if type(plan) is not JointMatrixPlan:
        raise ValueError("matrix realization requires a frozen joint plan")
    plan.validate(decisions)
    if type(rng) is not np.random.Generator or type(limits) is not CausalLimits:
        raise ValueError("matrix realization requires injected RNG and frozen limits")
    b, n, m = plan.batch, plan.rows, plan.columns
    admit_matrix_realization(plan, limits)
    with raw_transaction(rng, capture) as (staged, capture):
        sources: list[SimpleWeights | TransformedWeights] = []
        cloud = None
        if plan.family == "gaussian":
            sources = [
                GaussianWeights(frozen_matrix(a))
                for a in recorded_draw(
                    staged,
                    capture,
                    owner=plan.operation,
                    role="matrix-normals",
                    distribution="normal",
                    shape=(b, n, m),
                    parameters=(),
                )
            ]
        elif plan.family == "importance":
            assert plan.weights is not None
            weights = materialize_joint_weights(
                plan.weights, decisions, rng=staged, limits=limits, capture=capture
            )
            sources = [
                ImportanceWeights(frozen_matrix(a), weights[i * n : (i + 1) * n])
                for i, a in enumerate(
                    recorded_draw(
                        staged,
                        capture,
                        owner=plan.operation,
                        role="matrix-normals",
                        distribution="normal",
                        shape=(b, n, m),
                        parameters=(),
                    )
                )
            ]
        elif plan.family == "factorized":
            assert plan.weights is not None
            k = min(n, m)
            left = recorded_draw(
                staged,
                capture,
                owner=plan.operation,
                role="left-factor",
                distribution="normal",
                shape=(b, n, k),
                parameters=(),
            )
            weights = materialize_joint_weights(
                plan.weights, decisions, rng=staged, limits=limits, capture=capture
            )
            right = recorded_draw(
                staged,
                capture,
                owner=plan.operation,
                role="right-factor",
                distribution="normal",
                shape=(b, k, m),
                parameters=(),
            )
            sources = [
                ImportanceFactorizedWeights(frozen_matrix(u), frozen_matrix(v), w)
                for u, v, w in zip(left, right, weights, strict=True)
            ]
        elif plan.family == "kernel":
            assert plan.points is not None
            assert plan.distance is not None
            cloud = materialize_joint_points(
                plan.points, decisions, rng=staged, limits=limits, capture=capture
            )
            points = cloud.materialize()
            signs = np.sign(
                recorded_draw(
                    staged,
                    capture,
                    owner=plan.operation,
                    role="kernel-sign-normals",
                    distribution="normal",
                    shape=(b, n, m),
                    parameters=(),
                )
            )
            for i, sign in enumerate(signs):
                segment = points[i * (n + m) : (i + 1) * (n + m)]
                sources.append(
                    LaplaceKernelWeights(
                        left_points=segment[:n],
                        right_points=segment[n:],
                        signs=frozen_matrix(sign),
                        distance_scale=plan.distance,
                    )
                )
        elif plan.family == "transformed":
            assert plan.child is not None
            assert plan.activation is not None
            child = materialize_joint_matrix(
                plan.child, decisions, rng=staged, limits=limits, capture=capture
            )
            cloud = child.point_cloud
            sources = [
                TransformedWeights(
                    child=member,
                    activation=plan.activation.activation,
                    jitter=frozen_matrix(noise),
                )
                for member, noise in zip(
                    child.matrices,
                    recorded_draw(
                        staged,
                        capture,
                        owner=plan.operation,
                        role="transform-jitter",
                        distribution="normal",
                        shape=(b, n, m),
                        parameters=(),
                    ),
                    strict=True,
                )
            ]
        jitter = recorded_draw(
            staged,
            capture,
            owner=plan.operation,
            role="normalization-jitter",
            distribution="normal",
            shape=(b, n, m),
            parameters=(),
        )
        result = JointMatrixRealization(
            plan,
            tuple(
                NormalizedWeights(source=s, jitter=frozen_matrix(j))
                for s, j in zip(sources, jitter, strict=True)
            ),
            point_cloud=cloud,
        )
        result.validate(decisions)
        result.materialize()
        return result
