"""Finite source-catalog support bounds; no random choices or array materialization."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_matrix_plans import JointMatrixPrior


@dataclass(frozen=True, slots=True)
class MechanismSupport:
    numbers: int = 0
    draws: int = 0
    named_calls: int = 0
    named_values: int = 0
    choices: int = 0
    channels: int = 0
    floor: int = 1
    width: int = 1
    categorical: int = 0

    def add(self, other: "MechanismSupport") -> "MechanismSupport":
        return MechanismSupport(
            self.numbers + other.numbers,
            self.draws + other.draws,
            self.named_calls + other.named_calls,
            self.named_values + other.named_values,
            self.choices + other.choices,
            self.channels + other.channels,
            max(self.floor, other.floor),
            max(self.width, other.width),
            self.categorical + other.categorical,
        )


def support_max(values: tuple[MechanismSupport, ...]) -> MechanismSupport:
    return MechanismSupport(
        *(
            max(getattr(v, k) for v in values)
            for k in (
                "numbers",
                "draws",
                "named_calls",
                "named_values",
                "choices",
                "channels",
                "floor",
                "width",
                "categorical",
            )
        )
    )


def _weights(width: int, batch: int, prior: JointMatrixPrior) -> MechanismSupport:
    if prior.vector_family == "uniform":
        return MechanismSupport(batch * width, batch)
    calls = 2 * batch if prior.scalarization == "scalarized-weight-batches-v1" else 2
    return MechanismSupport(batch * (2 * width + 2), 2 * batch, calls, 2 * batch)


def matrix_support(
    rows: int, columns: int, batch: int, prior: JointMatrixPrior, *, simple: bool = False
) -> MechanismSupport:
    """All five matrix branches, including transformed→simple and 15-component 3D points."""
    n, m, b = rows, columns, batch
    catalog = prior.simple_families if simple else prior.families
    branches: list[MechanismSupport] = []
    # Every matrix retains raw construction, normalization and a dense equation copy.
    common = MechanismSupport(
        4 * b * n * m + 8 * b, b + 2, 1, 1, len(catalog), 16 * (n + m), max(n, m), categorical=1
    )
    for family in catalog:
        if family == "gaussian":
            branch = MechanismSupport(b * n * m, 1)
        elif family == "importance":
            branch = _weights(m, b * n, prior).add(MechanismSupport(b * (n * m + 1), 2))
        elif family == "factorized":
            k = min(n, m)
            branch = _weights(k, b, prior).add(MechanismSupport(b * k * (n + m), 2))
        elif family == "kernel":
            # Gaussian-mixture cloud: centers, 3x3 factors, weights, labels and innovations.
            count, c, d = b * (n + m), 15, 3
            branch = (
                _weights(c, 1, prior)
                .add(_weights(d, c, prior))
                .add(
                    MechanismSupport(
                        c * d + c * d * d + count * (d + 1) + 2 + c + b * ((n + m) * 3 + n * m + 1),
                        8,
                        2,
                        2,
                        0,
                        16 * count + 64,
                        count,
                    )
                )
            )
        else:
            # Source transformed matrices terminate in a simple catalog; one plain activation.
            branch = matrix_support(n, m, b, prior, simple=True).add(
                MechanismSupport(
                    b * (n * m + 4), 2, 2, 2, 64, 8 * (n + m), max(n, m), categorical=2
                )
            )
        branches.append(common.add(branch))
    return replace(
        support_max(tuple(branches)),
        width=max(
            n,
            m,
            3
            if "kernel" in catalog
            or ("transformed" in catalog and "kernel" in prior.simple_families)
            else 1,
        ),
    )


def function_support(
    prior: JointFunctionPrior, width: int, rows: int, *, cheap: bool = False
) -> MechanismSupport:
    selector = prior.request(cheap=cheap)
    branches: list[MechanismSupport] = []
    for slot in selector.choices:
        family = prior.family(slot, cheap=cheap)
        common = MechanismSupport(
            32, 0, 1, 1, len(selector.choices), 16 * width, rows, categorical=1
        )
        if family == "lin":
            branch = matrix_support(width, width, 1, prior.matrix)
        elif family == "quad":
            k = min(width, 20) + 1
            branch = matrix_support(k, k, width, prior.matrix).add(
                MechanismSupport(width + 1 + width * k * k, 1, channels=16 * width * k)
            )
        elif family == "gp":
            k = prior.frequencies
            branch = _weights(width, 1, prior.matrix).add(
                MechanismSupport(
                    4 * k * (2 * width + 1) + width * width,
                    8,
                    3,
                    3,
                    2,
                    16 * k + 16 * width,
                    max(rows, k),
                    categorical=1,
                )
            )
        elif family == "tree":
            trees, depth = 128, 7
            branch = MechanismSupport(
                rows * width + width + 4 * trees * depth + 2 * trees * 2**depth * width,
                6,
                2,
                2,
                0,
                16 * trees * (depth + width) + 16 * width,
                rows,
            )
        elif family == "disc":
            centers = min(
                rows, prior.discretization_centers or prior.max_discretization_cardinality
            )
            branch = matrix_support(width, width, 1, prior.matrix).add(
                MechanismSupport(
                    rows * (width + 1) + centers * (width + width) + width * width + 1,
                    2,
                    2,
                    2,
                    0,
                    32 * centers * width + 24 * centers + 32 * width,
                    rows,
                )
            )
        elif family == "em":
            count = max(16, 2 * width)
            branch = matrix_support(width, count, 1, prior.matrix).add(
                MechanismSupport(
                    rows * width + count * (width + 2) + rows * count + count * (2 * width + 1) + 2,
                    8,
                    4,
                    4,
                    0,
                    32 * count * width + 24 * count + 32 * width,
                    rows,
                )
            )
        elif family == "mlp":
            hidden = 127
            branch = (
                matrix_support(hidden, width, 1, prior.matrix)
                .add(matrix_support(hidden, hidden, 1, prior.matrix))
                .add(matrix_support(width, hidden, 1, prior.matrix))
                .add(
                    MechanismSupport(
                        rows * (4 * width + 4 * hidden) + 24 * (width + hidden) + 32,
                        4,
                        16,
                        16,
                        256,
                        48 * (2 * width + 2 * hidden),
                        max(rows, hidden),
                        categorical=10,
                    )
                )
            )
            one_layer = matrix_support(width, width, 1, prior.matrix).add(
                MechanismSupport(
                    4 * rows * width + 24 * width + 32,
                    4,
                    16,
                    16,
                    256,
                    96 * width,
                    rows,
                    categorical=10,
                )
            )
            branch = support_max((branch, one_layer))
        else:
            # A product has exactly two source-eligible non-product children; no recursion.
            child = function_support(prior.child_prior(slot), width, rows, cheap=True)
            branch = child.add(child).add(
                MechanismSupport(
                    rows * width + 2 * child.numbers, 1, channels=8 * width, floor=rows
                )
            )
        branches.append(common.add(branch))
    return support_max(tuple(branches))
