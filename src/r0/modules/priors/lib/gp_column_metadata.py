"""Pure scalar/shape validation for finite GP archive envelopes."""

from typing import Any

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.gp_column_decoding import gp_fields, gp_payload_equal
from r0.modules.priors.lib.gp_column_draw import GPColumnDraw
from r0.modules.priors.lib.gp_column_population import GPColumnRequest
from r0.modules.priors.lib.gp_column_target import GPColumnTarget


def gp_sequence(value: Any, count: int | None = None) -> None:
    if type(value) not in (tuple, list) or (count is not None and len(value) != count):
        raise ValueError("GP archive sequence geometry differs from replay")


def gp_vector_metadata(value: Any, count: int) -> None:
    gp_sequence(value, count)
    if any(not finite_number(v) for v in value):
        raise ValueError("GP archive vector requires finite numbers")


def gp_matrix_metadata(value: Any, rows: int, columns: int) -> None:
    gp_sequence(value, rows)
    for row in value:
        gp_vector_metadata(row, columns)


def gp_target_metadata(data: Any) -> GPColumnTarget:
    target = gp_fields(data, GPColumnTarget)
    for key in ("weights", "hidden"):
        gp_sequence(target[key])
        for row in target[key]:
            gp_sequence(row)
        target[key] = tuple(tuple(row) for row in target[key])
    for key in ("bias", "hidden_bias"):
        gp_sequence(target[key])
        target[key] = tuple(target[key])
    gp_sequence(target["quadratic"])
    for matrix in target["quadratic"]:
        gp_sequence(matrix)
        for row in matrix:
            gp_sequence(row)
    target["quadratic"] = tuple(
        tuple(tuple(row) for row in matrix) for matrix in target["quadratic"]
    )
    return GPColumnTarget(**target)


def gp_draw_metadata(data: Any, request: GPColumnRequest) -> None:
    from dataclasses import asdict

    draw = gp_fields(data, GPColumnDraw)
    gp_matrix_metadata(draw["locations"], request.rows, request.dimension)
    expected_locations = (
        request.locations
        if request.location_family == "fixed-r0"
        else tuple((float(value),) for value in np.linspace(-1, 1, request.rows))
        if request.location_family == "grid-r0"
        else None
    )
    if expected_locations is not None and any(
        tuple(row) != expected
        for row, expected in zip(draw["locations"], expected_locations, strict=True)
    ):
        raise ValueError("GP retained locations differ from declared fixed/grid coordinates")
    for key in ("innovations", "values"):
        gp_matrix_metadata(draw[key], request.rows, request.columns)
    gp_sequence(draw["kernels"], len(request.kernels))
    if not gp_payload_equal(tuple(asdict(k) for k in request.kernels), draw["kernels"]):
        raise ValueError("GP draw kernels differ from request")
    for key in ("covariance", "factors"):
        gp_sequence(draw[key], len(request.kernels))
        for matrix in draw[key]:
            gp_matrix_metadata(matrix, request.rows, request.rows)
