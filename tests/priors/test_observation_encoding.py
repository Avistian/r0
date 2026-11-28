"""Fixed observation encodings preserve declared boundaries and category meaning."""

from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize("equality", ["lower", "upper"])
def test_bins_against_scalar_boundary_oracle(equality: str) -> None:
    thresholds = (-1.0, 0.0, 2.0)
    points = [v for t in thresholds for v in (np.nextafter(t, -np.inf), t, np.nextafter(t, np.inf))]
    raw = np.column_stack((points, np.arange(len(points), dtype=np.float64)))
    before = raw.copy()
    encoding = p.ColumnEncoding(0, "ordinal", thresholds, ("z", "b", "a", "q"), equality)
    result = p.encode_observation_columns((encoding,), raw)
    expected = [sum(x > t if equality == "lower" else x >= t for t in thresholds) for x in points]
    np.testing.assert_array_equal(result.values[:, 0], expected)
    np.testing.assert_array_equal(result.values[:, 1], raw[:, 1])
    np.testing.assert_array_equal(raw, before)
    assert result.schema[0].vocabulary == ("z", "b", "a", "q")
    assert result.schema[1] == p.TrainingFeature()
    assert [result.schema[0].decode(float(v)) for v in expected] == [
        encoding.vocabulary[v] for v in expected
    ]


@pytest.mark.parametrize(
    ("rounding", "expected"),
    [("floor", [0, 0, 1, 2, 3]), ("nearest", [0, 0, 2, 2, 4]), ("ceil", [0, 1, 2, 3, 4])],
)
def test_count_rounding(rounding: str, expected: list[int]) -> None:
    raw = np.array([[-9.0], [1.0], [3.0], [5.0], [7.0]])
    result = p.encode_observation_columns(
        (p.ColumnEncoding(0, "count", count_rounding=rounding, count_scale=2),), raw
    )
    np.testing.assert_array_equal(result.values[:, 0], expected)
    assert [result.schema[0].decode(v) for v in result.values[:, 0]] == expected


def test_absent_and_singleton_categories_and_binding() -> None:
    raw = np.array([[-10.0, 5.0], [-9.0, 5.0]])
    encodings = (
        p.ColumnEncoding(0, "binary", (0.0,), ("no", "yes")),
        p.ColumnEncoding(1, "categorical", (), ("only",)),
    )
    result = p.encode_observation_columns(encodings, raw)
    np.testing.assert_array_equal(result.values, np.zeros((2, 2)))
    assert result.schema[0].vocabulary == ("no", "yes")
    assert result.schema[1].decode(0) == "only"
    changed = raw.copy()
    changed[0, 0] -= 1
    assert p.encode_observation_columns(encodings, changed).binding != result.binding
    relabeled = (replace(encodings[0], vocabulary=("off", "on")), encodings[1])
    assert p.encode_observation_columns(relabeled, raw).binding != result.binding


def test_invalid_controls_source_schema_and_limits() -> None:
    encoding = p.ColumnEncoding(0, "count")
    raw = np.ones((3, 1))
    with pytest.raises(ValueError, match="bounded mappings"):
        p.encode_observation_columns((encoding, encoding), raw)
    with pytest.raises(ValueError, match="numeric source"):
        p.encode_observation_columns(
            (encoding,), raw, schema=(p.TrainingFeature("binary", ("a", "b")),)
        )
    with pytest.raises(ValueError, match="workspace|storage limit"):
        p.encode_observation_columns((encoding,), raw, limits=p.CausalLimits(working_bytes=1))
    with pytest.raises(ValueError, match="finite"):
        p.encode_observation_columns((encoding,), np.array([[np.nan]]))
    with pytest.raises(ValueError, match="overflow"):
        p.encode_observation_columns((replace(encoding, count_scale=1e-300),), np.array([[1e300]]))
    with pytest.raises(ValueError, match="invalid bounded"):
        p.ColumnEncoding(0, "ordinal", (1.0, 1.0), ("a", "b", "c"))
    with pytest.raises(ValueError, match="vocabulary"):
        p.ColumnEncoding(0, "binary", (), ("a",))
