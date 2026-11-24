"""Authored observations, not draws from the production training prior."""

from dataclasses import dataclass
from typing import Literal

from r0.modules.episodes import Episode, Feature, Row

Family = Literal["single-feature-lookup", "single-feature-affine"]
Shortcut = Literal["fixed-position", "fixed-label", "range-clipping"]


@dataclass(frozen=True)
class ChallengePair:
    name: str
    family: Family
    shortcut: Shortcut
    transformation: str
    before: Episode
    after: Episode
    derivation: str


def public_pairs() -> tuple[ChallengePair, ...]:
    """Return fresh immutable public cases. The sealed case is never constructed here."""
    features = (Feature("a", "binary"), Feature("b", "binary"))
    context = tuple(
        Row(f"c{i}", values) for i, values in enumerate(((0, 0), (0, 1), (1, 0), (1, 1)))
    )
    queries = (Row("q0", (0, 1)), Row("q1", (1, 0)))
    before = Episode(
        "permutation-v1",
        "permutation-before-v1",
        "binary",
        features,
        context,
        ("low", "low", "high", "high"),
        queries,
        ("low", "high"),
        ("low", "high"),
        "authored vocabulary, independent of query targets",
    )
    after = Episode(
        before.source_task_id,
        "permutation-after-v1",
        "binary",
        features[::-1],
        tuple(Row(r.id, r.values[::-1]) for r in context),
        before.context_targets,
        tuple(Row(r.id, r.values[::-1]) for r in queries),
        ("low", "high"),
        before.classes,
        before.class_provenance,
    )
    return (
        ChallengePair(
            "column-permutation",
            "single-feature-lookup",
            "fixed-position",
            "Reverse context/query columns and schema; preserve answers.",
            before,
            after,
            "Context has both b values for each a: a=0 maps to low, a=1 to high. "
            "Query a values are 0,1 on both sides: low,high.",
        ),
        _remapping(),
        _extrapolation(),
    )


def _remapping() -> ChallengePair:
    features = (Feature("symbol", "categorical"),)
    context = tuple(Row(f"c{i}", (value,)) for i, value in enumerate(("A", "B", "C")))
    queries = tuple(Row(f"q{i}", (value,)) for i, value in enumerate(("A", "B", "C")))
    classes = ("red", "green", "blue")
    before = Episode(
        "mapping-v1",
        "mapping-before-v1",
        "multiclass",
        features,
        context,
        classes,
        queries,
        classes,
        classes,
        "fixed declared vocabulary",
    )
    after = Episode(
        before.source_task_id,
        "mapping-after-v1",
        "multiclass",
        features,
        context,
        ("blue", "red", "green"),
        queries,
        ("blue", "red", "green"),
        classes,
        before.class_provenance,
    )
    return ChallengePair(
        "context-remapping",
        "single-feature-lookup",
        "fixed-label",
        "Cyclically rename targets, preserving vocabulary and input rows.",
        before,
        after,
        "Each query symbol occurs in context. Before: A=red,B=green,C=blue. "
        "After: A=blue,B=red,C=green. Read the matching context label.",
    )


def _extrapolation() -> ChallengePair:
    features = (Feature("x", "continuous"),)
    context = (Row("c0", (0.0,)), Row("c1", (1.0,)), Row("c2", (2.0,)))
    before = Episode(
        "affine-v1",
        "affine-before-v1",
        "regression",
        features,
        context,
        (1.0, 3.0, 5.0),
        (Row("q0", (0.5,)), Row("q1", (1.5,))),
        (2.0, 4.0),
    )
    after = Episode(
        before.source_task_id,
        "affine-after-v1",
        "regression",
        features,
        context,
        before.context_targets,
        (Row("q0", (-2.0,)), Row("q1", (3.0,))),
        (-3.0, 7.0),
    )
    return ChallengePair(
        "affine-extrapolation",
        "single-feature-affine",
        "range-clipping",
        "Keep context fixed; move queries outside the context range.",
        before,
        after,
        "The affine intercept is 1 at x=0; slope is (3-1)/(1-0)=2. "
        "At x=0.5,1.5 answers are 2,4; at x=-2,3 answers are -3,7. "
        "Clipping to observed target range [1,5] gives 1,5 outside support.",
    )
