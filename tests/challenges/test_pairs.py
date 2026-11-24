from dataclasses import replace

from r0.modules.challenges import public_pairs, reference_predictions, shortcut_predictions


def test_column_permutation_preserves_answers_and_exposes_position_shortcut():
    pair = public_pairs()[0]
    assert pair.name == "column-permutation"
    assert tuple(f.name for f in pair.before.features) == ("a", "b")
    assert tuple(f.name for f in pair.after.features) == ("b", "a")
    for episode in (pair.before, pair.after):
        assert reference_predictions(episode.predictor_view(), pair.family) == ("low", "high")
    assert shortcut_predictions(pair.before.predictor_view(), pair.shortcut, pair.family) == (
        "low",
        "high",
    )
    assert shortcut_predictions(pair.after.predictor_view(), pair.shortcut, pair.family) == (
        "high",
        "low",
    )
    changed = replace(pair.after, query_targets=("high", "low"))
    assert reference_predictions(changed.predictor_view(), pair.family) == ("low", "high")


def test_context_mapping_changes_answers_and_exposes_fixed_labels():
    pair = public_pairs()[1]
    assert pair.name == "context-remapping"
    assert reference_predictions(pair.before.predictor_view(), pair.family) == (
        "red",
        "green",
        "blue",
    )
    assert reference_predictions(pair.after.predictor_view(), pair.family) == (
        "blue",
        "red",
        "green",
    )
    assert shortcut_predictions(pair.after.predictor_view(), pair.shortcut, pair.family) == (
        "red",
        "green",
        "blue",
    )
    assert pair.before.queries == pair.after.queries
    assert pair.before.classes == pair.after.classes


def test_affine_extrapolation_changes_only_queries_and_defeats_clipping():
    pair = public_pairs()[2]
    assert pair.name == "affine-extrapolation"
    assert pair.before.context == pair.after.context
    assert pair.before.context_targets == pair.after.context_targets
    assert reference_predictions(pair.before.predictor_view(), pair.family) == (2.0, 4.0)
    assert reference_predictions(pair.after.predictor_view(), pair.family) == (-3.0, 7.0)
    assert shortcut_predictions(pair.before.predictor_view(), pair.shortcut, pair.family) == (
        2.0,
        4.0,
    )
    assert shortcut_predictions(pair.after.predictor_view(), pair.shortcut, pair.family) == (
        1.0,
        5.0,
    )


def test_references_refuse_unidentifiable_rules():
    import pytest

    from r0.modules.episodes import Row

    pair = public_pairs()[0]
    view = pair.before.predictor_view()
    ambiguous = replace(
        view, context=(Row("c0", (0, 0)), Row("c1", (1, 1))), context_targets=("low", "high")
    )
    with pytest.raises(ValueError, match="exactly one"):
        reference_predictions(ambiguous, pair.family)
    affine = public_pairs()[2]
    constant = replace(
        affine.before.predictor_view(), context=(Row("c0", (0.0,)),), context_targets=(1.0,)
    )
    with pytest.raises(ValueError, match="slope"):
        reference_predictions(constant, affine.family)


def test_all_pairs_keep_query_targets_out_of_both_predictors():
    for pair in public_pairs():
        episode = pair.after
        targets = (
            tuple(999.0 for _ in episode.queries)
            if episode.task_type == "regression"
            else tuple(episode.classes[0] for _ in episode.queries)
        )
        altered = replace(episode, query_targets=targets)
        assert altered.predictor_view() == episode.predictor_view()
        assert reference_predictions(
            altered.predictor_view(), pair.family
        ) == reference_predictions(episode.predictor_view(), pair.family)
        assert shortcut_predictions(
            altered.predictor_view(), pair.shortcut, pair.family
        ) == shortcut_predictions(episode.predictor_view(), pair.shortcut, pair.family)


def test_pair_report_retains_failed_expected_answers():
    from r0.modules.challenges import evaluate_pair

    pair = public_pairs()[2]
    report = evaluate_pair(pair)
    assert report["status"] == "PASS"
    bad = replace(pair, after=replace(pair.after, query_targets=(0.0, 0.0)))
    report = evaluate_pair(bad)
    assert report["status"] == "FAIL"
    assert report["after"]["reference"]["matches"] == [False, False]
    assert report["after"]["reference"]["predictions"] == [-3.0, 7.0]


def test_query_support_cannot_resolve_ambiguous_context():
    import pytest

    from r0.modules.episodes import Row

    pair = public_pairs()[0]
    view = replace(
        pair.before.predictor_view(),
        context=(Row("c0", (0, 0)),),
        context_targets=("low",),
        queries=(Row("q0", (0, 1)),),
    )
    with pytest.raises(ValueError, match="exactly one"):
        reference_predictions(view, pair.family)


def test_shortcut_receives_the_same_declared_family_as_reference():
    pair = public_pairs()[0]
    assert shortcut_predictions(pair.before.predictor_view(), pair.shortcut, pair.family) == (
        "low",
        "high",
    )


def test_invalid_context_retains_raw_case_and_failure_reason():
    from r0.modules.challenges import evaluate_pair
    from r0.modules.episodes import Row

    pair = public_pairs()[2]
    invalid = replace(
        pair, after=replace(pair.after, context=(Row("c0", (0.0,)),), context_targets=(1.0,))
    )
    report = evaluate_pair(invalid)
    assert report["status"] == "FAIL"
    assert report["before"]["reference"]["predictions"] == [2.0, 4.0]
    assert "slope" in report["after"]["errors"][0]
    assert report["after"]["expected_answers"] == [-3.0, 7.0]
