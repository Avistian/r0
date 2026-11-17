import r0.modules.featurize


def test_featurize_entry_point_imports() -> None:
    assert r0.modules.featurize.__name__ == "r0.modules.featurize"
