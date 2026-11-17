import r0.modules.training


def test_training_entry_point_imports() -> None:
    assert r0.modules.training.__name__ == "r0.modules.training"
