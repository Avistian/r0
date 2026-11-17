import r0.modules.model


def test_model_entry_point_imports() -> None:
    assert r0.modules.model.__name__ == "r0.modules.model"
