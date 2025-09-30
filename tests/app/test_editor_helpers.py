import appV5


class DummyVar:
    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def test_update_material_price_field_uses_fallback(monkeypatch):
    choice_var = DummyVar("Custom Alloy")
    price_var = DummyVar("")
    material_lookup = {}

    monkeypatch.setattr(
        appV5,
        "_resolve_material_unit_price",
        lambda choice, unit="kg": (5.5, "backup_csv:material_price_backup.csv"),
    )

    changed = appV5._update_material_price_field(choice_var, price_var, material_lookup)

    assert changed is True
    assert price_var.get() == "0.0055"
