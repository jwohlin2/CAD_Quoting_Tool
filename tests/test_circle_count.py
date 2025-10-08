from __future__ import annotations

from types import SimpleNamespace

from appV5 import classify_concentric, hole_count_from_geometry


class FakeCircle:
    def __init__(self, center: tuple[float, float, float], radius: float, layer: str = "0") -> None:
        self.dxf = SimpleNamespace(center=center, radius=radius, layer=layer)


class FakeSpace:
    def __init__(self, circles: list[FakeCircle] | None = None) -> None:
        self._circles = circles or []

    def query(self, types: str):  # pragma: no cover - trivial wrapper
        if types == "CIRCLE":
            return list(self._circles)
        return []


class FakeLayouts:
    def __init__(self, spaces: dict[str, FakeSpace] | None = None) -> None:
        self._spaces = spaces or {}

    def names_in_taborder(self):  # pragma: no cover - trivial wrapper
        return list(self._spaces)

    def get(self, name: str):  # pragma: no cover - trivial wrapper
        return SimpleNamespace(entity_space=self._spaces[name])


class FakeDoc:
    def __init__(
        self,
        model_circles: list[FakeCircle] | None = None,
        layout_circles: list[list[FakeCircle]] | None = None,
    ) -> None:
        self._model = FakeSpace(model_circles)
        layouts = {
            f"Layout{i}": FakeSpace(circles)
            for i, circles in enumerate(layout_circles or [])
        }
        self.layouts = FakeLayouts(layouts)

    def modelspace(self):  # pragma: no cover - trivial wrapper
        return self._model


def test_hole_count_prefers_modelspace_over_layout_duplicates() -> None:
    doc = FakeDoc(
        model_circles=[FakeCircle((0.0, 0.0, 0.0), 0.25)],
        layout_circles=[[FakeCircle((10.0, 5.0, 0.0), 0.25)]],
    )

    count, families = hole_count_from_geometry(doc, to_in=1.0)

    assert count == 1
    assert families == {0.5: 1}


def test_hole_count_uses_layout_when_modelspace_empty() -> None:
    doc = FakeDoc(
        model_circles=[],
        layout_circles=[[FakeCircle((0.0, 0.0, 0.0), 0.2), FakeCircle((1.0, 0.0, 0.0), 0.2)]],
    )

    count, families = hole_count_from_geometry(doc, to_in=1.0)

    assert count == 2
    assert families == {0.4: 2}


def test_classify_concentric_ignores_duplicate_layout_views() -> None:
    doc = FakeDoc(
        model_circles=[
            FakeCircle((0.0, 0.0, 0.0), 0.5),
            FakeCircle((0.0, 0.0, 0.0), 0.65),
        ],
        layout_circles=[[FakeCircle((10.0, 10.0, 0.0), 0.5), FakeCircle((10.0, 10.0, 0.0), 0.65)]],
    )

    result = classify_concentric(doc, to_in=1.0)

    total_pairs = result.get("cbore_pairs_geom", 0) + result.get("csk_pairs_geom", 0)
    assert total_pairs == 1
