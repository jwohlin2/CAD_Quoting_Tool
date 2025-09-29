from __future__ import annotations

from dataclasses import dataclass

from cad_quoter.app.container import ServiceContainer


@dataclass
class DummyEngine:
    cleared: bool = False

    def clear_cache(self) -> None:
        self.cleared = True


def test_service_container_caches_pricing_engine() -> None:
    factory_calls = []

    def factory() -> DummyEngine:
        engine = DummyEngine()
        factory_calls.append(engine)
        return engine

    container = ServiceContainer(
        load_params=lambda: {"ok": True},
        load_rates=lambda: {"rate": 1.0},
        pricing_engine_factory=factory,
    )

    first = container.get_pricing_engine()
    second = container.get_pricing_engine()

    assert first is second
    assert len(factory_calls) == 1


def test_service_container_create_pricing_engine_returns_new_instance() -> None:
    container = ServiceContainer(
        load_params=lambda: {},
        load_rates=lambda: {},
        pricing_engine_factory=DummyEngine,
    )

    assert container.create_pricing_engine() is not container.create_pricing_engine()
