from __future__ import annotations

from types import MappingProxyType

from cad_quoter.render import RenderState
from cad_quoter.render.buckets import detect_planner_drilling, has_planner_drilling


def test_detect_planner_drilling_direct_match() -> None:
    buckets = {"drilling": {"minutes": 12}}

    assert detect_planner_drilling(buckets) is True


def test_detect_planner_drilling_nested_structure() -> None:
    buckets = {
        "milling": {},
        "buckets": {
            "turning": {},
            "drilling": {"minutes": 7},
        },
    }

    assert detect_planner_drilling(buckets) is True


def test_detect_planner_drilling_self_reference_guard() -> None:
    buckets: dict[str, object] = {}
    buckets["buckets"] = buckets

    assert detect_planner_drilling(buckets) is False


def test_detect_planner_drilling_non_mapping() -> None:
    assert detect_planner_drilling([{"drilling": 1}]) is False


def test_has_planner_drilling_uses_bucket_view() -> None:
    payload = {
        "breakdown": {
            "bucket_view": {
                "buckets": {
                    "drilling": {"minutes": 3},
                }
            }
        }
    }

    state = RenderState(payload)

    assert has_planner_drilling(state) is True


def test_has_planner_drilling_mapping_input() -> None:
    breakdown = MappingProxyType(
        {
            "planner_buckets": {
                "buckets": {
                    "drilling": {"minutes": 4},
                }
            }
        }
    )

    assert has_planner_drilling(breakdown) is True


def test_has_planner_drilling_negative_case() -> None:
    payload = {"breakdown": {"planner_buckets": {"milling": {}}}}

    state = RenderState(payload)

    assert has_planner_drilling(state) is False
