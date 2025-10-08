from __future__ import annotations

import appV5


def test_outsourced_pass_hidden_without_geo_or_cost() -> None:
    assert not appV5._should_include_outsourced_pass(0.0, {})


def test_outsourced_pass_shown_when_cost_present() -> None:
    assert appV5._should_include_outsourced_pass(12.5, {})


def test_outsourced_pass_shown_when_geo_lists_finishes() -> None:
    geo = {"finishes": ["Anodize Type II"]}
    assert appV5._should_include_outsourced_pass(0.0, geo)


def test_outsourced_pass_shown_when_nested_geo_has_finish_flags() -> None:
    geo = {"geo": {"finish_flags": ["PASSIVATION"]}}
    assert appV5._should_include_outsourced_pass(0.0, geo)


def test_outsourced_pass_hidden_for_blank_finish_entries() -> None:
    geo = {"finishes": ["", None, "  "], "finish_flags": []}
    assert not appV5._should_include_outsourced_pass(0.0, geo)
