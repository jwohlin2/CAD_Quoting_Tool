from __future__ import annotations

import pytest

from cad_quoter.utils.render_utils import (
    QuoteDocRecorder,
    fmt_hours,
    fmt_money,
    fmt_percent,
    fmt_range,
    format_currency,
    format_dimension,
    format_hours,
    format_hours_with_rate,
    format_percent,
    format_weight_lb_decimal,
    format_weight_lb_oz,
    render_quote_doc,
)


@pytest.mark.parametrize(
    "value,currency,expected",
    [
        (1234.567, "$", "$1,234.57"),
        ("99.9", "€", "€99.90"),
        (None, "£", "£0.00"),
    ],
)
def test_format_currency(value, currency, expected):
    assert format_currency(value, currency) == expected


@pytest.mark.parametrize(
    "value,currency,expected",
    [
        (1234.567, "$", "$1,234.57"),
        ("99.9", "€", "€99.90"),
        (None, "£", "£0.00"),
    ],
)
def test_fmt_money(value, currency, expected):
    assert fmt_money(value, currency) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (2, "2.00 hr"),
        ("1.5", "1.50 hr"),
        (None, "0.00 hr"),
    ],
)
def test_format_hours(value, expected):
    assert format_hours(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (2, "2.00"),
        ("1.5", "1.50"),
        (None, "0.00"),
    ],
)
def test_fmt_hours_without_unit(value, expected):
    assert fmt_hours(value, include_unit=False) == expected


@pytest.mark.parametrize(
    "hours,rate,currency,expected",
    [
        (1.5, 25, "$", "1.50 hr @ $25.00/hr"),
        (0, 0, "$", "0.00 hr @ —/hr"),
        ("2", None, "€", "2.00 hr @ —/hr"),
    ],
)
def test_format_hours_with_rate(hours, rate, currency, expected):
    assert format_hours_with_rate(hours, rate, currency) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.125, "12.5%"),
        (None, "0.0%"),
    ],
)
def test_format_percent(value, expected):
    assert format_percent(value) == expected


@pytest.mark.parametrize(
    "value,decimals,expected",
    [
        (0.125, 1, "12.5%"),
        (0.125, 2, "12.50%"),
        (None, 0, "0%"),
    ],
)
def test_fmt_percent(value, decimals, expected):
    assert fmt_percent(value, decimals=decimals) == expected


@pytest.mark.parametrize(
    "lower,upper,unit,expected",
    [
        (1.0, 2.5, "hr", "1.0–2.5 hr"),
        ("A", "C", None, "A–C"),
    ],
)
def test_fmt_range(lower, upper, unit, expected):
    result = fmt_range(lower, upper, formatter=str, unit=unit)
    assert result == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, "1"),
        (1.25, "1.25"),
        (1.23456, "1.235"),
        (None, "—"),
        ("  ", "—"),
        (" custom ", "custom"),
    ],
)
def test_format_dimension(value, expected):
    assert format_dimension(value) == expected


@pytest.mark.parametrize(
    "grams,expected",
    [
        (0, "0.00 lb"),
        (453.59237, "1 lb"),
        (907.18474, "2 lb"),
    ],
)
def test_format_weight_lb_decimal(grams, expected):
    assert format_weight_lb_decimal(grams) == expected


@pytest.mark.parametrize(
    "grams,expected",
    [
        (0, "0 oz"),
        (453.59237, "1 lb"),
        (226.796185, "8 oz"),
        (680.388555, "1 lb 8 oz"),
    ],
)
def test_format_weight_lb_oz(grams, expected):
    assert format_weight_lb_oz(grams) == expected


class TestQuoteDocRecorder:
    def test_records_sections_and_renders(self) -> None:
        divider = "-" * 5
        recorder = QuoteDocRecorder(divider)
        raw_lines = [
            "Main Title",
            divider,
            "intro line",
            "intro follow-up",
            "Section A",
            divider,
            "row a1",
            "row a2",
            "Section B",
            divider,
            "row b1",
        ]
        for index, line in enumerate(raw_lines):
            previous = raw_lines[index - 1] if index else None
            recorder.observe_line(index, line, previous)
        recorder.replace_line(3, "intro updated")
        doc = recorder.build_doc()

        assert doc.title == "Main Title"
        assert [section.title for section in doc.sections] == [
            "Main Title",
            "Section A",
            "Section B",
        ]
        assert [row.text for row in doc.sections[0].rows] == [
            "intro line",
            "intro updated",
        ]
        assert [row.text for row in doc.sections[1].rows] == ["row a1", "row a2"]
        assert [row.text for row in doc.sections[2].rows] == ["row b1"]

        rendered = render_quote_doc(doc, divider=divider)
        assert rendered.splitlines() == [
            "Main Title",
            divider,
            "intro line",
            "intro updated",
            "Section A",
            divider,
            "row a1",
            "row a2",
            "Section B",
            divider,
            "row b1",
        ]

    def test_empty_sections_removed(self) -> None:
        divider = "=" * 3
        recorder = QuoteDocRecorder(divider)
        lines = [
            "Header",
            "body line",
            "lonely title",
            divider,
        ]
        for index, line in enumerate(lines):
            previous = lines[index - 1] if index else None
            recorder.observe_line(index, line, previous)
        doc = recorder.build_doc()

        assert doc.title == "Header"
        assert [section.title for section in doc.sections] == [None, "lonely title"]
        assert [row.text for row in doc.sections[0].rows] == ["body line"]
        assert doc.sections[1].rows == []

        rendered = render_quote_doc(doc, divider=divider)
        assert rendered.splitlines() == [
            "Header",
            "body line",
            "lonely title",
            divider,
        ]
