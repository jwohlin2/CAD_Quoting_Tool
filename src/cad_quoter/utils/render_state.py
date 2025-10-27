"""Render state helpers shared between quote rendering sections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping, TYPE_CHECKING

from cad_quoter.domain_models import coerce_float_or_none as _coerce_float_or_none
from cad_quoter.domain_models.values import safe_float as _safe_float
from cad_quoter.utils.render_utils.tables import draw_kv_table
from cad_quoter.utils.rendering import format_currency, format_hours

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from cad_quoter.pricing import QuoteConfiguration


@dataclass
class RenderState:
    """Mutable context shared by section renderers."""

    result: MutableMapping[str, Any]
    breakdown: MutableMapping[str, Any]
    currency: str
    show_zeros: bool
    page_width: int
    divider: str
    cfg: "QuoteConfiguration | None" = None
    geometry: Mapping[str, Any] | None = None
    lines: list[str] = field(default_factory=list)
    material_warning_needed: bool = False
    material_total_for_directs: float | None = None
    scrap_credit_for_directs: float | None = None
    material_tax_for_directs: float | None = None
    material_component_total: float | None = None
    material_component_net: float | None = None
    material_net_cost: float | None = None
    material_cost_components: Mapping[str, Any] | None = None

    append_line: Callable[[Any], None] | None = None
    append_lines: Callable[[Iterable[str]], None] | None = None
    write_wrapped: Callable[[str, str], None] | None = None

    def coerce_float(self, value: Any) -> float | None:
        """Best-effort conversion mirroring legacy helpers."""

        return _coerce_float_or_none(value)

    def safe_float(self, value: Any, default: float = 0.0) -> float:
        """Fallback float conversion that never raises."""

        return float(_safe_float(value, default))

    def format_currency(self, value: Any) -> str:
        return format_currency(value, self.currency)

    def render_kv_line(self, label: str, value_text: str, indent: str = "") -> str:
        left = f"{indent}{label}"
        right = value_text
        right_width = max(len(right), 1)
        pad = max(1, self.page_width - len(left) - len(right))
        left_width = len(left) + pad
        table_text = draw_kv_table(
            [(left, right)],
            left_width=left_width,
            right_width=right_width,
            left_align="L",
            right_align="R",
        )
        for line in table_text.splitlines():
            if line.startswith("|") and line.endswith("|"):
                body = line[1:-1]
                try:
                    left_segment, right_segment = body.split("|", 1)
                except ValueError:
                    continue
                return f"{left_segment}{right_segment}"
        return f"{left}{' ' * pad}{right}"

    def render_currency_row(self, label: str, value: float, indent: str = "") -> str:
        return self.render_kv_line(label, self.format_currency(value), indent)

    def render_hours_row(self, label: str, value: float, indent: str = "") -> str:
        return self.render_kv_line(label, format_hours(value), indent)

    def push(self, text: Any) -> None:
        if self.append_line is not None:
            self.append_line(text)
        else:
            self.lines.append(str(text))

    def extend(self, values: Iterable[str]) -> None:
        if self.append_lines is not None:
            self.append_lines(values)
        else:
            for value in values:
                self.push(value)
