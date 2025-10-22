"""Optional third-party integrations used by the quoting UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional
import typing

try:  # pragma: no cover - optional dependency
    import pandas as _pd  # type: ignore[import]
except Exception:  # pragma: no cover - pandas is optional
    _pd = None  # type: ignore[assignment]

pd = typing.cast(typing.Any, _pd)

try:  # pragma: no cover - optional DXF enrichment hook
    from cad_quoter.geometry.dxf_enrich import build_geo_from_dxf as _build_geo_from_dxf_path
except Exception:  # pragma: no cover - defensive fallback
    _build_geo_from_dxf_path = None  # type: ignore[assignment]

_build_geo_from_dxf_hook: Optional[Callable[[str], Dict[str, Any]]] = None

try:  # pragma: no cover - optional PyMuPDF dependency
    from cad_quoter.geometry import (
        build_llm_payload as _build_pdf_llm_payload,
        fitz as _fitz_mod,
        pdf_to_structured as _pdf_to_structured,
    )
except Exception:  # pragma: no cover - keep optional at runtime
    _build_pdf_llm_payload = None  # type: ignore[assignment]
    _fitz_mod = None  # type: ignore[assignment]
    _pdf_to_structured = None  # type: ignore[assignment]


def build_geo_from_dxf(path: str) -> Dict[str, Any]:
    """Return auxiliary DXF metadata via the configured loader."""

    loader: Optional[Callable[[str], Dict[str, Any]]]
    loader = _build_geo_from_dxf_hook or _build_geo_from_dxf_path
    if loader is None:
        raise RuntimeError(
            "DXF metadata loader is unavailable; install cad_quoter.geometry extras or register a hook."
        )
    result = loader(path)
    if not isinstance(result, dict):
        raise TypeError("DXF metadata loader must return a dictionary")
    return result


def set_build_geo_from_dxf_hook(loader: Optional[Callable[[str], Dict[str, Any]]]) -> None:
    """Register a callable used by :func:`build_geo_from_dxf`."""

    if loader is not None and not callable(loader):
        raise TypeError("DXF metadata hook must be callable or ``None``")

    global _build_geo_from_dxf_hook
    _build_geo_from_dxf_hook = loader


def pdf_document_to_structured(path: str | Path, dpi: int = 300) -> Dict[str, Any]:
    """Return structured PDF metadata using the shared geometry helper."""

    if _pdf_to_structured is None or _fitz_mod is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "PyMuPDF (fitz) is required for PDF extraction; install 'pymupdf' to enable this feature."
        )

    pdf_path = Path(path)
    try:
        doc = _fitz_mod.open(pdf_path)
    except Exception as exc:  # pragma: no cover - surface a consistent runtime error
        raise RuntimeError(f"Failed to open PDF: {exc}") from exc

    pages: list[Dict[str, Any]] = []
    best_index: int | None = None
    best_key: tuple[int, int] | None = None

    try:
        for index in range(doc.page_count):
            structured = _pdf_to_structured(pdf_path, page_index=index, dpi=dpi)
            text = str(structured.get("raw_text") or "")
            tables = structured.get("tables")
            table_count = len(tables) if isinstance(tables, list) else 0
            page_info = {
                "index": index,
                "text": text,
                "char_count": len(text),
                "table_count": table_count,
                "image_path": structured.get("image_path"),
                "structured": structured,
            }
            pages.append(page_info)

            key = (1 if table_count > 0 else 0, len(text))
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
    finally:
        try:
            doc.close()
        except Exception:
            pass

    best_page: Dict[str, Any] | None = None
    if best_index is not None and 0 <= best_index < len(pages):
        best_page = pages[best_index]

    return {"pages": pages, "best": best_page or {}}


def build_pdf_llm_payload(
    structured: Mapping[str, Any] | None,
    image_path: str | None = None,
) -> list[dict[str, Any]]:
    """Construct a multimodal chat payload for the shared PDF prompt."""

    if _build_pdf_llm_payload is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("PDF LLM payload helper is unavailable; install optional geometry extras.")

    struct_map = dict(structured or {})
    page_image = image_path or struct_map.get("image_path")
    image_arg = page_image if isinstance(page_image, str) and page_image else None
    return _build_pdf_llm_payload(struct_map, image_arg)


__all__ = [
    "pd",
    "build_geo_from_dxf",
    "set_build_geo_from_dxf_hook",
    "pdf_document_to_structured",
    "build_pdf_llm_payload",
]
