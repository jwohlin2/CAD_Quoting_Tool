"""DXF text harvesting helpers shared across the application."""

from __future__ import annotations

from typing import Iterable, List

from cad_quoter.vendors import ezdxf as _ezdxf_vendor


def _iter_text_entities(space: object) -> Iterable[str]:
    """Yield plain text content from TEXT/MTEXT entities within ``space``."""

    try:
        for entity in space:  # type: ignore[assignment]
            try:
                dxftype = entity.dxftype()
            except Exception:
                continue
            if dxftype in {"TEXT", "MTEXT"}:
                try:
                    if dxftype == "MTEXT" and hasattr(entity, "plain_text"):
                        text = entity.plain_text()
                    else:
                        text = entity.dxf.text
                except Exception:
                    text = ""
                if text:
                    yield str(text)
            elif dxftype == "INSERT":
                try:
                    virtual_entities = entity.virtual_entities()
                except Exception:
                    virtual_entities = []
                for child in virtual_entities:
                    try:
                        child_type = child.dxftype()
                    except Exception:
                        continue
                    if child_type in {"TEXT", "MTEXT"}:
                        try:
                            if child_type == "MTEXT" and hasattr(child, "plain_text"):
                                child_text = child.plain_text()
                            else:
                                child_text = child.dxf.text
                        except Exception:
                            child_text = ""
                        if child_text:
                            yield str(child_text)
    except Exception:
        return


def extract_text_lines_from_dxf(path: str) -> List[str]:
    """Return a list of text fragments extracted from a DXF document."""

    doc = _ezdxf_vendor.read_document(path)
    lines: List[str] = []

    try:
        modelspace = doc.modelspace()
    except Exception:
        modelspace = None

    if modelspace is not None:
        lines.extend(_iter_text_entities(modelspace))

    try:
        layouts = doc.layouts.names_in_taborder()
    except Exception:
        layouts = []

    for layout_name in layouts:
        if layout_name.lower() in {"model", "defpoints"}:
            continue
        try:
            space = doc.layouts.get(layout_name).entity_space
        except Exception:
            continue
        lines.extend(_iter_text_entities(space))

    joined = "\n".join(lines)
    upper = joined.upper()
    if "HOLE TABLE" in upper:
        start = upper.index("HOLE TABLE")
        joined = joined[start:]

    return [fragment for fragment in joined.splitlines() if fragment.strip()]


__all__ = ["extract_text_lines_from_dxf"]

