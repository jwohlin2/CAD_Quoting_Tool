from cad_quoter.vendors import ezdxf as _ezdxf_vendor


def extract_text_lines_from_dxf(path: str) -> list[str]:
    doc = _ezdxf_vendor.read_document(path)
    msp = doc.modelspace()
    lines: list[str] = []
    for e in msp:
        if e.dxftype() in ("TEXT", "MTEXT"):
            s = e.plain_text() if hasattr(e, "plain_text") else e.dxf.text
            if s:
                lines.append(s)
        elif e.dxftype() == "INSERT":
            try:
                for sub in e.virtual_entities():
                    if sub.dxftype() in ("TEXT", "MTEXT"):
                        s = sub.plain_text() if hasattr(sub, "plain_text") else sub.dxf.text
                        if s:
                            lines.append(s)
            except Exception:
                pass
    joined = "\n".join(lines)
    if "HOLE TABLE" in joined.upper():
        upper = joined.upper()
        start = upper.index("HOLE TABLE")
        joined = joined[start:]
    return [ln for ln in joined.splitlines() if ln.strip()]
