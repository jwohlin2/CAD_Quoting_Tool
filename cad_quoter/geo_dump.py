#GEO dump
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Sequence

try:  # pragma: no cover - optional dependency guard
    from ezdxf.math import Matrix44, Vec3
except Exception:  # pragma: no cover - fallback for environments without ezdxf
    Matrix44 = None  # type: ignore[assignment]
    Vec3 = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cad_quoter import geo_extractor
from cad_quoter.geo_extractor import (
    DEFAULT_TEXT_LAYER_EXCLUDE_REGEX,
    NO_TEXT_ROWS_MESSAGE,
    NoTextRowsError,
    TextScanOpts,
    extract_for_app,
)
from cad_quoter.geometry.mtext_utils import normalize_mtext_plain_text

DEFAULT_SAMPLE_PATH = REPO_ROOT / "Cad Files" / "301_redacted.dwg"
ARTIFACT_DIR = REPO_ROOT / "out"
DEFAULT_EXCLUDE_PATTERN_TEXT = ", ".join(DEFAULT_TEXT_LAYER_EXCLUDE_REGEX) or "<none>"
DEFAULT_TEXT_LAYER_INCLUDE_REGEX = (".*",)

_FULL_TEXT_FIELDS = [
    "layout",
    "entity_type",
    "layer",
    "height",
    "width",
    "rotation",
    "x",
    "y",
    "raw_text",
    "plain_text",
    "style",
    "handle",
    "block_name",
    "from_block",
]

_HEIGHT_ATTRS = tuple(
    getattr(
        geo_extractor,
        "_DEFAULT_HEIGHT_ATTRS",
        ("char_height", "text_height", "height", "size"),
    )
)
_ROTATION_ATTRS = tuple(
    getattr(geo_extractor, "_DEFAULT_ROTATION_ATTRS", ("rotation", "rot"))
)
_INSERT_ATTRS = tuple(
    getattr(
        geo_extractor,
        "_DEFAULT_INSERT_ATTRS",
        (
            "insert",
            "alignment_point",
            "location",
            "base_point",
            "defpoint",
            "start",
            "point",
        ),
    )
)

TABLE_EXTRACT_ALLOWED_KEYS = {
    "layer_allowlist",
    "roi_hint",
    "block_name_allowlist",
    "block_name_regex",
    "layer_include_regex",
    "layer_exclude_regex",
    "layout_filters",
    "debug_layouts",
    "debug_scan",
    "text_scan_opts",
}


def iter_table_cells(layout: tuple[str, Any] | Any) -> list[dict[str, Any]]:
    """Return table cell records for ``layout``."""

    layout_name = None
    layout_obj = layout
    if isinstance(layout, tuple) and len(layout) == 2:
        candidate_name, candidate_layout = layout
        if isinstance(candidate_name, str):
            layout_name = candidate_name
        layout_obj = candidate_layout
    return geo_extractor.iter_table_cells((layout_name, layout_obj))


def _sum_qty(rows: list[Mapping[str, object]] | None) -> int:
    total = 0
    if not rows:
        return total
    for row in rows:
        try:
            total += int(float(row.get("qty", 0) or 0))  # type: ignore[arg-type]
        except Exception:
            continue
    return total

def _int_from_value(value: Any) -> int:
    try:
        return int(round(float(value or 0)))
    except Exception:
        return 0


def _format_height_display(value: Any) -> str:
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except Exception:
            return "-"
        if math.isfinite(number):
            return f"{number:.3f}"
    return "-"


def _format_rotation_display(value: Any) -> str:
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except Exception:
            return "-"
        if math.isfinite(number):
            return f"{number:.1f}"
    return "-"


def _format_text_preview(text: Any, limit: int = 120) -> str:
    try:
        value = str(text or "")
    except Exception:
        value = ""
    preview = value.replace("\n", "\\n")
    if len(preview) > limit:
        preview = preview[: limit - 3] + "..."
    return preview


def _format_counter_summary(counter: Counter[str]) -> str:
    if not counter:
        return "{}"
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0] or ""))
    summary = ", ".join(f"{name or '-'}:{count}" for name, count in ordered)
    return "{" + summary + "}"


def _log_text_stats(entries: Sequence[Mapping[str, Any]]) -> Counter[str]:
    total = len(entries)
    type_counts = Counter(str(entry.get("etype") or "-") for entry in entries)
    layout_counts = Counter(str(entry.get("layout") or "-") for entry in entries)
    print(f"[TEXT-DUMP] total={total} by etype={_format_counter_summary(type_counts)}")
    print(f"[TEXT-DUMP] by layout={_format_counter_summary(layout_counts)}")
    if entries:
        layout_type_counts: dict[str, Counter[str]] = {}
        for entry in entries:
            layout = str(entry.get("layout") or "-")
            etype = str(entry.get("etype") or "-")
            layout_type_counts.setdefault(layout, Counter())[etype] += 1
        priority_types = ("TEXT", "MTEXT", "ATTRIB", "MLEADER", "TABLE", "TABLECELL")
        for layout in sorted(layout_type_counts):
            counter = layout_type_counts[layout]
            parts: list[str] = []
            for etype in priority_types:
                parts.append(f"{etype}={counter.get(etype, 0)}")
            extra_types = sorted(
                etype for etype in counter if etype not in priority_types
            )
            for etype in extra_types:
                parts.append(f"{etype}={counter.get(etype, 0)}")
            summary = " ".join(parts) if parts else "-"
            print(f"[TEXT-DUMP] {layout}: {summary}")
    return layout_counts


def _print_text_dump(
    entries: Sequence[Mapping[str, Any]],
    *,
    sample_limit: int | None = None,
    group_by_layout: bool = True,
    layout_counts: Mapping[str, int] | None = None,
) -> None:
    if layout_counts is None:
        layout_counts = Counter(str(entry.get("layout") or "-") for entry in entries)

    if not entries:
        return

    sample = entries[0]
    sample_height = _format_height_display(sample.get("height"))
    sample_text = _format_text_preview(sample.get("text"), limit=100)
    sample_layer = str(sample.get("layer") or "-")
    sample_layout = str(sample.get("layout") or "-")
    print(
        "[TEXT-DUMP] sample: [{etype} layout={layout} layer={layer} h={height} \"{text}\"]".format(
            etype=str(sample.get("etype") or "-"),
            layout=sample_layout,
            layer=sample_layer,
            height=sample_height,
            text=sample_text,
        )
    )

    if sample_limit is not None and sample_limit <= 0:
        return

    max_lines = sample_limit if sample_limit is not None else 20
    lines_shown = 0
    seen_layouts: set[str] = set()
    for entry in entries:
        if lines_shown >= max_lines:
            break
        layout = str(entry.get("layout") or "-")
        if group_by_layout and layout not in seen_layouts:
            print(f"[TEXT-DUMP] layout={layout} count={layout_counts.get(layout, 0)}")
            seen_layouts.add(layout)
        elif not group_by_layout and layout not in seen_layouts:
            seen_layouts.add(layout)
        height_display = _format_height_display(entry.get("height"))
        rotation_display = _format_rotation_display(entry.get("rotation"))
        layer_display = str(entry.get("layer") or "-")
        preview = _format_text_preview(entry.get("text"))
        block_path = entry.get("block_path") or ()
        block_display = ""
        if block_path:
            block_display = " blocks=" + " > ".join(str(name) for name in block_path)
        print(
            "[TEXT-DUMP]   {etype:<9} layer={layer} h={height} rot={rot}{blocks} \"{text}\"".format(
                etype=str(entry.get("etype") or "-"),
                layer=layer_display,
                height=height_display,
                rot=rotation_display,
                blocks=block_display,
                text=preview,
            )
        )
        lines_shown += 1


def write_text_dump_csv(rows: Sequence[Mapping[str, Any]], out_dir: str | os.PathLike[str]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dxf_text_dump.csv")
    cols = [
        "handle",
        "etype",
        "layout",
        "layer",
        "height",
        "rotation",
        "insert_x",
        "insert_y",
        "block_path",
        "text",
        "raw",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in rows:
            writer.writerow(
                [
                    row.get("handle"),
                    row.get("etype"),
                    row.get("layout"),
                    row.get("layer"),
                    row.get("height"),
                    row.get("rotation"),
                    row.get("insert_x"),
                    row.get("insert_y"),
                    " :: ".join(str(part) for part in (row.get("block_path") or [])),
                    row.get("text"),
                    row.get("raw"),
                ]
            )
    print(f"[TEXT-DUMP] csv -> {path}")
    return path


def write_text_dump_jsonl(rows: Sequence[Mapping[str, Any]], out_dir: str | os.PathLike[str]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dxf_text_dump.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")
    print(f"[TEXT-DUMP] jsonl -> {path}")
    return path


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _resolve_first_attr(dxf: Any, attrs: Sequence[str]) -> Any:
    if dxf is None:
        return None
    for name in attrs:
        if not hasattr(dxf, name):
            continue
        try:
            value = getattr(dxf, name)
        except Exception:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        return value
    return None


def _vector_xy(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return (None, None)
    if hasattr(value, "x") and hasattr(value, "y"):
        return (_safe_float(value.x), _safe_float(value.y))
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        try:
            data = list(value)
        except Exception:
            data = []
        if len(data) >= 2:
            return (_safe_float(data[0]), _safe_float(data[1]))
    return (None, None)


def _resolve_xy_from_dxf(dxf: Any) -> tuple[float | None, float | None]:
    for attr in _INSERT_ATTRS:
        if not hasattr(dxf, attr):
            continue
        try:
            point = getattr(dxf, attr)
        except Exception:
            continue
        x_val, y_val = _vector_xy(point)
        if x_val is not None or y_val is not None:
            return (x_val, y_val)
    return (None, None)


def _resolve_point_vec(dxf: Any) -> Vec3 | None:
    if Vec3 is None:
        return None
    for attr in _INSERT_ATTRS:
        if not hasattr(dxf, attr):
            continue
        try:
            point = getattr(dxf, attr)
        except Exception:
            continue
        if point is None:
            continue
        if hasattr(point, "x") and hasattr(point, "y"):
            try:
                x_val = _safe_float(point.x)
                y_val = _safe_float(point.y)
                z_val = _safe_float(getattr(point, "z", 0.0))
            except Exception:
                continue
            return Vec3(x_val or 0.0, y_val or 0.0, z_val or 0.0)
        if isinstance(point, Iterable) and not isinstance(point, (str, bytes, bytearray)):
            try:
                data = list(point)
            except Exception:
                continue
            if not data:
                continue
            x_val = _safe_float(data[0]) or 0.0
            y_val = _safe_float(data[1]) or 0.0
            z_val = _safe_float(data[2]) if len(data) >= 3 else 0.0
            return Vec3(x_val, y_val, z_val)
    return None


def _transform_direction(matrix: Matrix44 | None, vector: Vec3) -> Vec3:
    if Matrix44 is None or matrix is None:
        return vector
    try:
        return matrix.transform_direction(vector)
    except Exception:
        origin = matrix.transform(Vec3(0.0, 0.0, 0.0))
        target = matrix.transform(vector)
        return target - origin


def _resolve_style_name(dxf: Any) -> str | None:
    if dxf is None:
        return None
    for attr in ("style", "text_style", "dimstyle"):
        if not hasattr(dxf, attr):
            continue
        try:
            value = getattr(dxf, attr)
        except Exception:
            continue
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return None


def _resolve_handle(entity: Any) -> str | None:
    handle = None
    dxf = getattr(entity, "dxf", None)
    if dxf is not None and hasattr(dxf, "handle"):
        try:
            handle = dxf.handle
        except Exception:
            handle = None
    if handle in (None, "") and hasattr(entity, "handle"):
        try:
            handle = entity.handle
        except Exception:
            handle = None
    if handle in (None, ""):
        return None
    return str(handle)
def _extract_text_strings(entity: Any, etype: str) -> tuple[str | None, str | None]:
    raw_text: str | None = None
    plain_text: str | None = None
    dxf = getattr(entity, "dxf", None)

    if etype == "MTEXT":
        try:
            raw_text = str(getattr(entity, "text", ""))
        except Exception:
            raw_text = None
        if hasattr(entity, "plain_text"):
            try:
                plain_text = str(entity.plain_text())
            except Exception:
                plain_text = None
        if plain_text is None and raw_text is not None:
            plain_text = normalize_mtext_plain_text(raw_text)
    elif etype in {"TEXT", "ATTRIB", "ATTDEF"}:
        if dxf is not None and hasattr(dxf, "text"):
            try:
                raw_text = str(dxf.text)
            except Exception:
                raw_text = None
        if raw_text is not None:
            plain_text = raw_text
    elif etype == "MLEADER":
        mtext_obj = None
        get_mtext = getattr(entity, "get_mtext", None)
        if callable(get_mtext):
            try:
                mtext_obj = get_mtext()
            except Exception:
                mtext_obj = None
        if mtext_obj is not None:
            try:
                raw_text = str(getattr(mtext_obj, "text", ""))
            except Exception:
                raw_text = None
            if hasattr(mtext_obj, "plain_text"):
                try:
                    plain_text = str(mtext_obj.plain_text())
                except Exception:
                    plain_text = None
            if plain_text is None and raw_text is not None:
                plain_text = normalize_mtext_plain_text(raw_text)
            if hasattr(mtext_obj, "destroy"):
                try:
                    mtext_obj.destroy()
                except Exception:
                    pass
        if raw_text is None and dxf is not None and hasattr(dxf, "text"):
            try:
                raw_text = str(dxf.text)
            except Exception:
                raw_text = None
        if raw_text is not None and plain_text is None:
            plain_text = normalize_mtext_plain_text(raw_text)

    if plain_text is None and raw_text is not None:
        plain_text = raw_text

    return (raw_text, plain_text)


def _compile_regex_list(patterns: Sequence[str] | None, *, default: Sequence[str] | None = None) -> list[re.Pattern[str]]:
    raw_patterns: list[str] = []
    if patterns:
        for value in patterns:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                raw_patterns.append(text)
    elif default:
        raw_patterns.extend(str(value).strip() for value in default if str(value).strip())
    compiled: list[re.Pattern[str]] = []
    for text in raw_patterns:
        try:
            compiled.append(re.compile(text, re.IGNORECASE))
        except re.error as exc:
            print(f"[TEXT-DUMP] layer regex error pattern={text!r} err={exc}")
    return compiled


def _iter_insert_attribs(insert: Any) -> list[Any]:
    attribs: list[Any] = []
    if insert is None:
        return attribs
    candidate = getattr(insert, "attribs", None)
    if callable(candidate):
        try:
            attribs = list(candidate())
        except Exception:
            attribs = []
    elif isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes, bytearray)):
        try:
            attribs = list(candidate)
        except Exception:
            attribs = []
    return attribs


def dump_all_text(doc: Any, out_dir: Path | str, opts: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], Path, Path]:
    options = dict(opts or {})
    min_height_raw = options.get("text_min_height", 0.0)
    min_height_value: float | None = None
    if min_height_raw is not None:
        try:
            candidate = float(min_height_raw)
        except Exception:
            candidate = None
        if candidate is not None and math.isfinite(candidate) and candidate > 0:
            min_height_value = candidate

    include_patterns = _compile_regex_list(
        options.get("text_include_layers"), default=[".*"]
    )
    exclude_patterns = _compile_regex_list(options.get("text_exclude_layers"))

    layout_filters = [
        str(value).strip()
        for value in options.get("text_layouts") or []
        if isinstance(value, str) and value.strip()
    ]
    if layout_filters:
        layout_arg: Mapping[str, Any] | Iterable[str] | None = {
            "all_layouts": False,
            "patterns": layout_filters,
        }
    else:
        layout_arg = {"all_layouts": True, "patterns": []}

    try:
        layout_spaces = geo_extractor.iter_layouts(doc, layout_arg, log=False)
    except Exception:
        layout_spaces = []

    records: list[dict[str, Any]] = []
    mleader_total = 0
    mleader_captured = 0
    from_blocks_depth_max = 0

    def record_matches_filters(entry: Mapping[str, Any]) -> bool:
        layer_name = str(entry.get("layer") or "")
        if include_patterns and not any(pattern.search(layer_name) for pattern in include_patterns):
            return False
        if exclude_patterns and any(pattern.search(layer_name) for pattern in exclude_patterns):
            return False
        if min_height_value is not None:
            height_value = entry.get("height")
            if isinstance(height_value, (int, float)) and height_value < min_height_value:
                return False
        return True

    def build_tablecell_record(entry: Mapping[str, Any]) -> dict[str, Any] | None:
        if not isinstance(entry, Mapping):
            return None
        layout_value = str(entry.get("layout") or "").strip() or "-"
        raw_value = entry.get("raw")
        text_value = entry.get("text")
        try:
            raw_text = str(raw_value) if raw_value not in (None, "") else ""
        except Exception:
            raw_text = ""
        try:
            plain_text = str(text_value) if text_value not in (None, "") else ""
        except Exception:
            plain_text = raw_text
        if not plain_text:
            plain_text = raw_text
        if not (plain_text or raw_text):
            return None

        block_path = entry.get("block_path")
        if isinstance(block_path, (list, tuple)) and block_path:
            block_name = " > ".join(str(part) for part in block_path if part)
            from_block = 1
        else:
            block_name = None
            from_block = 0

        handle_value = entry.get("handle")
        if isinstance(handle_value, str):
            handle_text = handle_value
        elif handle_value in (None, ""):
            handle_text = ""
        else:
            try:
                handle_text = str(handle_value)
            except Exception:
                handle_text = ""

        return {
            "layout": layout_value,
            "entity_type": "TABLECELL",
            "layer": str(entry.get("layer") or ""),
            "height": _safe_float(entry.get("height")),
            "width": _safe_float(entry.get("width")),
            "rotation": _safe_float(entry.get("rotation")),
            "x": _safe_float(entry.get("insert_x")),
            "y": _safe_float(entry.get("insert_y")),
            "raw_text": raw_text or plain_text,
            "plain_text": plain_text or raw_text,
            "style": str(entry.get("style") or ""),
            "handle": handle_text,
            "block_name": block_name,
            "from_block": from_block,
        }

    def build_record(entity: Any, layout_name: str, *, from_block: bool, block_name: str | None) -> dict[str, Any] | None:
        dxf = getattr(entity, "dxf", None)
        try:
            etype = str(entity.dxftype()).upper()
        except Exception:
            etype = ""
        raw_text, plain_text = _extract_text_strings(entity, etype)
        if raw_text is None and plain_text is None:
            return None
        layer_value = ""
        if dxf is not None and hasattr(dxf, "layer"):
            try:
                layer_value = str(dxf.layer or "")
            except Exception:
                layer_value = ""
        height_candidate = _resolve_first_attr(dxf, _HEIGHT_ATTRS)
        width_candidate = _resolve_first_attr(dxf, ("width", "text_width", "char_width"))
        rotation_candidate = _resolve_first_attr(dxf, _ROTATION_ATTRS)
        x_val, y_val = _world_point(entity, dxf, matrix)
        style_value = _resolve_style_name(dxf)
        handle_value = _resolve_handle(entity)
        block_value = _compose_block_name(block_path)
        rotation_value = _safe_float(rotation_candidate)
        rotation_world = _world_rotation(entity, rotation_value, matrix)
        height_value = _safe_float(height_candidate)
        height_world = _world_height(entity, height_value, matrix)
        entry = {
            "layout": str(layout_name or "").strip() or "-",
            "entity_type": etype or "-",
            "layer": layer_value,
            "height": height_world,
            "width": _safe_float(width_candidate),
            "rotation": rotation_world,
            "x": x_val,
            "y": y_val,
            "raw_text": raw_text or "",
            "plain_text": plain_text or "",
            "style": style_value,
            "handle": handle_value,
            "block_name": block_value,
            "from_block": 1 if from_block else 0,
            "block_path": list(block_path),
        }
        return entry

    def walk_entity(
        entity: Any,
        layout_name: str,
        *,
        transform: Matrix44 | None,
        block_path: tuple[str, ...] = (),
    ) -> None:
        nonlocal mleader_total, mleader_captured, from_blocks_depth_max
        if entity is None:
            return
        try:
            etype = str(entity.dxftype()).upper()
        except Exception:
            etype = ""
        if block_path:
            depth = len(block_path)
            if depth > from_blocks_depth_max:
                from_blocks_depth_max = depth
        if etype in {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"}:
            record = build_record(
                entity,
                layout_name,
                from_block=bool(block_path),
                block_path=block_path,
                matrix=transform,
            )
            if record and record_matches_filters(record):
                records.append(record)
            return
        if etype == "MLEADER":
            mleader_total += 1
            record = build_record(entity, layout_name, from_block=from_block, block_name=block_name)
            if record and record.get("plain_text") and record_matches_filters(record):
                records.append(record)
                mleader_captured += 1
            return
        if etype != "INSERT":
            return

        insert_name = None
        dxf = getattr(entity, "dxf", None)
        if dxf is not None:
            for attr in ("name", "block_name"):
                if hasattr(dxf, attr):
                    try:
                        value = getattr(dxf, attr)
                    except Exception:
                        value = None
                    if value not in (None, ""):
                        insert_name = str(value)
                        break

        combined_path: tuple[str, ...]
        if insert_name:
            combined_path = (*block_path, insert_name)
        else:
            combined_path = block_path

        child_transform: Matrix44 | None = transform
        if Matrix44 is not None:
            matrix_value = None
            try:
                matrix_value = entity.matrix44()
            except Exception:
                matrix_value = None
            if matrix_value is not None:
                if child_transform is None:
                    child_transform = matrix_value
                else:
                    try:
                        child_transform = child_transform @ matrix_value
                    except Exception:
                        child_transform = matrix_value

        for attrib in _iter_insert_attribs(entity):
            walk_entity(
                attrib,
                layout_name,
                transform=child_transform,
                block_path=combined_path,
            )

        virtual_entities: list[Any] = []
        try:
            virtual_entities = list(entity.virtual_entities(deep=True))
        except Exception:
            virtual_entities = []
        for child in virtual_entities:
            walk_entity(
                child,
                layout_name,
                transform=child_transform,
                block_path=combined_path,
            )
            if hasattr(child, "destroy"):
                try:
                    child.destroy()
                except Exception:
                    pass

    for layout_name, layout in layout_spaces:
        if layout is None:
            continue
        for table_entry in iter_table_cells((layout_name, layout)):
            table_record = build_tablecell_record(table_entry)
            if table_record and record_matches_filters(table_record):
                records.append(table_record)
        for entity in layout:
            walk_entity(
                entity,
                str(layout_name or ""),
                transform=Matrix44() if Matrix44 is not None else None,
                block_path=(),
            )

    out_dir_path = Path(out_dir).expanduser()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir_path / "dxf_text_dump_full.csv"
    jsonl_path = out_dir_path / "dxf_text_dump_full.jsonl"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_FULL_TEXT_FIELDS)
        for entry in records:
            writer.writerow([entry.get(field) for field in _FULL_TEXT_FIELDS])

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for entry in records:
            json.dump(entry, handle, ensure_ascii=False)
            handle.write("\n")

    print(f"[TEXT-DUMP] full csv -> {csv_path}")
    print(f"[TEXT-DUMP] full jsonl -> {jsonl_path}")
    print(f"[TEXT-DUMP] from_blocks_depth_max={from_blocks_depth_max}")

    return (records, csv_path, jsonl_path)


def _write_rows_csv(
    rows: Sequence[Mapping[str, Any]] | None,
    path: Path | str = Path("debug/hole_table_rows.csv"),
) -> Path | None:
    csv_path = Path(path)
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["qty", "ref", "side", "desc", "hole"])
            for row in rows or []:
                if not isinstance(row, Mapping):
                    continue
                qty_val = row.get("qty")
                writer.writerow(
                    [
                        "" if qty_val in (None, "") else str(qty_val),
                        str(row.get("ref") or ""),
                        str(row.get("side") or ""),
                        str(row.get("desc") or ""),
                        str(row.get("hole") or ""),
                    ]
                )
    except OSError as exc:
        print(f"[geo_dump] failed to write hole table CSV: {exc}")
        return None
    else:
        print(f"[geo_dump] wrote hole table CSV to {csv_path}")
        return csv_path


def _write_json_debug(
    path: Path | str,
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> Path | None:
    json_path = Path(path)
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload or {}, handle, indent=2)
            handle.write("\n")
    except OSError as exc:
        print(f"[geo_dump] failed to write JSON debug payload {json_path}: {exc}")
        return None
    else:
        print(f"[geo_dump] wrote JSON debug payload to {json_path}")
        return json_path


def _write_geo_summary(
    path: Path | str,
    *,
    rows: Sequence[Mapping[str, Any]] | None,
    ops_totals: Mapping[str, Any] | None,
    geom_summary: Mapping[str, Any],
    artifact_paths: Mapping[str, Path | None],
) -> Path | None:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            handle.write("GEO SUMMARY\n")
            handle.write("===========\n\n")

            ordered_ops: dict[str, Any] = {}
            if ops_totals and isinstance(ops_totals, Mapping):
                totals_map = ops_totals.get("totals")
                if isinstance(totals_map, Mapping):
                    ordered_ops.update(_ordered_totals_map(totals_map))
                else:
                    ordered_ops.update(
                        {
                            str(key): value
                            for key, value in ops_totals.items()
                            if key != "totals"
                        }
                    )
            if ordered_ops:
                handle.write("[OPERATIONS]\n")
                for key, value in ordered_ops.items():
                    handle.write(f"- {key}: {value}\n")
                handle.write("\n")

            handle.write("[HOLE TABLE]\n")
            if rows:
                handle.write("qty | ref | side | desc\n")
                handle.write("----+-----+------+-----\n")
                for row in rows[:200]:
                    qty_val = row.get("qty") if isinstance(row, Mapping) else None
                    qty_display = "" if qty_val in (None, "") else str(qty_val)
                    ref_display = str(row.get("ref") or "")
                    side_display = str(row.get("side") or row.get("face") or "")
                    desc_display = str(row.get("desc") or row.get("description") or "")
                    handle.write(
                        "{qty} | {ref} | {side} | {desc}\n".format(
                            qty=qty_display,
                            ref=ref_display,
                            side=side_display,
                            desc=desc_display,
                        )
                    )
                if len(rows) > 200:
                    handle.write(f"... ({len(rows) - 200} more rows)\n")
            else:
                handle.write("<no rows>\n")
            handle.write("\n")

            handle.write("[GEOMETRY]\n")
            circle_total = geom_summary.get("circle_total")
            unique_count = geom_summary.get("unique_diameter_count")
            handle.write(
                "Circles total: {total} | Unique diameters: {unique}\n".format(
                    total=circle_total if circle_total is not None else "-",
                    unique=unique_count if unique_count is not None else "-",
                )
            )
            layer_counts = geom_summary.get("layer_counts")
            if isinstance(layer_counts, Mapping) and layer_counts:
                top_layers = sorted(
                    layer_counts.items(), key=lambda item: (-int(item[1]), item[0])
                )
                handle.write("Top layers:\n")
                for name, count in top_layers[:10]:
                    handle.write(f"  - {name}: {count}\n")
                if len(top_layers) > 10:
                    handle.write("  - …\n")
            guard_counts = geom_summary.get("guard_drop_counts")
            if isinstance(guard_counts, Mapping) and guard_counts:
                handle.write("Guard drops:\n")
                for name, count in sorted(
                    guard_counts.items(), key=lambda item: (-int(item[1]), item[0])
                ):
                    handle.write(f"  - {name}: {count}\n")
            handle.write("\n")

            handle.write("[ARTIFACTS]\n")
            for label, artifact_path in artifact_paths.items():
                path_display = "-"
                if artifact_path:
                    path_display = str(artifact_path)
                handle.write(f"{label}: {path_display}\n")
    except OSError as exc:
        print(f"[geo_dump] failed to write GEO summary: {exc}")
        return None
    else:
        print(f"[geo_dump] wrote GEO summary to {target}")
        return target


def _normalize_pattern_args(values: Sequence[str] | None) -> list[str]:
    patterns: list[str] = []
    if not values:
        return patterns
    for value in values:
        if value is None:
            continue
        text = str(value)
        for part in text.split(","):
            candidate = part.strip()
            if candidate:
                patterns.append(candidate)
    return patterns


def _unique_patterns(patterns: Sequence[str]) -> tuple[str, ...]:
    if not patterns:
        return ()
    return tuple(dict.fromkeys(patterns))


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "t", "yes", "y"}
    return False


def _anchor_authoritative_from_candidates(
    *candidates: Mapping[str, Any] | None,
) -> bool:
    keys = ("anchor_authoritative", "table_authoritative", "authoritative")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        for key in keys:
            if key not in candidate:
                continue
            if _truthy_flag(candidate.get(key)):
                return True
    return False


def _provenance_is_anchor(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        return "hole table" in normalized and "anchor" in normalized
    return False


def _coerce_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _format_float_str(value: Any) -> str:
    number = _coerce_float(value)
    if number is None:
        return "-"
    return f"{number:.3f}"


def _compile_regex_patterns_for_audit(
    patterns: Iterable[Any] | None,
) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    if not patterns:
        return compiled
    for pattern in patterns:
        text = str(pattern or "").strip()
        if not text:
            continue
        try:
            compiled.append(re.compile(text, re.IGNORECASE))
        except re.error:
            continue
    return compiled


def _chart_height_value(entry: Mapping[str, Any]) -> float | None:
    value = entry.get("height")
    number = _coerce_float(value)
    if number is None or not math.isfinite(number):
        return None
    if number <= 0:
        return None
    return float(number)


def _print_chart_audit(
    debug_info: Mapping[str, Any] | None,
    *,
    tables_found: int,
    text_min_height: float | None = None,
) -> None:
    if not isinstance(debug_info, Mapping):
        return
    raw_entities = debug_info.get("collected_entities")
    if not isinstance(raw_entities, Sequence):
        return
    chart_entries: list[Mapping[str, Any]] = []
    for entry in raw_entities:
        if not isinstance(entry, Mapping):
            continue
        layout_name = str(entry.get("layout") or "")
        if "CHART" not in layout_name.upper():
            continue
        chart_entries.append(entry)
    if not chart_entries:
        return

    text_raw = len(chart_entries)
    in_blocks = sum(1 for entry in chart_entries if bool(entry.get("from_block")))

    include_patterns = _compile_regex_patterns_for_audit(
        debug_info.get("layer_regex_include")
    )
    exclude_patterns = _compile_regex_patterns_for_audit(
        debug_info.get("layer_regex_exclude")
    )

    def _matches(patterns: Sequence[re.Pattern[str]], layer: str) -> bool:
        for pattern in patterns:
            try:
                if pattern.search(layer):
                    return True
            except re.error:
                continue
        return False

    filtered_layer: list[Mapping[str, Any]] = []
    for entry in chart_entries:
        layer_name = str(entry.get("layer") or "")
        include_ok = True
        if include_patterns:
            include_ok = _matches(include_patterns, layer_name)
        if not include_ok:
            continue
        if exclude_patterns and _matches(exclude_patterns, layer_name):
            continue
        filtered_layer.append(entry)
    kept_by_layer = len(filtered_layer)

    height_threshold: float | None = None
    if text_min_height is not None:
        try:
            candidate = float(text_min_height)
        except Exception:
            candidate = None
        if candidate is not None and math.isfinite(candidate) and candidate > 0:
            height_threshold = candidate

    filtered_height: list[Mapping[str, Any]] = []
    for entry in filtered_layer:
        height_value = _chart_height_value(entry)
        if height_threshold is not None:
            if height_value is None or height_value < height_threshold:
                continue
        filtered_height.append(entry)
    kept_by_height = len(filtered_height)

    height_samples = [
        value
        for entry in chart_entries
        for value in [_chart_height_value(entry)]
        if value is not None
    ]
    if height_samples:
        min_display = f"{min(height_samples):.3f}"
        med_display = f"{statistics.median(height_samples):.3f}"
        max_display = f"{max(height_samples):.3f}"
    else:
        min_display = med_display = max_display = "-"

    layer_counter: Counter[str] = Counter()
    for entry in chart_entries:
        layer_name = str(entry.get("layer") or "").strip()
        if not layer_name:
            layer_name = "-"
        layer_counter[layer_name] += 1
    if layer_counter:
        top_layers = sorted(layer_counter.items(), key=lambda item: (-item[1], item[0]))
        top_layers_display = ", ".join(
            f"{name}:{count}" for name, count in top_layers[:3]
        )
    else:
        top_layers_display = "-"

    tables_count = int(tables_found)
    summary_message = (
        "[AUDIT] CHART text_raw={text_raw} in_blocks={in_blocks} "
        "kept_by_layer={kept} kept_by_height={height} tables={tables}".format(
            text_raw=text_raw,
            in_blocks=in_blocks,
            kept=kept_by_layer,
            height=kept_by_height,
            tables=tables_count,
        )
    )
    print(f"Top layers: {top_layers_display}")
    print(f"Min/Med/Max heights: {min_display} / {med_display} / {max_display}")

    def _significant_gap(raw: int, kept: int) -> bool:
        if raw <= 0:
            return False
        if kept <= 0:
            return raw >= 5
        return raw >= kept * 3 and raw - kept >= 5

    if (
        tables_count == 0
        and _significant_gap(text_raw, kept_by_layer)
        and _significant_gap(text_raw, kept_by_height)
    ):
        print(
            "[AUDIT] Recommendation: lower --text-min-height or adjust layer include."
        )

    print(summary_message)


def _am_bor_included_from_candidates(*candidates: Mapping[str, Any] | None) -> bool:
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        flag = candidate.get("am_bor_included")
        if isinstance(flag, bool):
            if flag:
                return True
        elif flag:
            return True
    return False


def _ordered_hole_row(row: Mapping[str, object]) -> dict[str, object]:
    ordered: dict[str, object] = {}
    preferred_order = ("hole", "qty", "ref", "side", "desc")
    for key in preferred_order:
        if key in row:
            ordered[key] = row[key]
    for key in sorted(row.keys()):
        if key not in ordered:
            ordered[key] = row[key]
    return ordered


def _build_hole_rows_artifact(
    rows: Iterable[Mapping[str, object]] | None,
    *,
    qty_sum: int,
    hole_count: int | None,
    provenance: object,
    source: object,
) -> dict[str, object]:
    serialized_rows: list[dict[str, object]] = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        serialized_rows.append(_ordered_hole_row(row))

    artifact: dict[str, object] = {"rows": serialized_rows, "qty_sum": int(qty_sum)}
    if hole_count not in (None, ""):
        try:
            artifact["hole_count"] = int(float(hole_count))
        except Exception:
            artifact["hole_count"] = hole_count
    if provenance not in (None, ""):
        artifact["provenance"] = provenance
    if source not in (None, ""):
        artifact["source"] = source
    return artifact


def _ordered_totals_map(totals: Mapping[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    preferred = (
        "tap",
        "tap_front",
        "tap_back",
        "counterbore",
        "counterbore_front",
        "counterbore_back",
        "drill",
        "spot",
        "jig_grind",
    )
    for key in preferred:
        if key in totals:
            ordered[key] = totals[key]
    for key in sorted(totals.keys()):
        if key not in ordered:
            ordered[key] = totals[key]
    return ordered


def _build_ops_totals_artifact(ops_summary: Mapping[str, object] | None) -> dict[str, object] | None:
    if not isinstance(ops_summary, Mapping):
        return None

    totals_raw = ops_summary.get("totals") if isinstance(ops_summary, Mapping) else None
    totals_ordered = _ordered_totals_map(totals_raw) if isinstance(totals_raw, Mapping) else None

    artifact: dict[str, object] = {}
    if totals_ordered:
        artifact["totals"] = totals_ordered

    supplemental_keys = (
        "tap_total",
        "cbore_total",
        "csk_total",
        "actions_total",
        "back_ops_total",
        "flip_required",
    )
    for key in supplemental_keys:
        if key in ops_summary:
            artifact[key] = ops_summary[key]

    if "source" in ops_summary:
        artifact["source"] = ops_summary["source"]

    return artifact or None


def _row_desc(row: Mapping[str, Any]) -> str:
    for key in ("desc", "description", "text", "hole"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                text = str(value)
            except Exception:
                continue
            return " ".join(text.split())
    return ""


def _infer_row_kind(row: Mapping[str, Any]) -> str:
    kind_value = row.get("kind")
    if kind_value not in (None, ""):
        return str(kind_value)
    desc_text = _row_desc(row)
    if not desc_text:
        return "-"
    try:
        actions = geo_extractor.classify_op_row(desc_text)
    except Exception:
        actions = []
    best = "-"
    for action in actions:
        candidate = action.get("kind")
        if candidate in (None, ""):
            continue
        candidate_text = str(candidate)
        if candidate_text and candidate_text != "unknown":
            return candidate_text
        if best == "-":
            best = candidate_text or "-"
    return best


def _normalize_circle_entry(entry: Mapping[str, Any]) -> dict[str, float] | None:
    if not isinstance(entry, Mapping):
        return None
    x_val = entry.get("x")
    if x_val in (None, ""):
        for key in ("x_in", "cx", "center_x", "centerX", "pos_x"):
            if entry.get(key) not in (None, ""):
                x_val = entry.get(key)
                break
    y_val = entry.get("y")
    if y_val in (None, ""):
        for key in ("y_in", "cy", "center_y", "centerY", "pos_y"):
            if entry.get(key) not in (None, ""):
                y_val = entry.get(key)
                break
    center_candidate = entry.get("center") or entry.get("point")
    if isinstance(center_candidate, Sequence) and not isinstance(center_candidate, (str, bytes, bytearray)):
        if len(center_candidate) >= 2:
            if x_val in (None, ""):
                x_val = center_candidate[0]
            if y_val in (None, ""):
                y_val = center_candidate[1]
    dia_val = entry.get("dia_in")
    if dia_val in (None, ""):
        for key in ("diam_in", "diameter_in", "diameter", "dia", "d_in"):
            if entry.get(key) not in (None, ""):
                dia_val = entry.get(key)
                break
    radius_val = entry.get("radius_in")
    if radius_val in (None, ""):
        for key in ("radius", "rad_in", "r", "rad"):
            if entry.get(key) not in (None, ""):
                radius_val = entry.get(key)
                break
    x_norm = _coerce_float(x_val)
    y_norm = _coerce_float(y_val)
    dia_norm = _coerce_float(dia_val)
    radius_norm = _coerce_float(radius_val)
    if dia_norm is None and radius_norm is not None:
        dia_norm = radius_norm * 2.0
    result: dict[str, float] = {}
    if x_norm is not None:
        result["x"] = x_norm
    if y_norm is not None:
        result["y"] = y_norm
    if dia_norm is not None:
        result["dia"] = dia_norm
    radius_value = radius_norm if radius_norm is not None else None
    if radius_value is None and dia_norm is not None:
        radius_value = dia_norm / 2.0
    if radius_value is not None:
        result["radius"] = radius_value
    return result or None


def _normalize_circle_list(candidate: Any) -> list[dict[str, float]]:
    if candidate is None:
        return []
    if isinstance(candidate, Mapping):
        nested_keys = (
            "records",
            "items",
            "values",
            "samples",
            "circles",
            "entries",
            "points",
            "data",
        )
        for key in nested_keys:
            nested = candidate.get(key)
            if nested not in (None, ""):
                normalized = _normalize_circle_list(nested)
                if normalized:
                    return normalized
        normalized_entry = _normalize_circle_entry(candidate)
        return [normalized_entry] if normalized_entry else []
    if isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes, bytearray)):
        result: list[dict[str, float]] = []
        for item in candidate:
            if isinstance(item, Mapping):
                normalized_entry = _normalize_circle_entry(item)
                if normalized_entry:
                    result.append(normalized_entry)
        return result
    return []


def _gather_circle_samples(*candidates: Mapping[str, Any] | None) -> list[dict[str, float]]:
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        sample_keys = (
            "samples",
            "sample_circles",
            "circle_samples",
            "kept_samples",
            "kept_records",
            "records",
            "circles",
        )
        for key in sample_keys:
            samples = candidate.get(key)
            normalized = _normalize_circle_list(samples)
            if normalized:
                return normalized
        normalized_direct = _normalize_circle_list(candidate)
        if normalized_direct:
            return normalized_direct
    return []


def _normalize_guard_records(value: Any, default_guard: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if value is None:
        return records
    if isinstance(value, Mapping):
        for guard_name, payload in value.items():
            guard_label = str(guard_name or default_guard)
            for entry in _normalize_circle_list(payload):
                record = dict(entry)
                record["guard"] = guard_label
                records.append(record)
        if not records:
            normalized_entry = _normalize_circle_entry(value)
            if normalized_entry:
                normalized_entry["guard"] = default_guard
                records.append(normalized_entry)
        return records
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if isinstance(item, Mapping):
                guard_label = str(item.get("guard") or item.get("reason") or default_guard)
                nested = False
                for nested_key in ("records", "items", "samples", "circles"):
                    nested_payload = item.get(nested_key)
                    normalized_nested = _normalize_circle_list(nested_payload)
                    if normalized_nested:
                        for entry in normalized_nested:
                            record = dict(entry)
                            record["guard"] = guard_label
                            records.append(record)
                        nested = True
                if nested:
                    continue
                normalized_entry = _normalize_circle_entry(item)
                if normalized_entry:
                    normalized_entry["guard"] = guard_label
                    records.append(normalized_entry)
        return records
    return records


def _collect_guard_drops(*candidates: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    guard_records: list[dict[str, Any]] = []
    guard_keys = (
        "guard_drops",
        "drops",
        "dropped",
        "drop_manifest",
        "dropped_circles",
    )
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        for key in guard_keys:
            payload = candidate.get(key)
            guard_records.extend(_normalize_guard_records(payload, key))
    return guard_records


def _extract_hole_sets(geo: Mapping[str, Any] | None) -> Any:
    if not isinstance(geo, Mapping):
        return None
    hole_sets = geo.get("hole_sets")
    if hole_sets:
        return hole_sets
    nested = geo.get("geo")
    if isinstance(nested, Mapping):
        return _extract_hole_sets(nested)
    return None


def _write_artifact(path: Path, payload: Mapping[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    except OSError as exc:  # pragma: no cover - filesystem issues
        print(f"[geo_dump] failed to write artifact {path}: {exc}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump GEO operations summary from a DXF/DWG file")
    parser.add_argument("path", nargs="?", help="Path to the DXF or DWG file")
    parser.add_argument("--no-oda", dest="use_oda", action="store_false", help="Disable ODA fallback")
    parser.add_argument("--debug", action="store_true", help="Print the first 10 rows for inspection")
    parser.add_argument(
        "--debug-entities",
        action="store_true",
        help="Print raw text table candidates from the DXF/DWG",
    )
    parser.add_argument(
        "--dump-table",
        action="store_true",
        help="Dump stitched rows to CSV and print the first 8 entries",
    )
    parser.add_argument(
        "--dump-geom",
        action="store_true",
        help="Dump geometry circle groups and guard drops to JSON",
    )
    parser.add_argument(
        "--dump-circles",
        action="store_true",
        help="Print sample circle centers/radii and guard drop details",
    )
    parser.add_argument(
        "--debug-scan",
        action="store_true",
        help="Emit detailed text-entity scan diagnostics",
    )
    parser.add_argument(
        "--show-helpers",
        action="store_true",
        help="Print helper resolution diagnostics",
    )
    parser.add_argument(
        "--dump-lines",
        help="Write text-line debug dump to this path (banded cells use *_bands.tsv)",
    )
    parser.add_argument(
        "--dump-bands",
        action="store_true",
        help="Print reconstructed [TABLE-X] band previews (first 30)",
    )
    parser.add_argument(
        "--all-layouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process text tables from all layouts (use --no-all-layouts to restrict)",
    )
    parser.add_argument(
        "--layout-pattern",
        "--layout-regex",
        dest="layout_patterns",
        action="append",
        help="Regex pattern to match layout names (repeatable; case-insensitive)",
    )
    parser.add_argument(
        "--layer-allow",
        dest="layer_allow",
        action="append",
        help="Restrict table/text scans to the specified layer (repeatable; use ALL to disable filtering)",
    )
    parser.add_argument(
        "--allow-layers",
        dest="allow_layers",
        help="Comma-separated glob patterns of layers to allow (use ALL to disable filtering)",
    )
    parser.add_argument(
        "--block-allow",
        dest="block_allow",
        action="append",
        help="Treat INSERTs with these block names as preferred ROI seeds (repeatable)",
    )
    parser.add_argument(
        "--block-regex",
        dest="block_regex",
        action="append",
        help="Regex pattern to match INSERT block names for ROI seeding (repeatable)",
    )
    parser.add_argument(
        "--include-layer",
        dest="include_layer",
        action="append",
        help=(
            "Regex pattern for layers to include when scanning text (repeatable; "
            f"defaults still exclude {DEFAULT_EXCLUDE_PATTERN_TEXT})"
        ),
    )
    parser.add_argument(
        "--exclude-layer",
        dest="exclude_layer",
        action="append",
        help=(
            "Regex pattern for layers to exclude when scanning text (repeatable; "
            f"defaults: {DEFAULT_EXCLUDE_PATTERN_TEXT})"
        ),
    )
    parser.add_argument(
        "--text-min-height",
        dest="text_min_height",
        type=float,
        help="Minimum text height (drawing units) for anchor text scan",
    )
    parser.add_argument(
        "--text-include-layers",
        dest="text_include_layers",
        action="append",
        metavar="REGEX",
        help="Regex pattern to include layers during anchor text scan (repeatable)",
    )
    parser.add_argument(
        "--text-exclude-layers",
        dest="text_exclude_layers",
        action="append",
        metavar="REGEX",
        help="Regex pattern to exclude layers during anchor text scan (repeatable)",
    )
    parser.add_argument(
        "--text-anchor-ratio",
        dest="text_anchor_ratio",
        type=float,
        help="Anchor height tolerance ratio for filtered text scan (e.g. 0.4 for ±40%%)",
    )
    parser.add_argument(
        "--text-layout",
        dest="text_layouts",
        action="append",
        metavar="NAME",
        help="Restrict anchor text scan to the specified layout name (repeatable)",
    )
    parser.add_argument(
        "--no-exclude-layer",
        dest="no_exclude_layer",
        action="store_true",
        help=(
            "Disable the default text-layer exclusions "
            f"({DEFAULT_EXCLUDE_PATTERN_TEXT}) before applying any --exclude-layer"
            " filters"
        ),
    )
    parser.add_argument(
        "--scan-acad-tables",
        action="store_true",
        help="Print ACAD_TABLE inventory details",
    )
    parser.add_argument(
        "--trace-acad",
        action="store_true",
        help="Enable detailed AutoCAD table tracing diagnostics",
    )
    parser.add_argument(
        "--depth-max",
        type=int,
        metavar="N",
        help="Override block INSERT recursion depth for ACAD tables",
    )
    parser.add_argument(
        "--show-rows",
        type=int,
        metavar="N",
        help="Print the first N rows as qty | ref | side | desc",
    )
    parser.add_argument(
        "--force-text",
        action="store_true",
        help="Force publishing text fallback rows when available",
    )
    parser.add_argument(
        "--pipeline",
        choices=("auto", "acad", "text", "geom"),
        default="auto",
        help=(
            "Select the extraction pipeline: 'auto' runs ACAD first then TEXT, "
            "while 'geom' returns raw geometry rows"
        ),
    )
    parser.add_argument(
        "--allow-geom",
        action="store_true",
        help="Permit geometry rows even when using the automatic pipeline",
    )
    parser.add_argument(
        "--debug-layouts",
        action="store_true",
        help="Print layout and layer summaries after extraction",
    )
    parser.add_argument(
        "--dump-rows-csv",
        nargs="?",
        const="__AUTO__",
        default=None,
        help="Write extracted rows to CSV (optional custom path; default uses --dump-dir)",
    )
    parser.add_argument(
        "--dump-ents",
        dest="dump_ents",
        action="store_true",
        help="Write scanned text entities to debug/ents_dump.jsonl",
    )
    parser.add_argument(
        "--dump-dir",
        default="debug",
        help="Directory for debug dump artifacts (default: %(default)s)",
    )
    parser.add_argument(
        "--dump-all-text",
        action="store_true",
        help="Dump all text entities with layout metadata",
    )
    parser.add_argument(
        "--dump-text-all",
        action="store_true",
        help="Write an unfiltered text dump (CSV and JSONL)",
    )
    parser.add_argument(
        "--dump-text-min-height",
        dest="dump_text_min_height",
        type=float,
        metavar="IN",
        default=0.0,
        help="Minimum text height in drawing units for --dump-text-all (default: %(default)s)",
    )
    parser.add_argument(
        "--dump-text-include-layer",
        "--dump-text-include-layers",
        dest="dump_text_include_layers",
        action="append",
        metavar="REGEX",
        help="Regex pattern to include layers for --dump-text-all (repeatable; default: .*)",
    )
    parser.add_argument(
        "--dump-text-exclude-layer",
        "--dump-text-exclude-layers",
        dest="dump_text_exclude_layers",
        action="append",
        metavar="REGEX",
        help="Regex pattern to exclude layers for --dump-text-all (repeatable)",
    )
    parser.add_argument(
        "--dump-text-layout",
        dest="dump_text_layouts",
        action="append",
        metavar="NAME",
        help="Restrict --dump-text-all to specific layouts (repeatable)",
    )
    parser.add_argument(
        "--layouts",
        help="Comma-separated list of layouts to include when dumping text entities",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        metavar="VALUE",
        default=0.0,
        help="Minimum text height (drawing units) when dumping text entities",
    )
    parser.add_argument(
        "--layers-include",
        action="append",
        metavar="REGEX",
        help="Regex pattern to include layers when dumping text entities (repeatable)",
    )
    parser.add_argument(
        "--layers-exclude",
        action="append",
        metavar="REGEX",
        help="Regex pattern to exclude layers when dumping text entities (repeatable)",
    )
    parser.add_argument(
        "--no-layer-filter",
        action="store_true",
        help="Disable layer include/exclude filtering when dumping text entities",
    )
    parser.add_argument(
        "--no-blocks",
        dest="include_blocks",
        action="store_false",
        default=True,
        help="Skip traversing INSERT blocks when dumping text",
    )
    parser.add_argument(
        "--no-paperspace",
        dest="include_paperspace",
        action="store_false",
        default=True,
        help="Skip paperspace layouts when dumping text",
    )
    parser.add_argument(
        "--sample",
        dest="sample",
        type=int,
        metavar="N",
        help="Print the first N dumped text rows grouped by layout",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        dest="save_geo_path",
        help="Write consolidated GEO-style output to the provided path",
    )
    args = parser.parse_args(argv)

    dump_dir_path = Path(args.dump_dir).expanduser()

    if args.show_rows is not None:
        os.environ["CAD_QUOTER_SHOW_ROWS"] = str(args.show_rows)

    path = (args.path or os.environ.get("GEO_DUMP_PATH") or "").strip()
    if not path:
        path = str(DEFAULT_SAMPLE_PATH)
        print(f"[geo_dump] Using default sample: {path}")

    if args.depth_max is not None:
        os.environ["CAD_QUOTER_ACAD_DEPTH_MAX"] = str(args.depth_max)

    geo_extractor.set_trace_acad(bool(args.trace_acad))

    if args.show_helpers:
        try:
            app_module = importlib.import_module("appV5")
        except Exception as exc:
            print(f"[geo_dump] appV5 import failed: {exc}")
        else:
            module_path = getattr(app_module, "__file__", None)
            print(f"[geo_dump] appV5 import ok: {module_path or app_module}")
        acad_helper = geo_extractor._resolve_app_callable("hole_count_from_acad_table")
        text_helper = geo_extractor._resolve_app_callable("extract_hole_table_from_text")
        text_alt_helper = geo_extractor._resolve_app_callable("hole_count_from_text_table")
        print(
            "helpers: acad={acad} text={text} text_alt={text_alt}".format(
                acad=geo_extractor._describe_helper(acad_helper),
                text=geo_extractor._describe_helper(text_helper),
                text_alt=geo_extractor._describe_helper(text_alt_helper),
            )
        )

    if args.debug_entities:
        os.environ["CAD_QUOTER_DEBUG_ENTITIES"] = "1"

    try:
        doc = geo_extractor._load_doc_for_path(Path(path), use_oda=args.use_oda)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[geo_dump] failed to load document: {exc}")
        return 1

    text_csv_path: str = "-"
    text_jsonl_path: str = "-"

    if args.dump_text_all:
        text_dump_opts = {
            "text_min_height": args.dump_text_min_height,
            "text_include_layers": args.dump_text_include_layers,
            "text_exclude_layers": args.dump_text_exclude_layers,
            "text_layouts": args.dump_text_layouts,
        }
        try:
            _full_entries, full_csv_path, full_jsonl_path = dump_all_text(
                doc,
                dump_dir_path,
                text_dump_opts,
            )
        except Exception as exc:
            print(f"[TEXT-DUMP] full dump failed: {exc}")
        else:
            # Update summary paths even if --dump-all-text also runs later.
            text_csv_path = str(full_csv_path)
            text_jsonl_path = str(full_jsonl_path)

    if args.dump_all_text:
        include_layers: list[str] | None = None
        exclude_layers: list[str] | None = None
        if args.no_layer_filter:
            print("[TEXT-DUMP] layer filtering disabled")
        else:
            include_layers = _normalize_pattern_args(args.layers_include)
            exclude_layers = _normalize_pattern_args(args.layers_exclude)
            if include_layers:
                print(f"[TEXT-DUMP] layers_include={sorted(set(include_layers))}")
            if exclude_layers:
                print(f"[TEXT-DUMP] layers_exclude={sorted(set(exclude_layers))}")
        if args.min_height:
            print(f"[TEXT-DUMP] min_height={args.min_height}")
        layout_filter = None
        if isinstance(args.layouts, str) and args.layouts.strip():
            layout_filter = [
                value.strip()
                for value in args.layouts.split(",")
                if value and value.strip()
            ]
            if layout_filter:
                print(f"[TEXT-DUMP] layouts={layout_filter}")
        try:
            entries = geo_extractor.collect_all_text(
                doc,
                include_blocks=bool(args.include_blocks),
                include_paperspace=bool(args.include_paperspace),
                min_height=args.min_height,
                layers_include=None if args.no_layer_filter else include_layers,
                layers_exclude=None if args.no_layer_filter else exclude_layers,
                layouts=layout_filter,
            )
        except Exception as exc:
            print(f"[TEXT-DUMP] failed to collect text entities: {exc}")
            entries = []
        layout_counts = _log_text_stats(entries)
        sample_limit = args.sample
        _print_text_dump(entries, sample_limit=sample_limit, layout_counts=layout_counts)
        text_csv_written = write_text_dump_csv(entries, dump_dir_path)
        text_jsonl_written = write_text_dump_jsonl(entries, dump_dir_path)
        text_csv_path = str(text_csv_written)
        text_jsonl_path = str(text_jsonl_written)

    read_kwargs: dict[str, object] = {}

    layout_patterns = [
        value.strip()
        for value in args.layout_patterns or []
        if isinstance(value, str) and value.strip()
    ]
    layout_regex = None
    if layout_patterns:
        if len(layout_patterns) == 1:
            layout_regex = layout_patterns[0]
        else:
            layout_regex = "|".join(f"(?:{pattern})" for pattern in layout_patterns)
    layout_filters_arg: dict[str, object] | None = None
    if layout_regex or not args.all_layouts:
        layout_filters_arg = {
            "all_layouts": bool(args.all_layouts),
            "patterns": [layout_regex] if layout_regex else [],
        }
        read_kwargs["layout_filters"] = layout_filters_arg
        if getattr(args, "debug_scan", False):
            preview = geo_extractor.iter_layouts(doc, layout_filters_arg, log=False)
            layout_names = [
                str(name or "").strip() or "-"
                for name, _ in preview
            ]
            display = ", ".join(layout_names) if layout_names else "<none>"
            print(f"[geo_dump] layouts={display}")

    text_layouts = [
        value.strip()
        for value in args.text_layouts or []
        if isinstance(value, str) and value.strip()
    ]
    if text_layouts:
        read_kwargs["layout_filters"] = list(text_layouts)
        print(f"[geo_dump] text_layouts={text_layouts}")

    layer_allow_args = list(args.layer_allow or [])
    allow_layers_arg = getattr(args, "allow_layers", None)
    if allow_layers_arg:
        layer_allow_args.extend(part.strip() for part in allow_layers_arg.split(","))
    if layer_allow_args:
        os.environ["CAD_QUOTER_ACAD_ALLOW_LAYERS"] = ",".join(layer_allow_args)
    else:
        os.environ.pop("CAD_QUOTER_ACAD_ALLOW_LAYERS", None)
    if layer_allow_args:
        normalized_layers: list[str] = []
        allow_all = False
        for value in layer_allow_args:
            if value is None:
                continue
            text = value.strip()
            if not text:
                continue
            upper = text.upper()
            if upper in {"ALL", "*", "<ALL>"}:
                allow_all = True
                break
            normalized_layers.append(text)
        if allow_all:
            read_kwargs["layer_allowlist"] = None
            print("[geo_dump] layer_allow=ALL")
        else:
            allowlist_set = {layer.strip() for layer in normalized_layers if layer.strip()}
            if allowlist_set:
                read_kwargs["layer_allowlist"] = allowlist_set
                print(f"[geo_dump] layer_allow={sorted(allowlist_set)}")
    block_allow_args = args.block_allow or []
    if block_allow_args:
        normalized_blocks = [
            value.strip()
            for value in block_allow_args
            if isinstance(value, str) and value.strip()
        ]
        if normalized_blocks:
            read_kwargs["block_name_allowlist"] = normalized_blocks
            print(f"[geo_dump] block_allow={normalized_blocks}")
    block_regex_args = args.block_regex or []
    if block_regex_args:
        normalized_patterns = [
            value.strip()
            for value in block_regex_args
            if isinstance(value, str) and value.strip()
        ]
        if normalized_patterns:
            read_kwargs["block_name_regex"] = normalized_patterns
            print(f"[geo_dump] block_regex={normalized_patterns}")
    include_layer_patterns = [
        value.strip()
        for value in args.include_layer or []
        if isinstance(value, str) and value.strip()
    ]
    if include_layer_patterns:
        read_kwargs["layer_include_regex"] = list(include_layer_patterns)
        print(f"[geo_dump] include_layer={include_layer_patterns}")
    active_layer_exclude: list[str] | None = None
    exclude_layer_patterns = [
        value.strip()
        for value in args.exclude_layer or []
        if isinstance(value, str) and value.strip()
    ]
    if args.no_exclude_layer:
        read_kwargs["layer_exclude_regex"] = list(exclude_layer_patterns)
        if exclude_layer_patterns:
            print(
                "[geo_dump] exclude_layer={patterns} (defaults disabled)".format(
                    patterns=exclude_layer_patterns
                )
            )
        else:
            print("[geo_dump] exclude_layer=<none> (defaults disabled)")
        active_layer_exclude = list(exclude_layer_patterns)
    elif exclude_layer_patterns:
        combined_patterns = list(DEFAULT_TEXT_LAYER_EXCLUDE_REGEX) + list(
            exclude_layer_patterns
        )
        read_kwargs["layer_exclude_regex"] = combined_patterns
        print(
            "[geo_dump] exclude_layer={patterns} (defaults + custom)".format(
                patterns=exclude_layer_patterns
            )
        )
        active_layer_exclude = list(combined_patterns)
    if active_layer_exclude is None:
        active_layer_exclude = list(DEFAULT_TEXT_LAYER_EXCLUDE_REGEX)
    if args.debug_layouts:
        read_kwargs["debug_layouts"] = True
    if args.debug_scan:
        read_kwargs["debug_scan"] = True
    if args.force_text:
        read_kwargs["force_text"] = True
    if args.pipeline:
        read_kwargs["pipeline"] = args.pipeline
    if args.allow_geom:
        read_kwargs["allow_geom"] = True
    if args.show_helpers:
        display_regex = ", ".join(active_layer_exclude) if active_layer_exclude else "<none>"
        print(f"[geo_dump] active layer exclude regex={display_regex}")

    text_anchor_ratio = args.text_anchor_ratio
    text_min_height = args.text_min_height
    text_include_patterns = _normalize_pattern_args(args.text_include_layers)
    text_exclude_patterns = _normalize_pattern_args(args.text_exclude_layers)
    include_tuple: tuple[str, ...] | None = None
    exclude_tuple: tuple[str, ...] | None = None
    if text_include_patterns:
        include_tuple = _unique_patterns(text_include_patterns)
        if text_exclude_patterns:
            exclude_tuple = _unique_patterns(text_exclude_patterns)
        else:
            exclude_tuple = DEFAULT_TEXT_LAYER_EXCLUDE_REGEX
    elif text_exclude_patterns:
        include_tuple = DEFAULT_TEXT_LAYER_INCLUDE_REGEX
        exclude_tuple = _unique_patterns(text_exclude_patterns)
    if (
        include_tuple is None
        and exclude_tuple is None
        and (
            text_anchor_ratio is not None
            or text_min_height is not None
        )
    ):
        include_tuple = DEFAULT_TEXT_LAYER_INCLUDE_REGEX
        exclude_tuple = DEFAULT_TEXT_LAYER_EXCLUDE_REGEX
    if (
        text_anchor_ratio is not None
        or text_min_height is not None
        or include_tuple is not None
        or exclude_tuple is not None
    ):
        scan_opts_obj = TextScanOpts(
            anchor_ratio=text_anchor_ratio,
            min_height=text_min_height,
            include_layers=include_tuple,
            exclude_layers=exclude_tuple,
        )
        read_kwargs["text_scan_opts"] = scan_opts_obj
        if text_anchor_ratio is not None:
            print(f"[geo_dump] text_anchor_ratio={text_anchor_ratio}")
        if text_min_height is not None:
            print(f"[geo_dump] text_min_height={text_min_height}")
        if text_include_patterns:
            print(f"[geo_dump] text_include_layers={text_include_patterns}")
        if text_exclude_patterns:
            print(f"[geo_dump] text_exclude_layers={text_exclude_patterns}")

    extract_opts = {
        key: read_kwargs[key]
        for key in TABLE_EXTRACT_ALLOWED_KEYS
        if key in read_kwargs
    }
    try:
        extract_result = extract_for_app(doc, opts=extract_opts, **read_kwargs)
    except NoTextRowsError:
        print(NO_TEXT_ROWS_MESSAGE)
        return 2
    if not isinstance(extract_result, Mapping):
        extract_result = {}
    payload = extract_result.get("payload") if isinstance(extract_result, Mapping) else {}
    if isinstance(payload, Mapping):
        payload = dict(payload)
    else:
        payload = {}
    rows = extract_result.get("rows") if isinstance(extract_result, Mapping) else None
    if not isinstance(rows, list):
        rows = list(rows or [])  # type: ignore[arg-type]
    published = bool(rows)
    scan_info = geo_extractor.get_last_acad_table_scan() or {}
    tables_found = 0
    try:
        tables_found = int(scan_info.get("tables_found", 0))  # type: ignore[arg-type]
    except Exception:
        tables_found = 0
    geo_extractor.log_last_dxf_fallback(tables_found)
    if (
        tables_found == 0
        and not published
        and Path(path).suffix.lower() == ".dwg"
    ):
        fallback_versions = [
            "ACAD2000",
            "ACAD2004",
            "ACAD2007",
            "ACAD2013",
            "ACAD2018",
        ]
        for version in fallback_versions:
            normalized_version = geo_extractor._normalize_oda_version(version) or version
            print(f"[ACAD-TABLE] trying DXF fallback version={normalized_version}")
            try:
                fallback_doc = geo_extractor._load_doc_for_path(
                    Path(path), use_oda=args.use_oda, out_ver=normalized_version
                )
            except Exception as exc:
                print(f"[ACAD-TABLE] DXF fallback {normalized_version} failed: {exc}")
                continue
            try:
                extract_result = extract_for_app(
                    fallback_doc, opts=extract_opts, **read_kwargs
                )
            except NoTextRowsError:
                print(NO_TEXT_ROWS_MESSAGE)
                return 2
            if not isinstance(extract_result, Mapping):
                extract_result = {}
            payload = extract_result.get("payload") if isinstance(extract_result, Mapping) else {}
            if isinstance(payload, Mapping):
                payload = dict(payload)
            else:
                payload = {}
            rows = extract_result.get("rows") if isinstance(extract_result, Mapping) else None
            if not isinstance(rows, list):
                rows = list(rows or [])  # type: ignore[arg-type]
            published = bool(rows)
            scan_info = geo_extractor.get_last_acad_table_scan() or {}
            try:
                tables_found = int(scan_info.get("tables_found", 0))  # type: ignore[arg-type]
            except Exception:
                tables_found = 0
            geo_extractor.log_last_dxf_fallback(tables_found)
            if tables_found or published:
                break
    if not isinstance(rows, list):
        rows = list(rows or [])  # type: ignore[arg-type]

    errors = extract_result.get("errors") if isinstance(extract_result, Mapping) else None
    if isinstance(errors, Mapping):
        err_text = errors.get("extract_hole_table")
        if err_text:
            print(f"[geo_dump] extract_hole_table failed: {err_text}")

    extractor_table = extract_result.get("extract_hole_table") if isinstance(
        extract_result, Mapping
    ) else None
    if not isinstance(extractor_table, Mapping):
        extractor_table = {}
    payload["extract_hole_table"] = extractor_table

    final_scan = geo_extractor.get_last_acad_table_scan() or scan_info
    if args.scan_acad_tables:
        tables: list[Mapping[str, object]] = []
        tables_found_display = 0
        if isinstance(final_scan, Mapping):
            try:
                tables_found_display = int(final_scan.get("tables_found", 0))
            except Exception:
                tables_found_display = 0
            raw_tables = final_scan.get("tables")
            if isinstance(raw_tables, list):
                tables = [entry for entry in raw_tables if isinstance(entry, Mapping)]
        print(f"[ACAD-TABLE] tables_found={tables_found_display}")
        if tables:
            for entry in tables:
                owner = str(entry.get("owner") or "-")
                layer = str(entry.get("layer") or "-")
                handle = str(entry.get("handle") or "-")
                dxftype = str(entry.get("type") or entry.get("dxftype") or "-")
                try:
                    rows_val = int(entry.get("rows") or entry.get("row_count") or 0)
                except Exception:
                    rows_val = 0
                try:
                    cols_val = int(entry.get("cols") or entry.get("n_cols") or 0)
                except Exception:
                    cols_val = 0
                print(
                    "[ACAD-TABLE] hit owner={owner} layer={layer} handle={handle} rows={rows} cols={cols} type={typ}".format(
                        owner=owner,
                        layer=layer,
                        handle=handle,
                        rows=rows_val,
                        cols=cols_val,
                        typ=dxftype,
                    )
                )
        else:
            print("[ACAD-TABLE] hit owner=<none> layer=- handle=- rows=0 cols=0 type=-")

    geo = payload.get("geo")
    if not isinstance(geo, Mapping):
        geo = {}
    ops_summary = extract_result.get("ops_summary") if isinstance(
        extract_result, Mapping
    ) else None
    if not isinstance(ops_summary, Mapping):
        ops_summary = payload.get("ops_summary") if isinstance(payload, Mapping) else None
    if not isinstance(ops_summary, Mapping):
        ops_summary = geo.get("ops_summary") if isinstance(geo, Mapping) else {}
    if not isinstance(ops_summary, Mapping):
        ops_summary = {}

    rows = extract_result.get("rows") if isinstance(extract_result, Mapping) else rows
    if not isinstance(rows, list):
        rows = list(rows or [])  # type: ignore[arg-type]

    qty_sum = extract_result.get("qty_sum") if isinstance(extract_result, Mapping) else None
    if not isinstance(qty_sum, (int, float)):
        qty_sum = payload.get("qty_sum")
    if isinstance(qty_sum, (int, float)):
        qty_sum = int(float(qty_sum))
    else:
        qty_sum = _sum_qty(rows)

    holes_source = (
        extract_result.get("provenance_holes")
        if isinstance(extract_result, Mapping)
        else None
    )
    if holes_source is None:
        holes_source = payload.get("provenance_holes") if isinstance(payload, Mapping) else None
    if holes_source is None:
        provenance = geo.get("provenance") if isinstance(geo, Mapping) else {}
        if isinstance(provenance, Mapping):
            holes_source = provenance.get("holes")

    hole_count = extract_result.get("hole_count") if isinstance(extract_result, Mapping) else None
    if hole_count in (None, ""):
        hole_count = payload.get("hole_count") if isinstance(payload, Mapping) else None
    if hole_count in (None, "") and isinstance(geo, Mapping):
        hole_count = geo.get("hole_count")
    try:
        if hole_count not in (None, ""):
            hole_count = int(float(hole_count))
    except Exception:
        pass
    if hole_count in (None, ""):
        hole_count = qty_sum

    source = extract_result.get("source") if isinstance(extract_result, Mapping) else None
    if source in (None, "") and isinstance(payload, Mapping):
        source = payload.get("source")
    if source in (None, "") and isinstance(ops_summary, Mapping):
        source = ops_summary.get("source")

    source_display = source if source not in (None, "") else "-"
    if source_display != "-":
        source_display = str(source_display)
    provenance_display = holes_source if holes_source not in (None, "") else "-"
    if provenance_display != "-":
        provenance_display = str(provenance_display)
    print(
        "[EXTRACT] published rows={rows} qty_sum={qty} source={src} provenance={prov}".format(
            rows=len(rows),
            qty=qty_sum,
            src=source_display,
            prov=provenance_display,
        )
    )

    def _format_ops_counts(
        counts: Mapping[str, Any] | None,
        order: Sequence[tuple[object, str]],
    ) -> str:
        parts: list[str] = []
        for key_variant, label_text in order:
            keys: tuple[str, ...]
            if isinstance(key_variant, (list, tuple)):
                keys = tuple(str(item) for item in key_variant)
            else:
                keys = (str(key_variant),)
            value_int = 0
            for key in keys:
                source = counts.get(key) if isinstance(counts, Mapping) else None
                value_int = _int_from_value(source)
                if isinstance(counts, Mapping) and key in counts:
                    break
            parts.append(f"{label_text} {value_int}")
        if not parts:
            return "Drill 0"
        return " | ".join(parts)

    def _counts_value(counts: Mapping[str, Any] | None, *keys: str) -> int:
        if not isinstance(counts, Mapping):
            return 0
        for key in keys:
            if key not in counts:
                continue
            return _int_from_value(counts.get(key))
        return 0

    hole_sets_payload = None
    if isinstance(extract_result, Mapping):
        hole_sets_payload = extract_result.get("hole_sets")
    if hole_sets_payload is None:
        hole_sets_payload = _extract_hole_sets(geo)
    geom_holes_payload: Mapping[str, Any] | None = None
    if isinstance(extract_result, Mapping):
        geom_candidate = extract_result.get("geom_holes")
        if isinstance(geom_candidate, Mapping):
            geom_holes_payload = geom_candidate
    if geom_holes_payload is None and isinstance(payload, Mapping):
        geom_candidate = payload.get("geom_holes")
        if isinstance(geom_candidate, Mapping):
            geom_holes_payload = geom_candidate
    if geom_holes_payload is None and isinstance(geo, Mapping):
        geom_candidate = geo.get("geom_holes")
        if isinstance(geom_candidate, Mapping):
            geom_holes_payload = geom_candidate

    def _geom_summary_sources() -> list[Mapping[str, Any]]:
        sources: list[Mapping[str, Any]] = []
        if isinstance(geom_holes_payload, Mapping):
            sources.append(geom_holes_payload)
        if isinstance(hole_sets_payload, Mapping):
            sources.append(hole_sets_payload)
        elif isinstance(hole_sets_payload, Iterable) and not isinstance(
            hole_sets_payload, (str, bytes, bytearray)
        ):
            for item in hole_sets_payload:
                if isinstance(item, Mapping):
                    sources.append(item)
        return sources

    def _geom_circle_summary() -> tuple[int, int]:
        total = 0
        unique: set[float] = set()
        total_candidates: list[int] = []

        def _register_total(value: Any) -> None:
            try:
                number = int(float(value))
            except Exception:
                return
            if number > 0:
                total_candidates.append(number)

        def _ingest_group(entry: Mapping[str, Any]) -> None:
            nonlocal total
            qty_value = None
            for key in ("count", "qty", "quantity", "total"):
                candidate = entry.get(key)
                if candidate not in (None, ""):
                    qty_value = candidate
                    break
            try:
                qty = int(float(qty_value)) if qty_value not in (None, "") else 0
            except Exception:
                qty = 0
            if qty <= 0:
                return
            total += qty
            dia_candidate = None
            for key in ("dia_in", "diam_in", "diameter_in", "diam", "dia"):
                if entry.get(key) not in (None, ""):
                    dia_candidate = entry.get(key)
                    break
            try:
                if dia_candidate not in (None, ""):
                    unique.add(round(float(dia_candidate), 4))
            except Exception:
                pass

        for source_map in _geom_summary_sources():
            for key in ("total", "hole_count", "hole_count_geom", "hole_count_geom_dedup"):
                value = source_map.get(key)
                if value not in (None, ""):
                    _register_total(value)
            groups_val = source_map.get("groups") or source_map.get("hole_groups")
            if isinstance(groups_val, Iterable) and not isinstance(
                groups_val, (str, bytes, bytearray)
            ):
                for entry in groups_val:
                    if isinstance(entry, Mapping):
                        _ingest_group(entry)
            families = source_map.get("hole_diam_families_in")
            if isinstance(families, Mapping):
                for key, qty in families.items():
                    try:
                        qty_int = int(float(qty))
                    except Exception:
                        continue
                    if qty_int <= 0:
                        continue
                    total += qty_int
                    try:
                        unique.add(round(float(key), 4))
                    except Exception:
                        continue

        if total <= 0 and total_candidates:
            total = max(total_candidates)
        return (total, len(unique))

    geom_total, geom_groups = _geom_circle_summary()
    print(f"[GEOM] circles total={geom_total} unique groups={geom_groups}")

    def _normalize_count(value: Any) -> int | None:
        try:
            number = int(round(float(value)))
        except Exception:
            return None
        return number if number >= 0 else None

    geom_layer_counts: dict[str, int] | None = None

    if isinstance(geom_holes_payload, Mapping):
        raw_layer_counts = geom_holes_payload.get("layer_counts")
        layers_iter: Iterable[Any]
        if isinstance(raw_layer_counts, Mapping):
            layers_iter = [
                {"layer": key, "count": value}
                for key, value in raw_layer_counts.items()
            ]
        elif isinstance(raw_layer_counts, Iterable) and not isinstance(
            raw_layer_counts, (str, bytes, bytearray)
        ):
            layers_iter = list(raw_layer_counts)
        else:
            alt_layers = geom_holes_payload.get("layers") or geom_holes_payload.get(
                "layer_histogram"
            )
            if isinstance(alt_layers, Mapping):
                layers_iter = [
                    {"layer": key, "count": value}
                    for key, value in alt_layers.items()
                ]
            elif isinstance(alt_layers, Iterable) and not isinstance(
                alt_layers, (str, bytes, bytearray)
            ):
                layers_iter = list(alt_layers)
            else:
                layers_iter = []
        counts_by_layer: dict[str, int] = {}
        for entry in layers_iter:
            if not isinstance(entry, Mapping):
                continue
            layer_name_raw = str(entry.get("layer") or entry.get("name") or "").strip()
            layer_name = layer_name_raw.upper()
            count_val = _normalize_count(entry.get("count"))
            if not layer_name or count_val in (None, 0):
                continue
            counts_by_layer[layer_name] = counts_by_layer.get(layer_name, 0) + count_val
        if not counts_by_layer and isinstance(raw_layer_counts, Mapping):
            for key, value in raw_layer_counts.items():
                count_val = _normalize_count(value)
                if count_val in (None, 0):
                    continue
                name_key = str(key).strip().upper()
                if not name_key:
                    continue
                counts_by_layer[name_key] = counts_by_layer.get(name_key, 0) + count_val
        if counts_by_layer:
            geom_layer_counts = dict(counts_by_layer)
            layer_totals = sorted(counts_by_layer.items(), key=lambda item: (-item[1], item[0]))
            top_layers = [f"{name or '-'}:{count}" for name, count in layer_totals[:5]]
            display = ", ".join(top_layers)
            if len(layer_totals) > 5:
                display += ", …"
            print(f"[GEOM] top layers: {display}")

        raw_contributors = geom_holes_payload.get("contributors")
        contributor_records: list[Mapping[str, Any]] = []
        if isinstance(raw_contributors, Mapping):
            contributor_records = [raw_contributors]
        elif isinstance(raw_contributors, Iterable) and not isinstance(
            raw_contributors, (str, bytes, bytearray)
        ):
            contributor_records = [
                entry for entry in raw_contributors if isinstance(entry, Mapping)
            ]
        top_lines: list[str] = []
        for entry in contributor_records[:5]:
            count_val = _normalize_count(entry.get("count"))
            if count_val in (None, 0):
                continue
            layer_text = str(entry.get("layer") or "").strip().upper()
            block_text = str(entry.get("block") or "").strip().upper()
            if layer_text and block_text:
                label = f"{layer_text}/{block_text}"
            elif layer_text:
                label = layer_text
            elif block_text:
                label = block_text
            else:
                label = "-"
            top_lines.append(f"{label} : {count_val}")
        if top_lines:
            summary_display = "; ".join(top_lines) + ";"
            print(f"[GEOM] top contributors: {summary_display}")

    manifest_payload: Mapping[str, Any] | None = None
    authoritative_table_from_source = geo_extractor._table_source_is_authoritative(
        source,
        len(rows),
    )
    if isinstance(extract_result, Mapping):
        manifest_candidate = extract_result.get("manifest") or extract_result.get(
            "ops_manifest"
        )
        if isinstance(manifest_candidate, Mapping):
            manifest_payload = dict(manifest_candidate)
    if manifest_payload is None:
        manifest_payload = geo_extractor.ops_manifest(
            rows,
            geom_holes=geom_holes_payload,
            hole_sets=hole_sets_payload,
            authoritative_table=authoritative_table_from_source,
        )
    if isinstance(manifest_payload, Mapping):
        payload["ops_manifest"] = dict(manifest_payload)

    table_counts = (
        manifest_payload.get("table") if isinstance(manifest_payload, Mapping) else {}
    )
    geom_counts = (
        manifest_payload.get("geom") if isinstance(manifest_payload, Mapping) else {}
    )
    total_counts = (
        manifest_payload.get("total") if isinstance(manifest_payload, Mapping) else {}
    )
    if not isinstance(table_counts, Mapping):
        table_counts = {}
    if not isinstance(geom_counts, Mapping):
        geom_counts = {}
    if not isinstance(total_counts, Mapping):
        total_counts = {}

    table_authoritative = _anchor_authoritative_from_candidates(
        extract_result if isinstance(extract_result, Mapping) else None,
        payload if isinstance(payload, Mapping) else None,
        geo if isinstance(geo, Mapping) else None,
        ops_summary if isinstance(ops_summary, Mapping) else None,
        manifest_payload if isinstance(manifest_payload, Mapping) else None,
    )
    if not table_authoritative:
        table_authoritative = _provenance_is_anchor(holes_source)
    if not table_authoritative:
        table_authoritative = authoritative_table_from_source

    source_normalized = str(source or "").strip().lower()
    authoritative_counts = (
        source_normalized in {"acad_table", "text_table", "text_fallback"}
        and len(rows) >= 8
    )

    def _table_totals_map(counts: Mapping[str, Any] | None) -> dict[str, int]:
        totals_map = {
            "drill": _counts_value(counts, "drill", "drill_only"),
            "tap": _counts_value(counts, "tap"),
            "counterbore": _counts_value(counts, "counterbore", "cbore"),
            "counterdrill": _counts_value(counts, "counterdrill", "cdrill"),
            "jig_grind": _counts_value(counts, "jig_grind", "jig"),
        }
        return totals_map

    effective_total_counts: Mapping[str, Any]
    if authoritative_counts:
        effective_total_counts = _table_totals_map(table_counts)
    else:
        effective_total_counts = total_counts

    if authoritative_counts and isinstance(manifest_payload, dict):
        manifest_payload["authoritative_counts"] = True

    apost = "\u2019"
    print(
        "[OPS] table: "
        + _format_ops_counts(
            table_counts,
            (
                (("drill_only", "drill"), "Drill"),
                ("tap", "Tap"),
                (("counterbore", "cbore"), f"C{apost}bore"),
                (("counterdrill", "cdrill"), f"C{apost}drill"),
                (("jig_grind", "jig"), "Jig"),
            ),
        )
    )
    print(
        "[OPS] geom : "
        + _format_ops_counts(
            geom_counts,
            ((("drill_residual", "drill"), "Drill"),),
        )
    )
    print(
        "[OPS] total: "
        + _format_ops_counts(
            effective_total_counts,
            (
                ("drill", "Drill"),
                ("tap", "Tap"),
                (("counterbore", "cbore"), f"C{apost}bore"),
                (("counterdrill", "cdrill"), f"C{apost}drill"),
                (("jig_grind", "jig"), "Jig"),
            ),
        )
    )

    if authoritative_counts:
        assert _counts_value(effective_total_counts, "drill") == _counts_value(
            table_counts, "drill", "drill_only"
        ), "authoritative drill mismatch"
        assert _counts_value(effective_total_counts, "tap") == _counts_value(
            table_counts, "tap"
        ), "authoritative tap mismatch"
        assert _counts_value(effective_total_counts, "counterbore") == _counts_value(
            table_counts, "counterbore", "cbore"
        ), "authoritative counterbore mismatch"
        assert _counts_value(
            effective_total_counts, "counterdrill"
        ) == _counts_value(table_counts, "counterdrill", "cdrill"), (
            "authoritative counterdrill mismatch"
        )
        assert _counts_value(effective_total_counts, "jig_grind") == _counts_value(
            table_counts, "jig_grind", "jig"
        ), "authoritative jig mismatch"

    text_drill_total = (
        _int_from_value(table_counts.get("drill_only"))
        if isinstance(table_counts, Mapping)
        else 0
    )
    text_cbore_total = (
        _int_from_value(table_counts.get("counterbore"))
        if isinstance(table_counts, Mapping)
        else 0
    )
    text_cdrill_total = (
        _int_from_value(table_counts.get("counterdrill"))
        if isinstance(table_counts, Mapping)
        else 0
    )
    text_ops_total = text_drill_total + text_cbore_total + text_cdrill_total
    geom_total = (
        _int_from_value(geom_counts.get("total"))
        if isinstance(geom_counts, Mapping)
        else 0
    )
    if geom_total <= 0 and isinstance(geom_counts, Mapping):
        geom_total = _int_from_value(geom_counts.get("drill"))
    manifest_existing = (
        ops_summary.get("manifest") if isinstance(ops_summary, Mapping) else None
    )
    am_bor_in_text_flow = _am_bor_included_from_candidates(
        payload if isinstance(payload, Mapping) else None,
        geo if isinstance(geo, Mapping) else None,
        ops_summary if isinstance(ops_summary, Mapping) else None,
        manifest_payload if isinstance(manifest_payload, Mapping) else None,
        manifest_existing if isinstance(manifest_existing, Mapping) else None,
    )
    total_drill_count = (
        _int_from_value(effective_total_counts.get("drill"))
        if isinstance(effective_total_counts, Mapping)
        else 0
    )
    if total_drill_count > 100 or (total_drill_count and total_drill_count < 50):
        print("[GEOM] suspect overcount – check layer blacklist or bbox guard")

    suspect_payload: Mapping[str, Any] | None = None
    if isinstance(manifest_payload, Mapping):
        flags_payload = manifest_payload.get("flags")
        if isinstance(flags_payload, Mapping):
            candidate = flags_payload.get("suspect_geometry")
            if isinstance(candidate, Mapping):
                suspect_payload = candidate
    if suspect_payload is None and isinstance(ops_summary, Mapping):
        manifest_existing = ops_summary.get("manifest")
        if isinstance(manifest_existing, Mapping):
            flags_payload = manifest_existing.get("flags")
            if isinstance(flags_payload, Mapping):
                candidate = flags_payload.get("suspect_geometry")
                if isinstance(candidate, Mapping):
                    suspect_payload = candidate
    if isinstance(suspect_payload, Mapping) and not suspect_payload.get("logged"):
        geom_total = suspect_payload.get("geom_total")
        text_estimated = suspect_payload.get("text_estimated_total_drills")
        print(
            "[OPS-GUARD] suspect geometry: "
            f"geom.total={geom_total} text.estimated_total_drills={text_estimated}"
        )
        try:
            suspect_payload["logged"] = True
        except Exception:
            pass

    default_sample = "301_redacted.dwg"
    try:
        path_name = Path(path).name
    except Exception:
        path_name = ""
    if path_name.lower() == default_sample:
        print("[OPS] expect: Drill 77 | Jig 8 | Tap 21 | C'bore 60 | C'drill 3")
        try:
            drill_total = int(effective_total_counts.get("drill", 0))
        except Exception:
            drill_total = 0
        try:
            jig_total = int(effective_total_counts.get("jig_grind", 0))
        except Exception:
            jig_total = 0
        try:
            tap_total = int(effective_total_counts.get("tap", 0))
        except Exception:
            tap_total = 0
        try:
            cbore_total = _counts_value(effective_total_counts, "counterbore", "cbore")
        except Exception:
            cbore_total = 0
        try:
            cdrill_total = _counts_value(effective_total_counts, "counterdrill", "cdrill")
        except Exception:
            cdrill_total = 0
        print(
            "[OPS] actual: Drill {drill} | Jig {jig} | Tap {tap} | C'bore {cbore} | C'drill {cdrill}".format(
                drill=drill_total,
                jig=jig_total,
                tap=tap_total,
                cbore=cbore_total,
                cdrill=cdrill_total,
            )
        )

    hole_rows_artifact = _build_hole_rows_artifact(
        rows,
        qty_sum=qty_sum,
        hole_count=hole_count,
        provenance=holes_source,
        source=source,
    )
    _write_artifact(ARTIFACT_DIR / "hole_rows.json", hole_rows_artifact)

    ops_totals_artifact = _build_ops_totals_artifact(ops_summary)
    if ops_totals_artifact:
        _write_artifact(ARTIFACT_DIR / "op_totals.json", ops_totals_artifact)

    ops_totals_payload: Mapping[str, Any] | None = None
    if ops_totals_artifact:
        ops_totals_payload = dict(ops_totals_artifact)
    elif isinstance(effective_total_counts, Mapping):
        ordered_totals = _ordered_totals_map(effective_total_counts)
        if ordered_totals:
            ops_totals_payload = {"totals": ordered_totals}

    ops_totals_debug_path = _write_json_debug(
        Path("debug/ops_table_totals.json"), ops_totals_payload
    )

    rebuilt_rows: list[Mapping[str, Any]] = []
    if isinstance(ops_summary, Mapping):
        summary_rows = ops_summary.get("rows")
        if isinstance(summary_rows, list):
            rebuilt_rows = [row for row in summary_rows if isinstance(row, Mapping)]
        elif isinstance(summary_rows, Iterable) and not isinstance(
            summary_rows, (str, bytes, bytearray)
        ):
            rebuilt_rows = [row for row in summary_rows if isinstance(row, Mapping)]
    if not rebuilt_rows:
        rebuilt_rows = [row for row in rows if isinstance(row, Mapping)]

    table_csv_path = _write_rows_csv(rebuilt_rows)

    if args.dump_table:
        if rebuilt_rows:
            limit = min(8, len(rebuilt_rows))
            print(f"[TABLE] dump_count={limit} rows_total={len(rebuilt_rows)}")
            for idx, row in enumerate(rebuilt_rows[:limit]):
                qty_val = row.get("qty")
                qty_display = "-"
                if qty_val not in (None, ""):
                    qty_number = _coerce_float(qty_val)
                    if qty_number is not None:
                        qty_display = str(int(round(qty_number)))
                    else:
                        qty_display = str(qty_val)
                side_val = row.get("side")
                if side_val in (None, ""):
                    side_val = row.get("face")
                side_display = str(side_val) if side_val not in (None, "") else "-"
                kind_display = _infer_row_kind(row)
                text_display = _row_desc(row)
                if text_display and len(text_display) > 180:
                    text_display = text_display[:177] + "…"
                print(
                    "[TABLE {idx:02d}] qty={qty} kind={kind} side={side} text={text}".format(
                        idx=idx,
                        qty=qty_display,
                        kind=kind_display,
                        side=side_display,
                        text=text_display,
                    )
                )
        else:
            print("[TABLE] rebuilt rows unavailable")

    geom_candidates: list[Mapping[str, Any]] = []
    if isinstance(geom_holes_payload, Mapping):
        geom_candidates.append(geom_holes_payload)
    if isinstance(payload, Mapping):
        payload_geom = payload.get("geom_holes")
        if isinstance(payload_geom, Mapping):
            geom_candidates.append(payload_geom)
    if isinstance(geo, Mapping):
        geo_geom = geo.get("geom_holes")
        if isinstance(geo_geom, Mapping):
            geom_candidates.append(geo_geom)

    circle_samples = _gather_circle_samples(*geom_candidates)
    guard_drop_samples = _collect_guard_drops(*geom_candidates)
    guard_drop_summary: dict[str, int] | None = None
    if guard_drop_samples:
        summary_counts: dict[str, int] = {}
        for entry in guard_drop_samples:
            guard_label = str(entry.get("guard") or "-")
            summary_counts[guard_label] = summary_counts.get(guard_label, 0) + 1
        guard_drop_summary = summary_counts

    if args.dump_circles:
        combined_samples: list[tuple[dict[str, Any], str]] = []
        combined_samples.extend((sample, "kept") for sample in circle_samples)
        combined_samples.extend((sample, "dropped") for sample in guard_drop_samples)

        if combined_samples:
            limit = min(10, len(combined_samples))
            print(
                "[CIRCLES] samples_shown={shown} kept_total={kept} dropped_total={dropped}".format(
                    shown=limit,
                    kept=len(circle_samples),
                    dropped=len(guard_drop_samples),
                )
            )
            for idx, (sample, status) in enumerate(combined_samples[:limit]):
                x_display = _format_float_str(sample.get("x"))
                y_display = _format_float_str(sample.get("y"))
                radius_source = sample.get("radius")
                has_radius = radius_source not in (None, "")
                if not has_radius:
                    radius_source = sample.get("dia")
                radius_number = _coerce_float(radius_source)
                if radius_number is not None and not has_radius:
                    radius_number = radius_number / 2.0
                radius_display = _format_float_str(radius_number)
                status_label = "kept" if status == "kept" else "dropped"
                reason_value: str | None = None
                for reason_key in ("reason", "guard", "note"):
                    candidate = sample.get(reason_key)
                    if candidate not in (None, ""):
                        reason_value = str(candidate)
                        break
                if reason_value and reason_value != "-":
                    status_label = f"{status_label} ({reason_value})"
                print(
                    "[CIRCLE {idx:02d}] X={x} Y={y} R={radius} STATUS={status}".format(
                        idx=idx,
                        x=x_display,
                        y=y_display,
                        radius=radius_display,
                        status=status_label,
                    )
                )
        else:
            print("[CIRCLES] sample circles unavailable")

        if guard_drop_summary:
            summary_items = sorted(guard_drop_summary.items(), key=lambda item: (-item[1], item[0]))
            summary_display = ", ".join(
                f"{name}:{count}" for name, count in summary_items[:5]
            )
            print(f"[CIRCLES] guard_drop_counts={summary_display or '-'}")
        else:
            print("[CIRCLES] guard drop samples unavailable")

    geom_json_payload = {
        "summary": {
            "circle_total": geom_total,
            "unique_diameter_count": geom_groups,
        },
        "layer_counts": geom_layer_counts or {},
        "kept_samples": [dict(sample) for sample in circle_samples],
        "guard_drops": [dict(entry) for entry in guard_drop_samples],
    }
    if guard_drop_summary:
        geom_json_payload["guard_drop_counts"] = dict(guard_drop_summary)
    geom_json_path = _write_json_debug(Path("debug/geom_circles.json"), geom_json_payload)

    if args.show_rows and rows:
        limit = max(args.show_rows, 0)
        if limit > 0:
            count = min(limit, len(rows))
            print(f"[ROWS] preview_count={count}")
            for idx, row in enumerate(rows[:count]):
                qty_display = row.get("qty") if isinstance(row, Mapping) else None
                ref_display = row.get("ref") if isinstance(row, Mapping) else None
                side_display = row.get("side") if isinstance(row, Mapping) else None
                desc_display = row.get("desc") if isinstance(row, Mapping) else None
                print(
                    "[ROW {idx:02d}] {qty} | {ref} | {side} | {desc}".format(
                        idx=idx,
                        qty=qty_display if qty_display not in (None, "") else "-",
                        ref=ref_display if ref_display not in (None, "") else "-",
                        side=side_display if side_display not in (None, "") else "-",
                        desc=desc_display if desc_display not in (None, "") else "",
                    )
                )
    if args.debug and rows:
        print("first_rows:")
        for idx, row in enumerate(rows[:10]):
            ref = row.get("ref") if isinstance(row, Mapping) else ""
            desc = row.get("desc") if isinstance(row, Mapping) else ""
            qty = row.get("qty") if isinstance(row, Mapping) else ""
            hole = row.get("hole") if isinstance(row, Mapping) else ""
            print(
                f"  [{idx:02d}] QTY={qty} REF={ref} SIDE={row.get('side') if isinstance(row, Mapping) else ''} DESC={desc} HOLE={hole}"
            )

    debug_payload = payload.get("debug_payload")
    if isinstance(debug_payload, Mapping):
        debug_payload = dict(debug_payload)
    else:
        debug_payload = geo_extractor.get_last_text_table_debug() or {}

    if args.dump_bands:
        row_debug = debug_payload.get("row_debug") or []
        columns = debug_payload.get("columns") or []
        if isinstance(row_debug, Mapping):
            row_debug = [row_debug]
        if isinstance(columns, Mapping):
            columns = list(columns.values())
        if isinstance(row_debug, list) and row_debug:
            print(
                f"[TABLE-Y] dump rows_total={len(row_debug)} columns={len(columns)}"
            )
            for entry in row_debug[:30]:
                if not isinstance(entry, Mapping):
                    continue
                row_idx = entry.get("index")
                try:
                    row_idx_int = int(row_idx)
                except Exception:
                    row_idx_int = -1
                cells = entry.get("cells") or []
                parts: list[str] = []
                for col_idx, cell_text in enumerate(cells):
                    preview = geo_extractor._truncate_cell_preview(str(cell_text or ""))
                    parts.append(f'C{col_idx}="{preview}"')
                preview_body = " | ".join(parts)
                label = row_idx_int if row_idx_int >= 0 else row_idx
                print(
                    f"[TABLE-X] row#{label} cols={len(cells)} | {preview_body}"
                )
        else:
            print("[TABLE-X] dump rows: no row_debug in debug payload")

    dump_base = args.dump_lines
    if args.debug_entities and len(rows) < 8 and not dump_base:
        base_name = Path(path).stem or "geo_dump"
        dump_base = str(Path.cwd() / f"{base_name}_lines.tsv")

    if dump_base:
        lines_path = Path(dump_base)
        bands_path = lines_path.with_name(f"{lines_path.stem}_bands.tsv")
        raw_lines = debug_payload.get("raw_lines") or []
        candidates = raw_lines or debug_payload.get("candidates", [])
        row_debug = debug_payload.get("row_debug", [])

        def _format_float(value: object) -> str:
            if isinstance(value, (int, float)):
                return f"{float(value):.3f}"
            try:
                return f"{float(value):.3f}"
            except Exception:
                return ""

        def _write_lines(path_obj: Path, header: str, rows_iter: list[dict[str, object]]) -> None:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with path_obj.open("w", encoding="utf-8") as handle:
                handle.write(header + "\n")
                for item in rows_iter:
                    if header.startswith("layout\t"):
                        layout = str(item.get("layout") or "")
                        in_block = "1" if item.get("in_block") else "0"
                        fields = [
                            layout,
                            in_block,
                            _format_float(item.get("x")),
                            _format_float(item.get("y")),
                            str(item.get("text") or ""),
                        ]
                    elif header.startswith("row\t"):
                        fields = [
                            str(item.get("row", "")),
                            str(item.get("col", "")),
                            _format_float(item.get("y_center")),
                            str(item.get("text") or ""),
                        ]
                    else:
                        fields = [
                            str(item.get("band", "")),
                            str(item.get("col", "")),
                            _format_float(item.get("x_center")),
                            _format_float(item.get("y_center")),
                            str(item.get("text") or ""),
                        ]
                    handle.write("\t".join(fields) + "\n")

        try:
            _write_lines(lines_path, "layout\tin_block\tx\ty\ttext", list(candidates))
            if row_debug:
                row_dump: list[dict[str, object]] = []
                for entry in row_debug:
                    if not isinstance(entry, Mapping):
                        continue
                    row_idx = entry.get("index")
                    y_center = entry.get("y")
                    cells = entry.get("cells") or []
                    for col_idx, cell_text in enumerate(cells):
                        row_dump.append(
                            {
                                "row": row_idx,
                                "col": col_idx,
                                "y_center": y_center,
                                "text": cell_text,
                            }
                        )
                if row_dump:
                    _write_lines(
                        bands_path,
                        "row\tcol\ty_center\ttext",
                        row_dump,
                    )
        except OSError as exc:  # pragma: no cover - filesystem issues
            print(f"[geo_dump] failed to write dumps: {exc}")
        else:
            print(f"[geo_dump] wrote debug dumps to {lines_path} and {bands_path}")

    rows_csv_path: Path | None = None
    if args.dump_rows_csv:
        if args.dump_rows_csv == "__AUTO__":
            csv_target = dump_dir_path / "hole_table_rows.csv"
        else:
            csv_target = Path(args.dump_rows_csv).expanduser()
        try:
            csv_target.parent.mkdir(parents=True, exist_ok=True)
            with csv_target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["qty", "ref", "side", "desc", "hole"])
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    qty_val = row.get("qty")
                    writer.writerow(
                        [
                            "" if qty_val in (None, "") else str(qty_val),
                            str(row.get("ref") or ""),
                            str(row.get("side") or ""),
                            str(row.get("desc") or ""),
                            str(row.get("hole") or ""),
                        ]
                    )
        except OSError as exc:  # pragma: no cover - filesystem issues
            print(f"[geo_dump] failed to write rows CSV: {exc}")
        else:
            rows_csv_path = csv_target
            print(f"[geo_dump] wrote rows CSV to {csv_target}")

    if args.save_geo_path:
        geom_summary_payload: dict[str, Any] = {
            "circle_total": geom_total,
            "unique_diameter_count": geom_groups,
        }
        if geom_layer_counts:
            geom_summary_payload["layer_counts"] = geom_layer_counts
        if guard_drop_summary:
            geom_summary_payload["guard_drop_counts"] = guard_drop_summary
        artifact_map = {
            "text_csv": None if text_csv_path == "-" else Path(text_csv_path),
            "text_jsonl": None if text_jsonl_path == "-" else Path(text_jsonl_path),
            "table_csv": table_csv_path,
            "geom_json": geom_json_path,
            "ops_json": ops_totals_debug_path,
            "rows_csv": rows_csv_path,
        }
        _write_geo_summary(
            args.save_geo_path,
            rows=rebuilt_rows,
            ops_totals=ops_totals_payload,
            geom_summary=geom_summary_payload,
            artifact_paths=artifact_map,
        )

    debug_info = geo_extractor.get_last_text_table_debug() or {}

    _print_chart_audit(
        debug_info,
        tables_found=tables_found,
        text_min_height=args.text_min_height,
    )

    if text_csv_path == "-" and debug_info:
        fallback_entities = debug_info.get("collected_entities")
        if isinstance(fallback_entities, Sequence):
            normalized_entries: list[dict[str, Any]] = []
            for entity in fallback_entities:
                if not isinstance(entity, Mapping):
                    continue
                normalized_entries.append(
                    {
                        "etype": entity.get("type"),
                        "layout": entity.get("layout"),
                        "layer": entity.get("layer"),
                        "height": entity.get("height"),
                        "rotation": entity.get("rotation"),
                        "insert_x": entity.get("x"),
                        "insert_y": entity.get("y"),
                        "block_path": entity.get("block_path") or [],
                        "text": entity.get("text"),
                        "raw": entity.get("raw", entity.get("text")),
                    }
                )
            if normalized_entries:
                try:
                    text_csv_written = write_text_dump_csv(normalized_entries, dump_dir_path)
                    text_jsonl_written = write_text_dump_jsonl(normalized_entries, dump_dir_path)
                except OSError as exc:
                    print(f"[TEXT-DUMP] fallback dump failed: {exc}")
                else:
                    text_csv_path = str(text_csv_written)
                    text_jsonl_path = str(text_jsonl_written)

    if args.dump_ents:
        entities = (
            debug_info.get("collected_entities") if isinstance(debug_info, Mapping) else None
        )
        target_path = dump_dir_path / "ents_dump.jsonl"
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("w", encoding="utf-8") as handle:
                for entry in entities or []:
                    if not isinstance(entry, Mapping):
                        continue
                    raw_text = entry.get("raw_text")
                    text_value = entry.get("text")
                    plain_text = entry.get("plain_text")
                    record = {
                        "type": entry.get("type"),
                        "handle": entry.get("handle"),
                        "layout": entry.get("layout"),
                        "layer": entry.get("layer"),
                        "from_block": bool(entry.get("from_block")),
                        "block_name": entry.get("block_name"),
                        "depth": entry.get("depth"),
                        "height": entry.get("height"),
                        "x": entry.get("x"),
                        "y": entry.get("y"),
                        "raw": raw_text if raw_text is not None else text_value,
                        "plain": plain_text if plain_text is not None else text_value,
                    }
                    json.dump(record, handle, ensure_ascii=False)
                    handle.write("\n")
        except OSError as exc:
            print(f"[geo_dump] failed to write entity dump: {exc}")
        else:
            print(f"[ENT-DUMP] -> {target_path}")

    def _format_counts(counts: Mapping[str, int] | None) -> str:
        if not counts:
            return "{}"
        items = sorted(counts.items(), key=lambda item: (-item[1], item[0] or ""))
        top = ", ".join(f"{name or '-'}:{count}" for name, count in items[:5])
        if len(items) > 5:
            top += ", …"
        return "{" + top + "}"

    scanned_layouts = list(dict.fromkeys(debug_info.get("scanned_layouts") or []))
    scanned_layers = list(dict.fromkeys(debug_info.get("scanned_layers") or []))
    include_patterns_dbg = list(
        dict.fromkeys(debug_info.get("layer_regex_include") or [])
    )
    exclude_patterns_dbg = list(
        dict.fromkeys(debug_info.get("layer_regex_exclude") or [])
    )
    layout_summary = ",".join(scanned_layouts) if scanned_layouts else "-"
    layer_summary = ",".join(scanned_layers) if scanned_layers else "-"
    include_summary = ",".join(include_patterns_dbg) if include_patterns_dbg else "-"
    exclude_summary = ",".join(exclude_patterns_dbg) if exclude_patterns_dbg else "-"
    rows_csv_display = str(rows_csv_path) if rows_csv_path else "-"
    text_csv_display = text_csv_path
    text_jsonl_display = text_jsonl_path
    table_csv_display = str(table_csv_path) if table_csv_path else rows_csv_display
    geom_json_display = str(geom_json_path) if geom_json_path else "-"
    ops_json_display = (
        str(ops_totals_debug_path) if ops_totals_debug_path else "-"
    )
    print(
        "[geo_dump] summary layouts={layouts} layers={layers} incl={incl} "
        "excl={excl} rows={rows} text_csv={text_csv} text_jsonl={text_jsonl} "
        "table_csv={table_csv} geom_json={geom_json} ops_json={ops_json} "
        "rows_csv={rows_csv}".format(
            layouts=layout_summary,
            layers=layer_summary,
            incl=include_summary,
            excl=exclude_summary,
            rows=len(rows),
            text_csv=text_csv_display,
            text_jsonl=text_jsonl_display,
            table_csv=table_csv_display,
            geom_json=geom_json_display,
            ops_json=ops_json_display,
            rows_csv=rows_csv_display,
        )
    )

    if args.debug_layouts:
        layer_pre = debug_info.get("layer_counts_pre")
        layer_regex = debug_info.get("layer_counts_post_regex")
        layer_post = debug_info.get("layer_counts_post_allow")
        layout_pre = debug_info.get("layout_counts_pre")
        layout_regex = debug_info.get("layout_counts_post_regex")
        layout_post = debug_info.get("layout_counts_post_allow")
        include_patterns = debug_info.get("layer_regex_include") or []
        exclude_patterns = debug_info.get("layer_regex_exclude") or []
        if include_patterns or exclude_patterns:
            print(
                "[geo_dump] layer_regex include={incl} exclude={excl}".format(
                    incl=include_patterns or "-",
                    excl=exclude_patterns or "-",
                )
            )
        print(f"[geo_dump] layer_counts_pre={_format_counts(layer_pre)}")
        if layer_regex is not None:
            print(f"[geo_dump] layer_counts_regex={_format_counts(layer_regex)}")
        print(f"[geo_dump] layer_counts_post={_format_counts(layer_post)}")
        print(f"[geo_dump] layout_counts_pre={_format_counts(layout_pre)}")
        if layout_regex is not None:
            print(f"[geo_dump] layout_counts_regex={_format_counts(layout_regex)}")
        print(f"[geo_dump] layout_counts_post={_format_counts(layout_post)}")

    return 0

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
