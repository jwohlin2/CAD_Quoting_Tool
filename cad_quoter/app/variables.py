"""Shared helpers for loading and sanitizing estimator variables sheets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING, TypeAlias, cast

from cad_quoter.app.optional_loaders import pd
from cad_quoter.config import logger
from cad_quoter.io.csv_utils import read_csv_as_dicts as _read_csv_as_dicts
from cad_quoter.io.csv_utils import sniff_delimiter as _sniff_csv_delimiter
from cad_quoter.resources import default_master_variables_csv

if TYPE_CHECKING:  # pragma: no cover - only used for type checking
    from pandas import DataFrame as PandasDataFrame
else:  # pragma: no cover - pandas is optional at runtime
    PandasDataFrame: TypeAlias = Any

try:  # pragma: no cover - optional dependency guard mirrors app usage
    _HAS_PANDAS = bool(pd is not None)
except NameError:  # pragma: no cover - defensive fallback for static analysers
    _HAS_PANDAS = False

CORE_COLS = ["Item", "Example Values / Options", "Data Type / Input Method"]

_MASTER_VARIABLES_CACHE: dict[str, Any] = {
    "loaded": False,
    "core": None,
    "full": None,
}


def _coerce_core_types(df_core: PandasDataFrame) -> PandasDataFrame:
    """Light normalization for estimator expectations."""

    core = df_core.copy()
    core["Item"] = core["Item"].astype(str)
    core["Data Type / Input Method"] = core["Data Type / Input Method"].astype(str).str.lower()
    # Leave "Example Values / Options" as-is (can be text or number); estimator coerces later.
    return core


def sanitize_vars_df(df_full: PandasDataFrame) -> PandasDataFrame:
    """Return a sanitized copy containing only the estimator's core columns."""

    if pd is None:  # pragma: no cover - defensive guard for static analysers
        raise RuntimeError("pandas is required to sanitize variables data frames")

    # Try to map any variant header names to our canon names (case/space tolerant)
    canon = {str(c).strip().lower(): c for c in df_full.columns}

    # Build list of the *actual* columns that correspond to CORE_COLS (if present)
    actual: list[str | None] = []
    for want in CORE_COLS:
        key = want.strip().lower()
        # allow loose matches for common variants
        candidates = [
            canon.get(key),
            canon.get(key.replace(" / ", " ").replace("/", " ")),
            canon.get(key.replace(" ", "")),
        ]
        col = next((c for c in candidates if c in df_full.columns), None)
        actual.append(col)

    # Start with what we can find; add any missing columns as empty
    core = pd.DataFrame()
    for want, col in zip(CORE_COLS, actual):
        if col is not None:
            core[want] = df_full[col]
        else:
            core[want] = "" if want != "Example Values / Options" else None

    return _coerce_core_types(core)


def read_variables_file(
    path: str, return_full: bool = False
) -> PandasDataFrame | tuple[PandasDataFrame, PandasDataFrame]:
    """
    Read .xlsx/.csv, keep original data intact, and return a sanitized copy for the estimator.
    - If return_full=True, returns (core_df, full_df); otherwise returns core_df only.
    """

    if not _HAS_PANDAS or pd is None:
        raise RuntimeError("pandas required (conda/pip install pandas)")

    assert pd is not None  # hint for type checkers

    lp = path.lower()
    if lp.endswith(".xlsx"):
        # Prefer a sheet named "Variables" if it exists; else first sheet
        xl = pd.ExcelFile(path)
        sheet_name = "Variables" if "Variables" in xl.sheet_names else xl.sheet_names[0]
        df_full = pd.read_excel(path, sheet_name=sheet_name)
    elif lp.endswith(".csv"):
        encoding = "utf-8-sig"
        try:
            with open(path, encoding=encoding) as sniff:
                header_line = sniff.readline()
        except Exception:
            header_line = ""
        delimiter = _sniff_csv_delimiter(header_line)

        read_csv_kwargs: dict[str, Any] = {"encoding": encoding}
        if delimiter == "\t":
            read_csv_kwargs["sep"] = "\t"

        try:
            df_full = pd.read_csv(path, **read_csv_kwargs)
        except Exception as csv_err:
            try:
                normalized_dicts = _read_csv_as_dicts(
                    path,
                    encoding=encoding,
                    delimiter=delimiter,
                )
            except Exception:
                raise csv_err

            try:
                df_full = pd.DataFrame(normalized_dicts)
            except Exception:
                raise csv_err
    else:
        raise ValueError("Variables must be .xlsx or .csv")

    core = sanitize_vars_df(df_full)

    return (core, df_full) if return_full else core


def _load_master_variables() -> tuple[PandasDataFrame | None, PandasDataFrame | None]:
    """Load the packaged master variables sheet once and serve cached copies."""

    if not _HAS_PANDAS or pd is None:
        return (None, None)

    assert pd is not None  # hint for type checkers

    global _MASTER_VARIABLES_CACHE
    cache = _MASTER_VARIABLES_CACHE

    if cache.get("loaded"):
        core_cached = cache.get("core")
        full_cached = cache.get("full")

        core_copy: PandasDataFrame | None = None
        if (
            _HAS_PANDAS
            and pd is not None
            and isinstance(core_cached, pd.DataFrame)
        ):
            core_copy = core_cached.copy()

        full_copy: PandasDataFrame | None = None
        if (
            _HAS_PANDAS
            and pd is not None
            and isinstance(full_cached, pd.DataFrame)
        ):
            full_copy = full_cached.copy()

        return (core_copy, full_copy)

    master_path = default_master_variables_csv()
    fallback = Path(r"D:\CAD_Quoting_Tool\Master_Variables.csv")
    if not master_path.exists() and fallback.exists():
        master_path = fallback
    if not master_path.exists():
        cache["loaded"] = True
        cache["core"] = None
        cache["full"] = None
        return (None, None)

    try:
        core_df, full_df = read_variables_file(str(master_path), return_full=True)
        core_df = cast(PandasDataFrame, core_df)
        full_df = cast(PandasDataFrame, full_df)
    except Exception:
        logger.warning("Failed to load master variables CSV from %s", master_path, exc_info=True)
        cache["loaded"] = True
        cache["core"] = None
        cache["full"] = None
        return (None, None)

    cache["loaded"] = True
    cache["core"] = core_df
    cache["full"] = full_df

    return (core_df.copy(), full_df.copy())


def find_variables_near(cad_path: str):
    """Look for ``variables.*`` in the same folder, then one level up."""

    import os

    folder = os.path.dirname(cad_path)
    names = ["variables.xlsx", "variables.csv"]
    subs = ["variables", "vars"]

    def _scan(dirpath: str) -> str | None:
        try:
            listing = os.listdir(dirpath)
        except Exception:
            return None
        low = {e.lower(): e for e in listing}
        for n in names:
            if n in low:
                return os.path.join(dirpath, low[n])
        for e in listing:
            le = e.lower()
            if le.endswith((".xlsx", ".csv")) and any(s in le for s in subs):
                return os.path.join(dirpath, e)
        return None

    hit = _scan(folder)
    if hit:
        return hit
    parent = os.path.dirname(folder)
    if os.path.isdir(parent):
        return _scan(parent)
    return None


__all__ = [
    "CORE_COLS",
    "_coerce_core_types",
    "sanitize_vars_df",
    "read_variables_file",
    "_load_master_variables",
    "find_variables_near",
]

