from __future__ import annotations

import sys
import importlib.machinery
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Callable, Dict

import pytest


# ----- stub heavy optional dependencies before importing application code -----


def _install_runtime_dependency_stubs() -> None:
    if "requests" not in sys.modules:
        requests_stub = types.ModuleType("requests")
        requests_stub.__spec__ = ModuleSpec("requests", loader=None)
        sys.modules["requests"] = requests_stub

    if "bs4" not in sys.modules:
        bs4_stub = types.ModuleType("bs4")
        bs4_stub.__spec__ = ModuleSpec("bs4", loader=None)

        class _BeautifulSoup:  # pragma: no cover - behaviour not needed in tests
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        bs4_stub.BeautifulSoup = _BeautifulSoup
        sys.modules["bs4"] = bs4_stub

    if "lxml" not in sys.modules:
        lxml_stub = types.ModuleType("lxml")
        lxml_stub.__spec__ = ModuleSpec("lxml", loader=None)
        sys.modules["lxml"] = lxml_stub


_install_runtime_dep_stubs = _install_runtime_dependency_stubs


def _install_ocp_stubs() -> None:
    if "OCP" in sys.modules:
        return

    ocp = types.ModuleType("OCP")
    ocp.__path__ = []  # type: ignore[attr-defined]
    sys.modules["OCP"] = ocp

    def add_submodule(name: str, attrs: Dict[str, object]) -> None:
        module = types.ModuleType(f"OCP.{name}")
        module.__dict__.update(attrs)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[f"OCP.{name}"] = module
        setattr(ocp, name, module)

    class _Dummy:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, *args, **kwargs):
            return self.__class__(*args, **kwargs)

    class _DummyTopoDS:
        @staticmethod
        def Face_s(shape):
            return shape

    add_submodule("TopAbs", {"TopAbs_EDGE": object(), "TopAbs_FACE": object()})
    add_submodule("TopExp", {"TopExp": _Dummy, "TopExp_Explorer": _Dummy})
    add_submodule("TopTools", {"TopTools_IndexedDataMapOfShapeListOfShape": _Dummy})
    add_submodule(
        "TopoDS",
        {
            "TopoDS": _DummyTopoDS,
            "TopoDS_Face": _Dummy,
            "TopoDS_Shape": _Dummy,
            "TopoDS_Compound": _Dummy,
            "TopoDS_Edge": _Dummy,
            "TopoDS_Solid": _Dummy,
            "TopoDS_Shell": _Dummy,
            "topods": _Dummy,
        },
    )
    add_submodule("BRep", {"BRep_Tool": _Dummy, "BRep_Builder": _Dummy})
    add_submodule("BRepAdaptor", {"BRepAdaptor_Surface": _Dummy})
    def _dummy_func(*args, **kwargs):
        return None

    add_submodule(
        "BRepBndLib",
        {
            "Add": _dummy_func,
            "Add_s": _dummy_func,
            "BRepBndLib_Add": _dummy_func,
            "brepbndlib_Add": _dummy_func,
            "BRepBndLib": _Dummy,
        },
    )
    add_submodule("STEPControl", {"STEPControl_Reader": _Dummy})
    add_submodule("IFSelect", {"IFSelect_RetDone": _Dummy})
    add_submodule("ShapeFix", {"ShapeFix_Shape": _Dummy})
    add_submodule("BRepCheck", {"BRepCheck_Analyzer": _Dummy})
    add_submodule("Bnd", {"Bnd_Box": _Dummy})
    add_submodule("TopLoc", {"TopLoc_Location": _Dummy})
    add_submodule("BRepGProp", {"BRepGProp": _Dummy})
    add_submodule("GProp", {"GProp_GProps": _Dummy})

    class _AutoAttrModule(types.ModuleType):
        def __getattr__(self, item):
            dummy = _Dummy
            setattr(self, item, dummy)
            return dummy

    class _AutoPackage(types.ModuleType):
        def __getattr__(self, item):
            full = f"{self.__name__}.{item}"
            mod = _AutoAttrModule(full)
            mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[full] = mod
            setattr(self, item, mod)
            return mod

    occ_pkg = _AutoPackage("OCC")
    occ_pkg.__path__ = []  # type: ignore[attr-defined]
    core_pkg = _AutoPackage("OCC.Core")
    core_pkg.__path__ = []  # type: ignore[attr-defined]
    occ_pkg.Core = core_pkg
    sys.modules["OCC"] = occ_pkg
    sys.modules["OCC.Core"] = core_pkg

    def _make_core_module(name: str):
        module = types.ModuleType(name)

        def _getattr(attr):
            dummy = _Dummy
            setattr(module, attr, dummy)
            return dummy

        module.__getattr__ = _getattr  # type: ignore[attr-defined]
        return module

    core_modules = [
        "STEPControl",
        "IFSelect",
        "TopoDS",
        "BRep",
        "ShapeFix",
        "BRepCheck",
        "Bnd",
        "BRepBndLib",
        "BRepBuilderAPI",
        "BRepTools",
        "BRepPrimAPI",
        "BRepAlgoAPI",
        "BRepAdaptor",
        "TopTools",
        "TopExp",
        "TopAbs",
        "TopLoc",
        "GeomAdaptor",
        "GeomAbs",
        "ShapeAnalysis",
        "gp",
        "BRepGProp",
        "GProp",
        "IGESControl",
    ]
    for mod_name in core_modules:
        full = f"OCC.Core.{mod_name}"
        module = _make_core_module(full)
        sys.modules[full] = module
        setattr(core_pkg, mod_name, module)


def _install_llama_stub() -> None:
    if "llama_cpp" in sys.modules:
        return

    module = types.ModuleType("llama_cpp")

    class _DummyLlama:
        def __init__(self, *_, **__):  # pragma: no cover - never used
            raise RuntimeError("llama-cpp is not available in tests")

    module.Llama = _DummyLlama
    sys.modules["llama_cpp"] = module


def _install_pandas_stub() -> None:
    if "pandas" in sys.modules:
        return

    class _Mask:
        def __init__(self, values):
            self.values = [bool(v) for v in values]

        def any(self) -> bool:
            return any(self.values)

        def __iter__(self):
            return iter(self.values)

        def __len__(self):
            return len(self.values)

        def __getitem__(self, idx):
            return self.values[idx]

        def _combine(self, other, op):
            if isinstance(other, _Mask):
                other_values = other.values
            else:
                other_values = [bool(v) for v in other]
            return _Mask(op(a, b) for a, b in zip(self.values, other_values))

        def __and__(self, other):
            return self._combine(other, lambda a, b: a and b)

        def __or__(self, other):
            return self._combine(other, lambda a, b: a or b)

        def __invert__(self):
            return _Mask(not v for v in self.values)

    class _StringMethods:
        def __init__(self, series):
            self._series = series

        def fullmatch(self, pattern: str, case: bool = True):
            pat = pattern if case else pattern.lower()
            values = []
            for item in self._series.data:
                text = "" if item is None else str(item)
                if not case:
                    text = text.lower()
                values.append(text == pat)
            return _Mask(values)

        def startswith(self, prefix: str, na: bool = False):
            pref = prefix
            values = []
            for item in self._series.data:
                if item is None:
                    values.append(False)
                    continue
                text = str(item)
                values.append(text.startswith(pref))
            return _Mask(values)

        def contains(self, pattern: str, case: bool = True, regex: bool = True, na: bool = False):
            import re

            flags = 0 if case else re.IGNORECASE
            compiled = re.compile(pattern, flags) if regex else None
            needle = pattern if case else pattern.lower()
            values = []
            for item in self._series.data:
                if item is None:
                    values.append(bool(na))
                    continue
                text = str(item)
                if compiled is not None:
                    values.append(bool(compiled.search(text)))
                else:
                    haystack = text if case else text.lower()
                    values.append(needle in haystack)
            return _Mask(values)

        def lower(self):
            return Series(("" if item is None else str(item).lower()) for item in self._series.data)

    class Series:
        def __init__(self, data):
            self.data = list(data)

        def astype(self, typ):
            if typ is str:
                return Series(str(x) if x is not None else "" for x in self.data)
            raise TypeError("Stub Series only supports astype(str)")

        @property
        def str(self):
            return _StringMethods(self)

        def __iter__(self):
            return iter(self.data)

        def __getitem__(self, idx):
            if isinstance(idx, _Mask):
                return Series(val for val, flag in zip(self.data, idx) if flag)
            if isinstance(idx, (list, tuple)):
                return Series(self.data[i] for i in idx)
            return self.data[idx]

        def __len__(self):
            return len(self.data)

        class _ILoc:
            def __init__(self, series):
                self._series = series

            def __getitem__(self, idx):
                return self._series.data[idx]

        @property
        def iloc(self):
            return Series._ILoc(self)

        def fillna(self, value):
            def _replace(item):
                if item is None:
                    return value
                if isinstance(item, float) and item != item:
                    return value
                return item

            return Series(_replace(item) for item in self.data)

        def sum(self):
            total = 0.0
            for item in self.data:
                if item is None:
                    continue
                if isinstance(item, float) and item != item:
                    continue
                total += float(item)
            return total

    class DataFrame:
        def __init__(self, rows=None, columns=None):
            if rows is None:
                self._rows = []
            else:
                self._rows = [dict(row) for row in rows]
            if columns is not None:
                self.columns = list(columns)
                for row in self._rows:
                    for col in self.columns:
                        row.setdefault(col, None)
            else:
                self.columns = list(self._rows[0].keys()) if self._rows else []

        def copy(self):
            return DataFrame([dict(row) for row in self._rows])

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, key):
            return Series(row.get(key) for row in self._rows)

        def __setitem__(self, key, value):
            if isinstance(value, Series):
                data = list(value.data)
            elif isinstance(value, (list, tuple)):
                data = list(value)
            else:
                data = [value] * len(self._rows) if self._rows else []

            if not self._rows and data:
                self._rows = [{} for _ in range(len(data))]

            if len(data) < len(self._rows):
                data.extend([None] * (len(self._rows) - len(data)))
            elif len(data) > len(self._rows):
                for _ in range(len(data) - len(self._rows)):
                    self._rows.append({})

            for row, item in zip(self._rows, data):
                row[key] = item

            if key not in self.columns:
                self.columns.append(key)

        class _Loc:
            def __init__(self, df):
                self._df = df

            def __getitem__(self, key):
                if isinstance(key, tuple):
                    rows, column = key
                    if isinstance(rows, _Mask):
                        data = [row[column] for row, flag in zip(self._df._rows, rows) if flag]
                        return Series(data)
                    if isinstance(rows, int):
                        return self._df._rows[rows][column]
                else:
                    rows = key
                    if isinstance(rows, _Mask):
                        filtered = [row for row, flag in zip(self._df._rows, rows) if flag]
                        return DataFrame(filtered)
                    if isinstance(rows, int):
                        return self._df._rows[rows]
                raise TypeError("Unsupported loc access in pandas stub")

            def __setitem__(self, key, value):
                if isinstance(key, tuple):
                    rows, column = key
                    if isinstance(rows, _Mask):
                        for row, flag in zip(self._df._rows, rows):
                            if flag:
                                row[column] = value
                        return
                    if isinstance(rows, int):
                        if isinstance(value, (list, tuple)):
                            row = {col: val for col, val in zip(self._df.columns, value)}
                        else:
                            row = value
                        self._df._rows.insert(rows, row)
                        return
                elif isinstance(key, int):
                    row = {col: val for col, val in zip(self._df.columns, value)}
                    if key == len(self._df._rows):
                        self._df._rows.append(row)
                    else:
                        self._df._rows[key] = row
                    return
                raise TypeError("Unsupported loc assignment in pandas stub")

        @property
        def loc(self):
            return DataFrame._Loc(self)

        def iterrows(self):
            for idx, row in enumerate(self._rows):
                yield idx, row

    import csv

    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = DataFrame
    pandas_stub.Series = Series

    def to_numeric(values, errors="raise"):
        scalar = False
        if isinstance(values, Series):
            iterable = values.data
        elif isinstance(values, (list, tuple)):
            iterable = list(values)
        elif isinstance(values, (str, bytes)):
            iterable = [values]
            scalar = True
        else:
            try:
                iterable = list(values)
            except TypeError:
                iterable = [values]
                scalar = True

        converted = []
        for item in iterable:
            try:
                converted.append(float(item))
            except (TypeError, ValueError):
                if errors == "coerce":
                    converted.append(float("nan"))
                else:
                    raise
        if scalar and len(converted) == 1:
            return converted[0]
        return Series(converted)

    def read_csv(path, sep=",", encoding="utf-8", engine=None, **_kwargs):
        with open(path, newline="", encoding=encoding) as handle:
            if sep is None:
                sample = handle.read(1024)
                handle.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    delimiter = dialect.delimiter
                except csv.Error:
                    delimiter = ","
            else:
                delimiter = sep
            if not delimiter:
                delimiter = ","
            reader = csv.DictReader(handle, delimiter=delimiter)
            return DataFrame(reader)

    pandas_stub.read_csv = read_csv
    pandas_stub.to_numeric = to_numeric

    def notna(value):
        def _is_not_na(item):
            if item is None:
                return False
            if isinstance(item, float) and item != item:
                return False
            return True

        if isinstance(value, Series):
            return _Mask(_is_not_na(item) for item in value.data)
        return _is_not_na(value)

    pandas_stub.notna = notna
    sys.modules["pandas"] = pandas_stub


_install_runtime_dependency_stubs()

_install_ocp_stubs()
_install_llama_stub()
_install_pandas_stub()
_install_runtime_dep_stubs()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # type: ignore


from cad_quoter.domain import QuoteState  # noqa: E402  (import after stubs installed)


@pytest.fixture
def fresh_quote_state() -> QuoteState:
    """Return a new :class:`QuoteState` with empty dictionaries for convenience."""

    return QuoteState()


@pytest.fixture
def sample_geo_metrics() -> dict:
    return {
        "GEO-01_Length_mm": 120.0,
        "GEO-02_Width_mm": 60.0,
        "GEO-03_Height_mm": 25.0,
        "GEO-Volume_mm3": 180000.0,
        "GEO-SurfaceArea_mm2": 42000.0,
        "Feature_Face_Count": 8,
        "GEO_WEDM_PathLen_mm": 95.0,
    }


@pytest.fixture
def sample_geo_dataframe(sample_geo_metrics: dict) -> pd.DataFrame:
    rows = [
        {"Item": "GEO__BBox_X_mm", "Example Values / Options": 0.0, "Data Type / Input Method": "number"},
        {"Item": "Existing", "Example Values / Options": 1.0, "Data Type / Input Method": "number"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sample_pricing_table() -> dict:
    return {
        "stainless steel": {"usd_per_kg": 5.0, "usd_per_lb": 5.0 / 2.2046226218, "notes": "test"},
        "aluminum": {"usd_per_kg": 3.1, "usd_per_lb": 3.1 / 2.2046226218, "notes": "test"},
    }


@pytest.fixture
def state_builder(fresh_quote_state: QuoteState) -> Callable[[dict], QuoteState]:
    def _builder(overrides: dict | None = None) -> QuoteState:
        state = QuoteState()
        for key, value in (overrides or {}).items():
            setattr(state, key, value)
        return state

    return _builder
