from __future__ import annotations

from typing import Any, Dict


def _raise_unavailable(action: str, exc: Exception | None = None) -> None:
    if exc:
        raise RuntimeError(f"{action} requires Tkinter, which is unavailable: {exc}") from exc
    raise RuntimeError(f"{action} requires Tkinter, which is unavailable.")


try:
    import tkinter as tk  # type: ignore
    from tkinter import filedialog, messagebox, scrolledtext, ttk  # type: ignore
    _TK_ERROR: Exception | None = None
except Exception as _e:
    tk = filedialog = messagebox = scrolledtext = ttk = None  # type: ignore
    _TK_ERROR = _e


def _ensure_tk(action: str = "Tkinter GUI") -> Dict[str, Any]:
    if _TK_ERROR is not None:
        _raise_unavailable(action, _TK_ERROR)
    if tk is None:  # type: ignore[truthy-bool]
        _raise_unavailable(action)
    return {
        "tk": tk,
        "filedialog": filedialog,
        "messagebox": messagebox,
        "scrolledtext": scrolledtext,
        "ttk": ttk,
    }


__all__ = ["tk", "filedialog", "messagebox", "scrolledtext", "ttk", "_ensure_tk"]

