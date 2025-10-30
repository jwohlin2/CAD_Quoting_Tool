from __future__ import annotations

from tkinter import filedialog
from typing import Any


def _ensure_settings_dict(target: Any) -> dict[str, Any]:
    settings = getattr(target, "settings", None)
    if not isinstance(settings, dict):
        settings = {}
        try:
            setattr(target, "settings", settings)
        except Exception:
            pass
    return settings


def _save_settings(target: Any, settings: dict[str, Any]) -> None:
    services = getattr(target, "llm_services", None)
    if services is None:
        return
    save_settings = getattr(services, "save_settings", None)
    if not callable(save_settings):
        return
    try:
        save_settings(getattr(target, "settings_path", None), settings)
    except Exception:
        pass


def _clear_pricing_cache(target: Any) -> None:
    pricing = getattr(target, "pricing", None)
    if pricing is None:
        return
    clear_cache = getattr(pricing, "clear_cache", None)
    if not callable(clear_cache):
        return
    try:
        clear_cache()
    except Exception:
        pass


def _set_status(target: Any, message: str) -> None:
    status_var = getattr(target, "status_var", None)
    if status_var is None:
        return
    setter = getattr(status_var, "set", None)
    if not callable(setter):
        return
    try:
        setter(message)
    except Exception:
        pass


def set_material_vendor_csv(target: Any) -> str | None:
    path = filedialog.askopenfilename(
        parent=target,
        title="Select Material Vendor CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not path:
        return None

    settings = _ensure_settings_dict(target)
    settings["material_vendor_csv"] = path

    _save_settings(target, settings)
    _clear_pricing_cache(target)
    _set_status(target, f"Material vendor CSV set to {path}")

    return path


def clear_material_vendor_csv(target: Any) -> None:
    settings = _ensure_settings_dict(target)
    settings["material_vendor_csv"] = ""

    _save_settings(target, settings)
    _clear_pricing_cache(target)
    _set_status(target, "Material vendor CSV cleared.")
