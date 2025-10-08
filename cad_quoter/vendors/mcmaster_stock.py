try:
    # Prefer the dedicated helper if present at repo root
    from mcmaster_stock import lookup_sku_and_price_for_mm  # type: ignore
except Exception as e:  # pragma: no cover - optional helper
    def lookup_sku_and_price_for_mm(*_args, **_kwargs):  # type: ignore
        raise ImportError("mcmaster_stock helper not available") from e

__all__ = ["lookup_sku_and_price_for_mm"]

