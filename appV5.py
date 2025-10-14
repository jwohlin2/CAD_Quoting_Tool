def set_build_geo_from_dxf_hook(
    loader: Optional[Callable[[str], Dict[str, Any]]]
) -> None:
    """Register a callable used by :func:`build_geo_from_dxf`."""

    if loader is not None and not callable(loader):
        raise TypeError("DXF metadata hook must be callable or ``None``")

    global _build_geo_from_dxf_hook
    _build_geo_from_dxf_hook = loader


