import os


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        t = value.strip().lower()
        if t in {"y", "yes", "true", "1", "on"}:
            return True
        if t in {"n", "no", "false", "0", "off"}:
            return False
    return None


def _coerce_env_bool(value: str | None) -> bool:
    if value is None:
        return False
    c = _coerce_bool(value)
    return bool(c)


class _EnvBoolFlag:
    """Boolean-like object that resolves its value from the environment."""

    __slots__ = ("_name", "_disable_by")

    def __init__(self, name: str, *, disable_by=()):
        self._name = name
        self._disable_by = tuple(disable_by)

    def __bool__(self) -> bool:  # pragma: no cover - exercised via callers
        for flag in self._disable_by:
            if bool(flag):
                return False
        return _coerce_env_bool(os.environ.get(self._name))

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<_EnvBoolFlag name={self._name!r} value={bool(self)!r}>"


FORCE_ESTIMATOR = _EnvBoolFlag("FORCE_ESTIMATOR")
FORCE_PLANNER = _EnvBoolFlag("FORCE_PLANNER", disable_by=(FORCE_ESTIMATOR,))
