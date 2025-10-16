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


FORCE_PLANNER = _coerce_env_bool(os.environ.get("FORCE_PLANNER"))
FORCE_ESTIMATOR = _coerce_env_bool(os.environ.get("FORCE_ESTIMATOR"))

# When the estimator is explicitly requested we should never force planner usage.
if FORCE_ESTIMATOR:
    FORCE_PLANNER = False
