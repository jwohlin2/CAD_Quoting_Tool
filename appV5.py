from appkit.data import load_text
try:
    _DEFAULT_SYSTEM_SUGGEST = load_text("system_suggest.txt").strip()
except FileNotFoundError:  # pragma: no cover - defensive fallback
    _DEFAULT_SYSTEM_SUGGEST = ""
