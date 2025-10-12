import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "appV5.py"
TARGET_HELPERS = {"_jsonify_debug_value", "_jsonify_debug_summary"}


def _load_module_ast() -> ast.AST:
    source = APP_PATH.read_text(encoding="utf-8")
    if source.startswith("\ufeff"):
        source = source.lstrip("\ufeff")
    return ast.parse(source, filename=str(APP_PATH))


def test_single_debug_helper_definitions() -> None:
    module = _load_module_ast()
    counts = {name: 0 for name in TARGET_HELPERS}
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in counts:
            counts[node.name] += 1
    assert counts == {name: 1 for name in TARGET_HELPERS}


def test_no_dynamic_debug_helper_lookups() -> None:
    module = _load_module_ast()

    def _is_dynamic_lookup(call: ast.Call) -> bool:
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "get":
            base = func.value
            if isinstance(base, ast.Call) and isinstance(base.func, ast.Name):
                if base.func.id in {"globals", "locals"}:
                    if call.args and isinstance(call.args[0], ast.Constant):
                        return call.args[0].value in TARGET_HELPERS
        if isinstance(func, ast.Name) and func.id == "getattr":
            if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
                return call.args[1].value in TARGET_HELPERS
        return False

    dynamic_calls: list[ast.Call] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and _is_dynamic_lookup(node):
            dynamic_calls.append(node)

    assert dynamic_calls == []
