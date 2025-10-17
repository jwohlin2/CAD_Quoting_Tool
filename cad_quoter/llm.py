"""Local LLM integration helpers for the CAD Quoter UI."""
from __future__ import annotations

import hashlib
import math
import json
import os
import re
import time
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from cad_quoter.utils import _dict, compact_dict, jdump, json_safe_copy
from cad_quoter.utils.render_utils import fmt_hours, fmt_money, fmt_percent
from appkit.data import load_text


def parse_llm_json(text: str) -> dict:
    """Best-effort JSON parser for model responses."""

    if not isinstance(text, str):
        return {}
    text2 = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text2, flags=re.DOTALL)
    if not m:
        return {}
    frag = m.group(0)
    try:
        return json.loads(frag)
    except Exception:
        frag2 = re.sub(r",\s*([}\]])", r"\1", frag)
        try:
            return json.loads(frag2)
        except Exception:
            return {}


SYSTEM_SUGGEST = """You are a manufacturing estimator.
Given GEO + baseline + bounds, propose bounded adjustments that improve realism.
ALWAYS return a complete JSON object with THESE KEYS, even if values are unchanged:
{
  "process_hour_multipliers": {"drilling": <float>, "milling": <float>},
  "process_hour_adders": {"inspection": <float>},
  "scrap_pct": <float>,           // fraction 0.00–0.25
  "setups": <int>,                // 1–4
  "fixture": "<string>",          // short phrase
  "notes": ["<short reason>"],
  "no_change_reason": "<why you kept near baseline, if so>"
}
All outputs MUST respect bounds in `bounds`.
Prefer small, explainable changes (±10–50%). Never leave fields out.
`signals` exposes dfm/tolerance context such as `dfm_geo` (thin walls, unique normals, deburr edge length),
`tolerance_inputs`, `default_tolerance_note`, `has_tight_tol`, `stock_catalog`, and `machine_limits`.
Use these when suggesting scrap, inspection, fixture, setup, or stock strategies and cite them in your notes.
You may use payload["seed"] heuristics (e.g., `dfm_summary`, `tolerance_focus`, `stock_focus`, `has_tight_tol`) to bias adjustments when helpful."""


try:
    SYSTEM_SUGGEST = load_text("system_suggest.txt").strip()
except FileNotFoundError:  # pragma: no cover - defensive fallback
    pass

SUGG_TO_EDITOR = {
    "scrap_pct": (
        "Scrap Percent (%)",
        lambda f: f * 100.0,
        lambda s: float(s) / 100.0,
    ),
    "setups": (
        "Number of Milling Setups",
        int,
        int,
    ),
    ("process_hour_adders", "inspection"): (
        "In-Process Inspection Hours",
        float,
        float,
    ),
    "fixture": (
        "Fixture plan",
        str,
        str,
    ),
    "fixture_build_hr": (
        "Fixture Build Hours",
        float,
        float,
    ),
    "in_process_inspection_hr": (
        "In-Process Inspection Hours",
        float,
        float,
    ),
    "cmm_minutes": (
        "CMM Run Time min",
        float,
        float,
    ),
    "packaging_hours": (
        "Packaging Labor Hours",
        float,
        float,
    ),
    "shipping_cost": (
        "Shipping Cost Override",
        float,
        float,
    ),
    "fai_required": (
        "FAIR Required",
        lambda flag: 1 if flag else 0,
        lambda raw: (
            str(raw).strip().lower() in {"1", "true", "yes", "y"}
            if raw not in (None, "")
            else False
        ),
    ),
    ("add_pass_through", "Utilities"): (
        "Utilities Cost",
        float,
        float,
    ),
    ("add_pass_through", "Shipping"): (
        "Shipping Cost",
        float,
        float,
    ),
}

EDITOR_TO_SUGG = {spec[0]: key for key, spec in SUGG_TO_EDITOR.items()}
EDITOR_FROM_UI = {spec[0]: spec[2] for _, spec in SUGG_TO_EDITOR.items()}


def _llama_missing_msg() -> str:
    return (
        "llama-cpp-python is not installed.\n"
        "Install in your env:\n"
        "  python -m pip install -U llama-cpp-python\n"
    )


def _llama_load_error_msg(model_path: str, exc: Exception) -> str:
    """Return a user-facing error message when llama.cpp fails to load."""

    path_hint = model_path.strip()
    base = [
        "Failed to initialize the local LLM.",
        (
            "Ensure the Qwen GGUF file exists and update the LLM model path "
            "(Settings → LLM or the QWEN_GGUF_PATH environment variable)."
        ),
    ]

    if path_hint:
        base.append(f"Attempted path: {path_hint}")
        if ":" in path_hint and "\\" in path_hint:
            base.append(
                "The detected path looks like a Windows drive. If you moved the "
                "tool to another machine, copy the model or point the setting to "
                "its new location."
            )

    details = str(exc).strip()
    if details:
        base.append(f"Original error: {details}")

    return "\n".join(base)


@dataclass
class _ModelConfig:
    model_path: str
    n_ctx: int
    n_gpu_layers: int
    n_threads: Optional[int]
    n_batch: int
    rope_freq_scale: Optional[float]


class _LocalLLM:
    """Minimal, tolerant wrapper around llama-cpp-python."""

    def __init__(self, config: _ModelConfig):
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(_llama_missing_msg()) from exc

        if not os.path.isfile(config.model_path):
            raise FileNotFoundError(f"Model GGUF not found: {config.model_path}")

        self._config = config
        self._Llama = Llama
        self._llm = None

    @property
    def model_path(self) -> str:
        return self._config.model_path

    @property
    def n_ctx(self) -> int:
        return self._config.n_ctx

    def _ensure(self) -> None:
        if self._llm is None:
            kwargs = compact_dict(
                {
                    "model_path": self._config.model_path,
                    "n_ctx": self._config.n_ctx,
                    "n_gpu_layers": self._config.n_gpu_layers,
                    "n_threads": self._config.n_threads,
                    "n_batch": self._config.n_batch,
                    "logits_all": False,
                    "verbose": False,
                    "rope_freq_scale": self._config.rope_freq_scale,
                }
            )
            try:
                self._llm = self._Llama(**kwargs)
            except Exception as exc:
                msg = _llama_load_error_msg(self._config.model_path, exc)
                raise RuntimeError(msg) from exc

    def create_chat_completion(self, **kwargs):
        self._ensure()
        return self._llm.create_chat_completion(**kwargs)

    def close(self) -> None:
        try:
            llm = getattr(self, "_llm", None)
            self._llm = None
            if llm is None:
                return
            try:
                getattr(llm, "sampler")
            except Exception:
                try:
                    setattr(llm, "sampler", None)
                except Exception:
                    pass
            try:
                if hasattr(llm, "close"):
                    llm.close()
            except Exception:
                pass
            try:
                del llm
            except Exception:
                pass
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


def _default_model_config(model_path: str) -> _ModelConfig:
    n_threads_env = os.getenv("QWEN_N_THREADS")
    n_threads = int(n_threads_env) if n_threads_env and n_threads_env.isdigit() else None
    rope_scale = os.getenv("ROPE_FREQ_SCALE")
    rope_freq_scale = None
    if rope_scale:
        try:
            rope_freq_scale = float(rope_scale)
        except Exception:
            rope_freq_scale = None
    return _ModelConfig(
        model_path=model_path,
        n_ctx=int(os.getenv("QWEN_N_CTX", 32768)),
        n_gpu_layers=int(os.getenv("QWEN_N_GPU_LAYERS", 0)),
        n_threads=n_threads,
        n_batch=int(os.getenv("QWEN_N_BATCH", 1024)),
        rope_freq_scale=rope_freq_scale,
    )


EventHook = Callable[[str, Dict[str, Any]], None]
ErrorHook = Callable[[Exception, Dict[str, Any]], None]


class LLMClient:
    """High-level interface for the UI to interact with the local LLM."""

    def __init__(
        self,
        model_path: str | None,
        *,
        debug_enabled: bool = False,
        debug_dir: Path | None = None,
        on_event: EventHook | None = None,
        on_error: ErrorHook | None = None,
    ) -> None:
        self._model_path = model_path or ""
        self._debug_enabled = debug_enabled
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._on_event = on_event
        self._on_error = on_error
        self._local: _LocalLLM | None = None

    # Public -----------------------------------------------------------------
    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def available(self) -> bool:
        return bool(self._model_path)

    def ask_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        context: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> tuple[dict, str, dict]:
        if not self.available:
            raise RuntimeError("LLM model path not configured")

        config = _default_model_config(self._model_path)
        if self._local is None or self._local.model_path != config.model_path or self._local.n_ctx != config.n_ctx:
            if self._local is not None:
                try:
                    self._local.close()
                except Exception:
                    pass
            self._local = _LocalLLM(config)

        chat_params = {
            "temperature": float(os.getenv("QWEN_TEMP", temperature)),
            "top_p": float(os.getenv("QWEN_TOP_P", 0.90)),
            "repeat_penalty": float(os.getenv("QWEN_REPEAT_PENALTY", 1.05)),
            "max_tokens": int(os.getenv("QWEN_MAX_TOKENS", max_tokens)),
        }
        if params:
            chat_params.update(params)

        request_payload = {
            "model": self._model_path,
            "n_ctx": config.n_ctx,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "params": chat_params,
            "context_payload": context or {},
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self._emit_event("request", request_payload)

        try:
            out = self._local.create_chat_completion(
                messages=request_payload["messages"],
                temperature=chat_params["temperature"],
                top_p=chat_params["top_p"],
                repeat_penalty=chat_params["repeat_penalty"],
                max_tokens=chat_params["max_tokens"],
            )
        except Exception as exc:
            self._emit_error(exc, request_payload)
            raise

        choices = out.get("choices") or []
        message = (choices[0] or {}).get("message") if choices else {}
        text = str((message or {}).get("content") or "")
        usage = out.get("usage", {}) or {}

        parsed = {}
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            parsed = parse_llm_json(text)
        if not isinstance(parsed, dict):
            parsed = {}

        response_snapshot = {
            "request": request_payload,
            "raw_response_text": text,
            "parsed_response": parsed,
            "usage": usage,
        }

        self._emit_event("response", response_snapshot)
        self._maybe_write_debug_snapshot(response_snapshot)

        return parsed, text, usage

    def close(self) -> None:
        if self._local is not None:
            try:
                self._local.close()
            finally:
                self._local = None

    # Private ----------------------------------------------------------------
    def _emit_event(self, kind: str, payload: Dict[str, Any]) -> None:
        if self._on_event:
            try:
                self._on_event(kind, payload)
            except Exception:
                pass

    def _emit_error(self, exc: Exception, context: Dict[str, Any]) -> None:
        if self._on_error:
            try:
                self._on_error(exc, context)
            except Exception:
                pass

    def _maybe_write_debug_snapshot(self, snapshot: Dict[str, Any]) -> None:
        if not self._debug_enabled or not self._debug_dir:
            return
        try:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            path = self._debug_dir / f"llm_snapshot_{int(time.time())}.json"
            path.write_text(
                jdump(json_safe_copy(snapshot)),
                encoding="utf-8",
            )
        except Exception:
            pass


def build_llm_sheet_prompt(geo: dict, allowed_items: list[str], params_snapshot: dict) -> tuple[str, str]:
    sys = (
        "You are a manufacturing estimator for precision machining (milling/turning/EDM/grinding). "
        "You will receive:\n"
        "1) GEO_* features extracted from CAD (units in mm/mm^2/mm^3), and\n"
        "2) a list of allowed Variables sheet row names you may edit.\n"
        "Return STRICT JSON only.\n"
        "Rules:\n"
        "- Only edit rows that are in the provided allowed_items list.\n"
        "- Use numbers only (no text). Time is in HOURS unless the row name includes 'min'.\n"
        "- Pass counts/integers for fields like '... Passes' or '... Count'.\n"
        "- Percents must be decimals (e.g., 0.20 for 20%).\n"
        "- Be conservative; if you are unsure, skip the edit.\n"
        "- You may also suggest param nudges in 'params' for: OEE_EfficiencyPct, FiveAxisMultiplier, TightToleranceMultiplier,\n"
        "  MillingConsumablesPerHr, TurningConsumablesPerHr, EDMConsumablesPerHr, GrindingConsumablesPerHr, InspectionConsumablesPerHr,\n"
        "  UtilitiesPerSpindleHr, ConsumablesFlat. Do not include other keys.\n"
    )
    u = {
        "geo": geo,
        "allowed_items": allowed_items,
        "params_snapshot": {k: params_snapshot.get(k) for k in [
            "OEE_EfficiencyPct","FiveAxisMultiplier","TightToleranceMultiplier",
            "MillingConsumablesPerHr","TurningConsumablesPerHr","EDMConsumablesPerHr",
            "GrindingConsumablesPerHr","InspectionConsumablesPerHr",
            "UtilitiesPerSpindleHr","ConsumablesFlat"
        ]},
        "output_shape": {
            "sheet_edits": [{"item": "<exact string from allowed_items>", "value": 0.0, "why": "<short reason, optional>"}],
            "params": {
                "OEE_EfficiencyPct": 1.0,
                "FiveAxisMultiplier": 1.0,
                "TightToleranceMultiplier": 1.0,
                "MillingConsumablesPerHr": 0.0,
                "TurningConsumablesPerHr": 0.0,
                "EDMConsumablesPerHr": 0.0,
                "GrindingConsumablesPerHr": 0.0,
                "InspectionConsumablesPerHr": 0.0,
                "UtilitiesPerSpindleHr": 0.0,
                "ConsumablesFlat": 0.0
            }
        }
    }
    user = textwrap.dedent(
        "Given the following, produce JSON ONLY in the shape shown above. "
        "Prefer these typical relationships:\n"
        "- WEDM: more path length/thickness -> higher 'EDM Passes' (int) or 'WEDM Hours'; adjust 'EDM Cut Rate_mm^2/min' if too aggressive.\n"
        "- Grinding: large grind volume -> adjust 'Grind MRR_mm^3/min', 'Grinding Passes', and 'Dress Frequency passes' / 'Dress Time min'.\n"
        "- Milling setups: high face count or multiple normals -> increase 'Number of Milling Setups' and 'Setup Hours / Setup'.\n"
        "- Small min wall or high area/volume -> consider 'Thin Wall Factor' and 'Tolerance Multiplier'.\n"
        "- Inspection: very small parts or tight tolerance -> bump 'CMM Run Time min' or 'Final Inspection'.\n"
        "- Only include edits you are confident about.\n\n"
        f"INPUT:\n{json.dumps(u, ensure_ascii=False)}"
    )
    return sys, user


def llm_sheet_and_param_overrides(geo: dict, df, params: dict, client: LLMClient) -> dict:
    allowed = _items_used_by_estimator(df)
    if not allowed:
        return {"sheet_edits": [], "params": {}, "meta": {}}

    sys, usr = build_llm_sheet_prompt(geo, allowed, params)
    prompt_sha = hashlib.sha256((sys + "\n" + usr).encode("utf-8")).hexdigest()

    error_text = ""
    raw_text = ""
    usage: Dict[str, Any] = {}
    try:
        parsed, raw_text, usage = client.ask_json(
            system_prompt=sys,
            user_prompt=usr,
            temperature=0.15,
            max_tokens=900,
        )
        if not isinstance(parsed, dict):
            parsed = parse_llm_json(raw_text)
        js = _dict(parsed)
        model_name = Path(client.model_path).name if client.model_path else "LLM-unavailable"
    except Exception as exc:
        js, model_name = {}, "LLM-unavailable"
        try:
            error_text = f"{type(exc).__name__}: {exc}"
        except Exception:
            error_text = "LLM error"

    sheet_edits = []
    for e in js.get("sheet_edits", []):
        item = str(e.get("item", "")); item = item.strip()
        if item and item in allowed:
            val = e.get("value", None)
            why = e.get("why", "").strip() if isinstance(e.get("why", ""), str) else ""
            if isinstance(val, (int, float, str)):
                try:
                    v = float(val)
                except Exception:
                    v = _parse_pct_like(val)
                    if v is None:
                        continue
                sheet_edits.append({"item": item, "value": v, "why": why})

    param_allow = {
        "OEE_EfficiencyPct","FiveAxisMultiplier","TightToleranceMultiplier",
        "MillingConsumablesPerHr","TurningConsumablesPerHr","EDMConsumablesPerHr",
        "GrindingConsumablesPerHr","InspectionConsumablesPerHr",
        "UtilitiesPerSpindleHr","ConsumablesFlat"
    }
    pmap = _dict(js.get("params"))
    param_whys = _dict(pmap.get("_why"))
    param_edits = {}
    for k, v in pmap.items():
        if k == "_why":
            continue
        if k in param_allow:
            try:
                param_edits[k] = float(v)
            except Exception:
                pv = _parse_pct_like(v)
                if pv is not None:
                    param_edits[k] = pv

    meta = {
        "model": model_name,
        "prompt_sha256": prompt_sha,
        "allowed_items_count": len(allowed),
        "error": error_text,
        "raw_text": raw_text,
        "usage": usage,
    }
    return {"sheet_edits": sheet_edits, "params": param_edits, "param_whys": param_whys, "allowed_items": allowed, "meta": meta}


def _jsonify_for_prompt(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify_for_prompt(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify_for_prompt(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        try:
            return value.item()  # numpy scalar
        except Exception:
            return str(value)


def run_llm_suggestions(client: LLMClient, payload: dict) -> tuple[dict, str, dict]:
    safe_payload = _jsonify_for_prompt(payload)
    try:
        prompt_body = json.dumps(safe_payload, indent=2)
    except TypeError:
        prompt_body = json.dumps(_jsonify_for_prompt(safe_payload), indent=2)
    parsed, raw, usage = client.ask_json(
        system_prompt=SYSTEM_SUGGEST,
        user_prompt=prompt_body,
        temperature=0.3,
        max_tokens=512,
        context=safe_payload,
        params={"top_p": 0.9},
    )
    if not parsed:
        parsed = parse_llm_json(raw)
    if not parsed:
        baseline = payload.get("baseline") or {}
        parsed = {
            "process_hour_multipliers": {"drilling": 1.0, "milling": 1.0},
            "process_hour_adders": {"inspection": 0.0},
            "scrap_pct": baseline.get("scrap_pct", 0.0),
            "setups": int(baseline.get("setups", 1) or 1),
            "fixture": baseline.get("fixture", "standard") or "standard",
            "notes": ["no parse; using baseline"],
            "no_change_reason": "fallback",
        }
    return parsed, raw, usage or {}


def infer_hours_and_overrides_from_geo(
    geo: dict,
    params: dict | None = None,
    rates: dict | None = None,
    *,
    client: LLMClient | None = None,
) -> dict:
    params = params or {}
    rates = rates or {}

    system = (
        "You are a senior manufacturing estimator in a tool & die/CNC job shop. "
        "Given CAD GEO features (mm/mm2/mm3), estimate realistic HOURS for a small lot of machined parts. "
        "Prefer simple, conservative estimates. Output ONLY JSON."
    )

    schema = {
        "hours": {
            "Programming_Hours": 0.0,
            "CAM_Programming_Hours": 0.0,
            "Engineering_Hours": 0.0,
            "Fixture_Build_Hours": 0.0,

            "Roughing_Cycle_Time_hr": 0.0,
            "Semi_Finish_Cycle_Time_hr": 0.0,
            "Finishing_Cycle_Time_hr": 0.0,

            "OD_Turning_Hours": 0.0,
            "ID_Bore_Drill_Hours": 0.0,
            "Threading_Hours": 0.0,
            "Cutoff_Hours": 0.0,

            "WireEDM_Hours": 0.0,
            "SinkerEDM_Hours": 0.0,

            "Grinding_Surface_Hours": 0.0,
            "Grinding_ODID_Hours": 0.0,
            "Jig_Grind_Hours": 0.0,

            "Lapping_Hours": 0.0,
            "Deburr_Hours": 0.0,
            "Tumble_Hours": 0.0,
            "Blast_Hours": 0.0,
            "Laser_Mark_Hours": 0.0,
            "Masking_Hours": 0.0,

            "InProcess_Inspection_Hours": 0.0,
            "Final_Inspection_Hours": 0.0,
            "CMM_Programming_Hours": 0.0,
            "CMM_RunTime_min": 0.0,

            "Saw_Waterjet_Hours": 0.0,
            "Assembly_Hours": 0.0,
            "Packaging_Labor_Hours": 0.0,

            "EHS_Hours": 0.0
        },
        "setups": {
            "Milling_Setups": 1,
            "Setup_Hours_per_Setup": 0.3
        },
        "inspection": {
            "FAIR_Required": False,
            "Source_Inspection_Required": False
        },
        "notes": []
    }

    prompt = f"""
GEO (mm / mm2 / mm3):
{jdump(geo, default=None)}

Rules of thumb:
- Small rectangular blocks with few features: Programming 0.2–1.0 hr, CAM 0.2–1.0 hr, Engineering 0 hr.
- Use setups 1–2 unless 3-axis accessibility is low (<0.6) or faces with unique normals > 4.
- Deburr 0.1–0.4 hr unless thin walls/freeform edges (then up to 0.8 hr).
- Final inspection ~0.2–0.6 hr; CMM only if tolerances < 0.02 mm or GD&T heavy.
- Grinding/EDM/Turning hours should be 0 unless features clearly require them.
- Never return huge numbers for tiny parts (<80 mm max dim).

Return JSON with this structure (numbers only, minutes only for CMM_RunTime_min):
{jdump(schema, default=None)}
"""

    if client and client.available:
        try:
            parsed, raw_text, _usage = client.ask_json(
                system_prompt=system,
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=1024,
            )
            if not isinstance(parsed, dict):
                parsed = parse_llm_json(raw_text)
            if isinstance(parsed, dict) and "hours" in parsed:
                return parsed
        except Exception:
            pass

    faces = float(geo.get("GEO__Face_Count", 0) or 0)
    max_dim = float(geo.get("GEO__MaxDim_mm", 0) or 0)
    min_wall = float(geo.get("GEO__MinWall_mm", 0) or 0)
    thin_wall = bool(geo.get("GEO__ThinWall_Present", False))
    wedm_len = float(geo.get("GEO_WEDM_PathLen_mm", 0) or 0)
    deburr_len = float(geo.get("GEO_Deburr_EdgeLen_mm", 0) or 0)
    complexity = float(geo.get("GEO_Complexity_0to100", 20) or 20)
    access = float(geo.get("GEO_3Axis_Accessible_Pct", 0.8) or 0.8)

    base_cycle = min(max_dim / 80.0, 2.0)
    deburr = min(0.8, 0.1 + deburr_len / 5000.0)
    deburr = deburr * (1.5 if thin_wall else 1.0)
    deburr = max(deburr, 0.15 if thin_wall else 0.1)
    # The rules-of-thumb above call out a 0.2–1.0 hr range for simple blocks.
    # When the LLM is unavailable we historically extrapolated programming
    # hours from face count, but that heuristic could explode for dense parts
    # (e.g. 500+ faces would yield 13+ hours).  Those huge defaults caused the
    # UI to display five-figure programming costs with no user input.  To keep
    # the fallback conservative, clamp the automatic estimate to a single hour
    # so that manual overrides always start from a sane baseline.
    programming = 1.0
    cam = min(2.0, 0.3 + complexity / 100.0)
    engineering = 0.0 if access > 0.6 else 0.5
    fixture = 0.0 if max_dim < 150 else 1.0

    wedm_hours = min(4.0, wedm_len / 2000.0)
    grind_hours = 0.2 if complexity > 60 else 0.0

    setups = 1
    if access < 0.6 or faces > 8:
        setups = 2
    if faces > 12:
        setups = 3

    cmm_minutes = 0.0
    if min_wall and min_wall < 1.5:
        cmm_minutes += 10
    if complexity > 70:
        cmm_minutes += 15

    return {
        "hours": {
            "Programming_Hours": round(programming, 2),
            "CAM_Programming_Hours": round(cam, 2),
            "Engineering_Hours": round(engineering, 2),
            "Fixture_Build_Hours": round(fixture, 2),
            "Roughing_Cycle_Time_hr": round(base_cycle, 2),
            "Semi_Finish_Cycle_Time_hr": round(base_cycle * 0.6, 2),
            "Finishing_Cycle_Time_hr": round(base_cycle * 0.4, 2),
            "OD_Turning_Hours": 0.0,
            "ID_Bore_Drill_Hours": round(min(2.0, faces / 50.0), 2),
            "Threading_Hours": 0.0,
            "Cutoff_Hours": 0.0,
            "WireEDM_Hours": round(wedm_hours, 2),
            "SinkerEDM_Hours": 0.0,
            "Grinding_Surface_Hours": round(grind_hours, 2),
            "Grinding_ODID_Hours": 0.0,
            "Jig_Grind_Hours": 0.0,
            "Lapping_Hours": 0.0,
            "Deburr_Hours": round(deburr, 2),
            "Tumble_Hours": 0.0,
            "Blast_Hours": 0.0,
            "Laser_Mark_Hours": 0.0,
            "Masking_Hours": 0.0,
            "InProcess_Inspection_Hours": round(0.2 + cmm_minutes / 120.0, 2),
            "Final_Inspection_Hours": round(0.3 + complexity / 300.0, 2),
            "CMM_Programming_Hours": 0.0,
            "CMM_RunTime_min": round(cmm_minutes, 1),
            "Saw_Waterjet_Hours": 0.1 if max_dim > 200 else 0.0,
            "Assembly_Hours": 0.0,
            "Packaging_Labor_Hours": 0.1,
            "EHS_Hours": 0.0,
        },
        "setups": {
            "Milling_Setups": setups,
            "Setup_Hours_per_Setup": round(0.3 + setups * 0.1, 2),
        },
        "inspection": {
            "FAIR_Required": False,
            "Source_Inspection_Required": False,
        },
        "notes": ["fallback heuristics"],
    }


# ---------------------------------------------------------------------------
# Quote explanations


def explain_quote(
    breakdown: Mapping[str, Any] | None,
    *,
    hour_trace: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    geometry: Mapping[str, Any] | None = None,
    render_state: Any | None = None,
    plan_info: Mapping[str, Any] | None = None,
) -> str:
    """Return a short natural-language explanation for a rendered quote."""

    if not isinstance(breakdown, Mapping):
        return ""

    def _coerce_float(value: Any) -> float | None:
        try:
            num = float(value)
        except Exception:
            return None
        if not math.isfinite(num):
            return None
        return num

    def _coerce_int(value: Any) -> int | None:
        num = _coerce_float(value)
        if num is None:
            return None
        try:
            return int(round(num))
        except Exception:
            return None

    def _format_pct(value: Any) -> str | None:
        pct = _coerce_float(value)
        if pct is None:
            return None
        if pct > 1.5:
            pct = pct / 100.0
        return fmt_percent(pct)

    totals = breakdown.get("totals") or {}

    currency_hint = (
        breakdown.get("currency")
        or totals.get("currency")
        or breakdown.get("currency_code")
        or totals.get("currency_code")
    )
    currency_text = str(currency_hint or "").strip()
    if not currency_text:
        currency_prefix = "$"
    elif len(currency_text) == 1 or currency_text[0] in "$€£¥₹":
        currency_prefix = currency_text
    elif currency_text.isalpha() and len(currency_text) <= 4:
        currency_prefix = f"{currency_text} "
    else:
        currency_prefix = currency_text

    def _format_money(value: Any) -> str | None:
        num = _coerce_float(value)
        if num is None:
            return None
        return fmt_money(num, currency_prefix)

    lines: list[str] = []

    geometry_map: Mapping[str, Any] | None = geometry if isinstance(geometry, Mapping) else None
    if geometry_map is None:
        geometry_candidates: list[Any] = []
        if isinstance(breakdown, Mapping):
            geometry_candidates.extend(
                breakdown.get(key)
                for key in ("geometry", "geo_context", "geometry_context", "geo")
            )
        for candidate in geometry_candidates:
            if isinstance(candidate, Mapping):
                geometry_map = candidate
                break

    def _hole_groups_present(source: Mapping[str, Any] | None) -> bool:
        if not isinstance(source, Mapping):
            return False
        for key in ("hole_groups", "hole_sets"):
            groups = source.get(key)
            if isinstance(groups, Sequence) and any(groups):
                return True
        return False

    hole_groups_flag = _hole_groups_present(geometry_map)
    if not hole_groups_flag and isinstance(breakdown, Mapping):
        hole_groups_flag = _hole_groups_present(breakdown.get("geometry"))
        if not hole_groups_flag:
            hole_groups_flag = _hole_groups_present(breakdown.get("geo_context"))

    def _extract_removal_hours(source: Any) -> float | None:
        if source is None:
            return None
        if isinstance(source, Mapping):
            direct = _coerce_float(source.get("removal_drilling_hours"))
            if direct is not None:
                return direct
            extra = source.get("extra")
            if isinstance(extra, Mapping):
                from_extra = _coerce_float(extra.get("removal_drilling_hours"))
                if from_extra is not None:
                    return from_extra
        try:
            extra_attr = getattr(source, "extra", None)
        except Exception:
            extra_attr = None
        if isinstance(extra_attr, Mapping):
            from_attr = _coerce_float(extra_attr.get("removal_drilling_hours"))
            if from_attr is not None:
                return from_attr
        return None

    removal_hr = _extract_removal_hours(render_state)
    if removal_hr is None and isinstance(breakdown, Mapping):
        removal_hr = _extract_removal_hours(breakdown.get("planner_render_state"))
    if removal_hr is None and isinstance(breakdown, Mapping):
        removal_hr = _extract_removal_hours(breakdown.get("bucket_render_state"))
    if removal_hr is None and isinstance(breakdown, Mapping):
        removal_hr = _extract_removal_hours(breakdown.get("removal_summary"))

    def _count_recognized_ops(plan_summary: Mapping[str, Any] | None) -> int:
        if not isinstance(plan_summary, Mapping):
            return 0
        ops = plan_summary.get("ops")
        if not isinstance(ops, Sequence):
            return 0
        count = 0
        for entry in ops:
            if isinstance(entry, Mapping):
                count += 1
            elif entry is not None:
                try:
                    if bool(entry):
                        count += 1
                except Exception:
                    count += 1
        return count

    plan_drilling_reasons: list[str] = []

    def _add_reason(text: str | None) -> None:
        if not text:
            return
        normalized = text.strip()
        if not normalized:
            return
        if normalized not in plan_drilling_reasons:
            plan_drilling_reasons.append(normalized)

    def _iter_plan_mappings(root: Mapping[str, Any] | None) -> Iterable[Mapping[str, Any]]:
        if not isinstance(root, Mapping):
            return []
        stack: list[Mapping[str, Any]] = [root]
        seen: set[int] = set()
        while stack:
            current = stack.pop()
            ident = id(current)
            if ident in seen:
                continue
            seen.add(ident)
            yield current
            for key in (
                "planner_pricing",
                "process_plan_summary",
                "process_plan",
                "pricing",
                "plan_summary",
                "plan",
                "bucket_state_extra",
            ):
                candidate = current.get(key) if isinstance(current, Mapping) else None
                if isinstance(candidate, Mapping):
                    stack.append(candidate)

    def _line_items_from(mapping: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
        items = mapping.get("line_items")
        if isinstance(items, Sequence):
            for entry in items:
                if isinstance(entry, Mapping):
                    yield entry

    def _check_bucket_map(mapping: Mapping[str, Any] | None) -> None:
        if not isinstance(mapping, Mapping):
            return
        for raw_key, raw_value in mapping.items():
            key_text = str(raw_key or "").strip().lower()
            if "drill" not in key_text:
                continue
            minutes_val = _coerce_float(raw_value)
            if minutes_val is None and isinstance(raw_value, Mapping):
                minutes_val = _coerce_float(raw_value.get("minutes"))
            if minutes_val is not None and minutes_val > 0:
                hours_text = fmt_hours(minutes_val / 60.0)
                _add_reason(f"planner buckets allocate {hours_text} to drilling")
            else:
                _add_reason("planner buckets include drilling")

    plan_info_mappings: list[Mapping[str, Any]] = []
    if isinstance(plan_info, Mapping) and plan_info:
        plan_info_mappings.extend(_iter_plan_mappings(plan_info))

    recognized_ops_from_plan = 0
    for mapping in plan_info_mappings:
        raw_recognized = mapping.get("recognized_line_items")
        recognized_val = _coerce_int(raw_recognized)
        if recognized_val is not None and recognized_val > 0:
            recognized_ops_from_plan = max(recognized_ops_from_plan, recognized_val)

        for item in _line_items_from(mapping):
            name = str(item.get("op") or item.get("name") or "").strip()
            if name and "drill" in name.lower():
                _add_reason(f"planner operations include {name}")

        for summary_key in ("plan_summary", "plan"):
            summary = mapping.get(summary_key)
            if isinstance(summary, Mapping):
                recognized_ops_from_plan = max(
                    recognized_ops_from_plan,
                    _count_recognized_ops(summary),
                )
                ops = summary.get("ops")
                if isinstance(ops, Sequence):
                    for entry in ops:
                        if isinstance(entry, Mapping):
                            label = (
                                entry.get("name")
                                or entry.get("op")
                                or entry.get("type")
                                or entry.get("bucket")
                            )
                        else:
                            label = entry
                        label_text = str(label or "").strip()
                        if label_text and "drill" in label_text.lower():
                            _add_reason(f"planner operations include {label_text}")

        for bucket_key in (
            "bucket_minutes_detail",
            "bucket_minutes_detail_for_render",
        ):
            _check_bucket_map(mapping.get(bucket_key) if isinstance(mapping, Mapping) else None)

        bucket_view = mapping.get("bucket_view") if isinstance(mapping, Mapping) else None
        if isinstance(bucket_view, Mapping):
            buckets = bucket_view.get("buckets")
            if isinstance(buckets, Mapping):
                _check_bucket_map(buckets)
            else:
                _check_bucket_map(bucket_view)

    if not plan_drilling_reasons and recognized_ops_from_plan > 0:
        # Recognized operations present but no explicit drilling signal; treat as informational only.
        plan_drilling_reasons = []

    render_state_extra: Mapping[str, Any] | None = None
    if isinstance(render_state, Mapping):
        extra_candidate = render_state.get("extra")
        if isinstance(extra_candidate, Mapping):
            render_state_extra = extra_candidate
    else:
        try:
            extra_candidate = getattr(render_state, "extra", None)
        except Exception:
            extra_candidate = None
        if isinstance(extra_candidate, Mapping):
            render_state_extra = extra_candidate

    def _extract_direct_drill_minutes(mapping: Mapping[str, Any]) -> tuple[float | None, bool]:
        minutes_found: float | None = None
        present = False
        for key in (
            "drill_total_minutes",
            "drill_total_minutes_with_toolchange",
            "total_minutes_with_toolchange",
            "total_minutes",
            "removal_drilling_minutes",
            "removal_drilling_minutes_subtotal",
        ):
            if key in mapping:
                present = True
                minutes_val = _coerce_float(mapping.get(key))
                if minutes_val is not None and minutes_val > 0:
                    minutes_found = max(minutes_found or 0.0, float(minutes_val))
        machine_val = _coerce_float(mapping.get("drill_machine_minutes"))
        labor_val = _coerce_float(mapping.get("drill_labor_minutes"))
        if machine_val is not None or labor_val is not None:
            present = True
            total = float(machine_val or 0.0) + float(labor_val or 0.0)
            if total > 0:
                minutes_found = max(minutes_found or 0.0, total)
        return minutes_found, present

    drill_card_minutes = 0.0
    drill_minutes_present = False

    def _maybe_update_drill_minutes(source: Mapping[str, Any] | None) -> None:
        nonlocal drill_card_minutes, drill_minutes_present
        if not isinstance(source, Mapping):
            return
        stack: list[Mapping[str, Any]] = [source]
        seen: set[int] = set()
        while stack:
            current = stack.pop()
            ident = id(current)
            if ident in seen:
                continue
            seen.add(ident)
            minutes, present = _extract_direct_drill_minutes(current)
            if present:
                drill_minutes_present = True
            if minutes is not None and minutes > drill_card_minutes:
                drill_card_minutes = minutes
            for next_key in ("extra", "drilling_meta"):
                candidate = current.get(next_key)
                if isinstance(candidate, Mapping):
                    stack.append(candidate)

    _maybe_update_drill_minutes(render_state_extra)
    if isinstance(render_state, Mapping):
        _maybe_update_drill_minutes(render_state)

    for mapping in plan_info_mappings:
        _maybe_update_drill_minutes(mapping)
        for key in (
            "bucket_state_extra",
            "planner_render_state",
            "bucket_render_state",
            "render_state",
        ):
            candidate = mapping.get(key) if isinstance(mapping, Mapping) else None
            _maybe_update_drill_minutes(candidate)

    if isinstance(breakdown, Mapping):
        for key in (
            "planner_render_state",
            "bucket_render_state",
            "removal_summary",
            "drilling_meta",
            "bucket_state_extra",
        ):
            _maybe_update_drill_minutes(breakdown.get(key))

    has_drilling = drill_card_minutes > 0.0

    should_note_drilling = hole_groups_flag or (removal_hr is not None and removal_hr > 0)
    if not should_note_drilling and plan_drilling_reasons:
        should_note_drilling = True
    if not should_note_drilling and has_drilling:
        should_note_drilling = True

    price_text = _format_money(totals.get("price") or breakdown.get("price"))
    qty_val = _coerce_int(totals.get("qty") or breakdown.get("qty") or breakdown.get("quantity"))
    if price_text:
        if qty_val and qty_val > 0:
            piece_label = "piece" if qty_val == 1 else "pieces"
            lines.append(f"Quote total {price_text} for {qty_val} {piece_label}.")
        else:
            lines.append(f"Quote total {price_text}.")

    material_selected = breakdown.get("material_selected") or {}
    material_canonical = ""
    if isinstance(material_selected, Mapping):
        material_canonical = str(material_selected.get("canonical") or "").strip()
    else:
        getter = getattr(material_selected, "get", None)
        if callable(getter):
            material_canonical = str(getter("canonical") or "").strip()
    if material_canonical:
        lines.append(f"Material: {material_canonical}.")

    labor_cost_rendered = _coerce_float(breakdown.get("labor_cost_rendered"))
    if labor_cost_rendered is None:
        labor_cost_rendered = _coerce_float(totals.get("labor_cost"))

    labor_text = _format_money(labor_cost_rendered)
    material_text = _format_money(
        breakdown.get("material_direct_cost")
        or totals.get("material_cost")
        or (breakdown.get("material") or {}).get("material_cost")
    )
    if labor_text or material_text:
        parts: list[str] = []
        if material_text:
            parts.append(f"material {material_text}")
        if labor_text:
            parts.append(f"labor & machine {labor_text}")
        lines.append("Cost makeup: " + "; ".join(parts) + ".")

    scrap_text = _format_pct(breakdown.get("scrap_pct"))
    if scrap_text and _coerce_float(breakdown.get("scrap_pct")):
        lines.append(f"Includes a {scrap_text} scrap allowance.")

    # === WHY: CONSISTENT DRILLING TEXT ===
    drill_detail_line: str | None = None
    if has_drilling and drill_card_minutes > 0.0:
        drill_detail_line = (
            f"Drilling time comes from removal-card math ({drill_card_minutes / 60.0:.2f} hr total)."
        )
    elif not should_note_drilling:
        drill_detail_line = "No drilling accounted."
    if drill_detail_line:
        if has_drilling:
            lowered_conflicts = ("no drilling", "does not involve drilling")
            lines[:] = [
                line
                for line in lines
                if not any(conflict in line.lower() for conflict in lowered_conflicts)
            ]
        lines.append(drill_detail_line)

    def _iter_named_values(values: Any) -> Iterable[tuple[str, float]]:
        items: Iterable[tuple[Any, Any]]
        if isinstance(values, Mapping):
            items = values.items()
        elif isinstance(values, Sequence):
            items = [tuple(item) for item in values if isinstance(item, (list, tuple)) and len(item) >= 2]
        else:
            return []
        for key, raw in items:
            amount = _coerce_float(raw)
            if amount is None or abs(amount) < 1e-2:
                continue
            name = str(key).strip()
            if not name:
                continue
            yield name, amount

    def _describe_top(entries: Iterable[tuple[str, float]], *, prefix: str) -> None:
        ranked = sorted(entries, key=lambda item: item[1], reverse=True)
        top = [
            f"{label}: {_format_money(value)}"
            for label, value in ranked[:3]
            if _format_money(value)
        ]
        if top:
            lines.append(f"{prefix} → " + "; ".join(top) + ".")

    process_entries = [
        (label, amount)
        for label, amount in _iter_named_values(breakdown.get("process_costs"))
        if not str(label).lower().startswith("planner_")
    ]
    ranked_process_entries = sorted(process_entries, key=lambda item: item[1], reverse=True)
    top_procs: list[str] = []
    for label, _ in ranked_process_entries[:3]:
        label_text = str(label).strip()
        if label_text:
            top_procs.append(label_text)

    render_state_extra: Mapping[str, Any] | None = None
    if isinstance(render_state, Mapping):
        extra_candidate = render_state.get("extra")
        if isinstance(extra_candidate, Mapping):
            render_state_extra = extra_candidate
    else:
        try:
            extra_candidate = getattr(render_state, "extra", None)
        except Exception:
            extra_candidate = None
        if isinstance(extra_candidate, Mapping):
            render_state_extra = extra_candidate

    drilling_hours = drill_card_minutes / 60.0 if drill_card_minutes > 0.0 else None
    has_card_minutes = drill_minutes_present
    has_drilling = bool(render_state_extra.get("drill_total_minutes")) if render_state_extra else False

    if has_drilling and "drilling" not in [p.lower() for p in top_procs]:
        top_procs.append("Drilling")

    if has_card_minutes:
        if drilling_hours is not None:
            hours_text = fmt_hours(drilling_hours)
            _add_reason(f"planner buckets allocate {hours_text} to drilling")
        else:
            _add_reason("planner buckets include drilling")

    if process_entries:
        _describe_top(process_entries, prefix="Largest process costs")

    if should_note_drilling:
        drill_line_present = any("drill" in line.lower() for line in lines)
        if not drill_line_present:
            reasons: list[str] = []
            if hole_groups_flag:
                reasons.append("geometry shows drilled holes")
            if removal_hr is not None and removal_hr > 0:
                hours_text = fmt_hours(removal_hr)
                reasons.append(f"removal card tracks {hours_text} of drilling")
            if plan_drilling_reasons:
                for reason in plan_drilling_reasons:
                    if reason not in reasons:
                        reasons.append(reason)
            reason_text = " and ".join(reasons) if reasons else "process data includes drilling"
            lines.append(f"Drilling is included because {reason_text}.")

    pass_through_entries = [
        (label, amount)
        for label, amount in _iter_named_values(breakdown.get("pass_through"))
        if str(label).strip().lower() != "material"
    ]
    if pass_through_entries:
        _describe_top(pass_through_entries, prefix="Pass-through items")

    if lines:
        return "\n".join(lines)
    return ""


# Helpers imported from appV5 -------------------------------------------------


def _parse_pct_like(x):
    try:
        v = float(x)
        return v/100.0 if v > 1.0 else v
    except Exception:
        return None


def _items_used_by_estimator(df):
    pats = _estimator_patterns()
    items = df["Item"].astype(str)
    used = []
    import re as _re
    for it in items:
        if any(p.search(it) for p in pats):
            if _re.search(r"\brate\b", it, _re.I):
                continue
            used.append(it)
    return used


def _estimator_patterns():
    pats = [
        r"\b(Qty|Lot Size|Quantity)\b",
        r"\b(Margin|Profit Margin)\b",
        r"\b(Expedite|Rush\s*Fee)\b",
        r"\b(Net\s*Volume|Volume_net|Volume\s*\(cm\^?3\))\b",
        r"\b(Density|Material\s*Density)\b", r"\b(Scrap\s*%|Expected\s*Scrap)\b",
        r"\b(Material\s*Price.*(per\s*g|/g)|Unit\s*Price\s*/\s*g)\b",
        r"\b(Supplier\s*Min\s*Charge|min\s*charge)\b",
        r"\b(Material\s*MOQ)\b",
        r"\b(Material\s*Cost|Raw\s*Material\s*Cost)\b",
        r"(Programming|CAM\s*Programming|2D\s*CAM|3D\s*CAM|Simulation|Verification|DFM\s*Review|Tool\s*Library|Setup\s*Sheets)",
        r"\b(CAM\s*Programming|CAM\s*Sim|Post\s*Processing)\b",
        r"(Fixture\s*Design|Process\s*Sheet|Traveler|Documentation|Complex\s*Assembly\s*Doc)",
        r"(Fixture\s*Build|Custom\s*Fixture\s*Build)",
        r"(Roughing\s*Cycle\s*Time|Adaptive|HSM)", r"(Semi[- ]?Finish|Rest\s*Milling)", r"(Finishing\s*Cycle\s*Time)",
        r"(Number\s*of\s*Milling\s*Setups|Milling\s*Setups)", r"(Setup\s*Time\s*per\s*Setup|Setup\s*Hours\s*/\s*Setup)",
        r"(Thin\s*Wall\s*Factor|Thin\s*Wall\s*Multiplier)", r"(Tolerance\s*Multiplier|Tight\s*Tolerance\s*Factor)",
        r"(Finish\s*Multiplier|Surface\s*Finish\s*Factor)",
        r"(OD\s*Turning|OD\s*Rough/Finish|Outer\s*Diameter)", r"(ID\s*Boring|Drilling|Reaming)",
        r"(Threading|Tapping|Single\s*Point)", r"(Cut[- ]?Off|Parting)",
        r"(WEDM\s*Hours|Wire\s*EDM\s*Hours|EDM\s*Burn\s*Time)",
        r"(EDM\s*Length_mm|WEDM\s*Length_mm|EDM\s*Perimeter_mm)",
        r"(EDM\s*Thickness_mm|Stock\s*Thickness_mm)", r"(EDM\s*Passes|WEDM\s*Passes)",
        r"(EDM\s*Cut\s*Rate_mm\^?2/min|WEDM\s*Cut\s*Rate)", r"(EDM\s*Edge\s*Factor)",
        r"(WEDM\s*Wire\s*Cost\s*/\s*m|Wire\s*Cost\s*/m)", r"(Wire\s*Usage\s*m\s*/\s*mm\^?2)",
        r"(Sinker\s*EDM\s*Hours|Ram\s*EDM\s*Hours|Burn\s*Time)", r"(Sinker\s*Burn\s*Volume_mm3|EDM\s*Volume_mm3)",
        r"(Sinker\s*MRR_mm\^?3/min|EDM\s*MRR)", r"(Electrode\s*Count)", r"(Electrode\s*(Cost|Material).*)",
        r"(Surface\s*Grind|Pre[- ]?Op\s*Grinding|Blank\s*Squaring)", r"(Jig\s*Grind)",
        r"(OD/ID\s*Grind|Cylindrical\s*Grind)", r"(Grind\s*Volume_mm3|Grinding\s*Volume)",
        r"(Grind\s*MRR_mm\^?3/min|Grinding\s*MRR)", r"(Grinding\s*Passes)",
        r"(Dress\s*Frequency\s*passes)", r"(Dress\s*Time\s*min(\s*/\s*pass)?)",
        r"(Grinding\s*Wheel\s*Cost)",
        r"(Lapping|Honing|Polishing)",
        r"(Deburr|Edge\s*Break)", r"(Tumbling|Vibratory)", r"(Bead\s*Blasting|Sanding)",
        r"(Laser\s*Mark|Engraving)", r"(Masking|Masking\s*for\s*Plating)",
        r"(In[- ]?Process\s*Inspection)", r"(Final\s*Inspection|Manual\s*Inspection)",
        r"(CMM\s*Programming)", r"(CMM\s*Run\s*Time\s*min)",
        r"(FAIR|ISIR|PPAP)", r"(Source\s*Inspection)",
        r"(Sawing|Waterjet|Blank\s*Prep)",
        r"(Assembly|Manual\s*Assembly|Precision\s*Fitting|Touch[- ]?up|Final\s*Fit)", r"(Hardware|BOM\s*Cost|Fasteners)",
        r"(Outsourced\s*Heat\s*Treat|Heat\s*Treat\s*Cost)", r"(Plating|Coating|Anodize|Black\s*Oxide|DLC|PVD|CVD)",
        r"(Passivation|Cleaning\s*Vendor)",
        r"(Packaging|Boxing|Crating\s*Labor)", r"(Custom\s*Crate\s*NRE)", r"(Packaging\s*Materials|Foam|Trays)",
        r"(Freight|Shipping\s*Cost)", r"(Insurance|Liability\s*Adder)",
        r"(EHS|Compliance|Training|Waste\s*Handling)", r"(Gauge|Check\s*Fixture\s*NRE)",
    ]
    return [re.compile(p, re.I) for p in pats]
