"""Local LLM integration helpers for the CAD Quoter UI."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


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
    ("add_pass_through", "Consumables /Hr"): (
        "Consumables /Hr Cost",
        float,
        float,
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
    "contingency_pct": (
        "ContingencyPct",
        float,
        float,
    ),
    "fixture_material_cost_delta": (
        "Fixture Material Cost",
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
            kwargs = dict(
                model_path=self._config.model_path,
                n_ctx=self._config.n_ctx,
                n_gpu_layers=self._config.n_gpu_layers,
                n_threads=self._config.n_threads,
                n_batch=self._config.n_batch,
                logits_all=False,
                verbose=False,
            )
            if self._config.rope_freq_scale is not None:
                kwargs["rope_freq_scale"] = self._config.rope_freq_scale
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
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
            path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
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
        js = parsed if isinstance(parsed, dict) else {}
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
    pmap = js.get("params", {}) if isinstance(js.get("params", {}), dict) else {}
    param_whys = pmap.get("_why", {}) if isinstance(pmap.get("_why", {}), dict) else {}
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


def run_llm_suggestions(client: LLMClient, payload: dict) -> tuple[dict, str, dict]:
    parsed, raw, usage = client.ask_json(
        system_prompt=SYSTEM_SUGGEST,
        user_prompt=json.dumps(payload, indent=2),
        temperature=0.3,
        max_tokens=512,
        context=payload,
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
{json.dumps(geo, indent=2)}

Rules of thumb:
- Small rectangular blocks with few features: Programming 0.2–1.0 hr, CAM 0.2–1.0 hr, Engineering 0 hr.
- Use setups 1–2 unless 3-axis accessibility is low (<0.6) or faces with unique normals > 4.
- Deburr 0.1–0.4 hr unless thin walls/freeform edges (then up to 0.8 hr).
- Final inspection ~0.2–0.6 hr; CMM only if tolerances < 0.02 mm or GD&T heavy.
- Grinding/EDM/Turning hours should be 0 unless features clearly require them.
- Never return huge numbers for tiny parts (<80 mm max dim).

Return JSON with this structure (numbers only, minutes only for CMM_RunTime_min):
{json.dumps(schema, indent=2)}
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
    programming = 0.3 + (faces / 40.0)
    programming = min(programming, 2.5)
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


# Helpers imported from appV5 -------------------------------------------------
import re


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
        r"\b(Overhead|Shop Overhead)\b", r"\b(Margin|Profit Margin)\b",
        r"\b(G&A|General\s*&\s*Admin)\b", r"\b(Contingency|Risk\s*Adder)\b",
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
        r"(Fixture\s*Build|Custom\s*Fixture\s*Build)", r"(Fixture\s*Material\s*Cost|Fixture\s*Hardware)",
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
