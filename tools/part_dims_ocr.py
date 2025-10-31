"""
part_dims_ocr.py
================
Standalone helper to extract a part's overall Length, Width, and Thickness from
a drawing using OCR, with an optional LLM arbitration step.

Features
--------
- Accepts PDF or image files.
- Local OCR via pytesseract (no network required).
- Heuristics & regexes for common formats:
  * "SIZE: 12.00 x 14.00 x 2.00"
  * "STOCK: 18.00 × 18.00 × 3.50 in"
  * "L=12.000  W=14.000  T=2.000"
  * "12 x 14 x 2 (INCH)" / "305 x 356 x 50 (MM)"
  * Title-block rows like "MATERIAL / SIZE"
- Units detection & conversion (in/mm).
- Confidence scoring + the raw snippets that led to the decision.
- Optional LLM arbitration (OpenAI Responses API) when multiple candidates conflict.

CLI
---
python tools/part_dims_ocr.py --input path/to/file.pdf --units auto --json-out dims.json
python tools/part_dims_ocr.py --input page.png --prefer-stock --use-llm

Env for LLM (optional):
  OPENAI_API_KEY=<your key>

Dependencies (install as needed):
  pip install pillow pytesseract pdf2image pydantic
  # For PDF rendering: poppler must be installed and on PATH for pdf2image.
"""

from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import math
import os
import pathlib
import re
from pathlib import Path
import shutil, subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Optional imports
try:
    from PIL import Image, ImageOps, ImageFilter
except Exception:
    Image = None  # type: ignore

try:
    import pytesseract
except Exception:
    pytesseract = None  # type: ignore

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None  # type: ignore

# ---------------------------------------------------------------------------

HOLE_TABLE_HDRS = ("HOLE TABLE", "LIST OF COORDINATES", "DESCRIPTION")


def _strip_tables(text: str) -> str:
    out = text
    for hdr in HOLE_TABLE_HDRS:
        pat = rf"(?:^|\n)[ \t]*{re.escape(hdr)}.*?(?=\n\s*\n|$\Z)"
        out = re.sub(pat, "", out, flags=re.IGNORECASE | re.DOTALL)
    return out


def _remove_fastener_numbers(text: str) -> str:
    t = re.sub(r"(?<=#)\d{1,2}-\d{2}", " ", text)
    t = re.sub(r"\b\d{1,2}-\d{2}\b", " ", t)
    t = re.sub(r"\b[1-9]/\d{1,2}\b", " ", t)
    return t


_num_pat = re.compile(r"(?<![#/\-])\b\d+(?:\.\d+)?\b")


def _extract_numeric_tokens(text: str):
    return [float(m.group(0)) for m in _num_pat.finditer(text)]


def _has_decimal(x: float) -> bool:
    s = f"{x}"
    return "." in s and len(s.split(".")[1]) >= 1


def _likely_thickness(x: float) -> bool:
    return 0.05 <= x <= 6.0


def _prefer_plate_thickness(vals):
    def t_key(v):
        frac = str(v).split(".")[1] if "." in str(v) else ""
        dec_score = -len(frac)
        half_score = -abs((v * 10000) % 50)
        return (dec_score, half_score, v)

    return max(vals, key=t_key)


def _plausible(L, W, T):
    if T is None or L is None or W is None:
        return False
    if T <= 0 or L <= 0 or W <= 0:
        return False
    if not _likely_thickness(T):
        return False
    if min(L, W) <= T * 3.0:
        return False
    return True


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _gate_vlm_dims(data: Dict[str, Any], clean_text: str, text_nums: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], str, str, Dict[str, Any]]:
    units_raw = data.get("units", "in")
    try:
        units = str(units_raw or "in").strip() or "in"
    except Exception:
        units = "in"

    json_L = _safe_float(data.get("length"))
    json_W = _safe_float(data.get("width"))
    json_T = _safe_float(data.get("thickness"))

    json_ok = _plausible(json_L, json_W, json_T)

    L2 = W2 = T2 = None
    if text_nums:
        t_candidates = [x for x in text_nums if _likely_thickness(x)]
        if t_candidates:
            T2 = _prefer_plate_thickness(t_candidates)

        floor = max(1.0, (T2 or 0) * 3.0)
        pool_dec = sorted([x for x in text_nums if x >= floor and _has_decimal(x)], reverse=True)
        pool_any = sorted([x for x in text_nums if x >= floor], reverse=True)
        pool = pool_dec or pool_any
        if len(pool) >= 2:
            L2, W2 = pool[0], pool[1]

    text_ok = _plausible(L2, W2, T2)

    def _value_in_text(val: Optional[float]) -> bool:
        if val is None:
            return False
        if any(abs(val - t) < 1e-6 for t in text_nums):
            return True
        return f"{val}" in clean_text

    json_in_text = all(_value_in_text(v) for v in (json_L, json_W, json_T))

    if (not json_ok or not json_in_text) and text_ok:
        L, W, T = L2, W2, T2
        source = "text_override"
    else:
        L, W, T = json_L, json_W, json_T
        source = "vlm_json"

    if L is not None and W is not None and L < W:
        L, W = W, L

    debug_block = {
        "json_candidate": {"L": json_L, "W": json_W, "T": json_T},
        "text_candidates_sample": sorted(list({float(x) for x in text_nums}))[:50],
        "decision_source": source,
    }

    return L, W, T, units, source, debug_block


def _file_fingerprint(path: str) -> str:
    p = pathlib.Path(path)
    try:
        st = p.stat()
    except FileNotFoundError:
        return hashlib.sha1(path.encode()).hexdigest()[:10]
    h = hashlib.sha1()
    h.update(path.encode("utf-8"))
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:10]


def _atomic_write_json(obj, out_path: str):
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------

# Local LLM (llama.cpp) support ----------------
_LOCAL_LLM = None
_LOCAL_LLM_PATH: Optional[Path] = None


def _load_qwen_local(model_path: str, mmproj_path: str, n_ctx: int = 4096):
    from llama_cpp import Llama  # pip install llama-cpp-python

    if not model_path or not mmproj_path:
        raise RuntimeError("Provide --vlm-local-model and --vlm-local-mmproj for vlm_local backend.")

    key = (str(Path(model_path).resolve()), str(Path(mmproj_path).resolve()), int(n_ctx))
    return Llama(model_path=key[0], mmproj=key[1], n_ctx=key[2], verbose=False)


def _img_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_NUMTOK = re.compile(r"\d+\s*\d+/\d+|\d+[.,]\d+|\.\d+|\d+")


def _parse_num(tok: str) -> float:
    tok = tok.replace(",", "").strip()
    m = re.match(r"^(\d+)\s+(\d+)/(\d+)$", tok)   # 1 3/8
    if m:
        a, b, c = m.groups()
        return float(a) + float(b) / float(c)
    if re.match(r"^\d+/\d+$", tok):               # 3/8
        a, b = tok.split("/")
        return float(a) / float(b)
    return float(tok)


def _extract_numbers_from_img(llm, pil_img) -> list[float]:
    """Ask the VLM to list only numeric dims, then parse to floats."""

    b64 = _img_b64(pil_img)
    resp = llm.create_chat_completion(messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": (
                "From this mechanical drawing, output ONLY the numeric dimensions you can read, "
                "separated by spaces (no words/units/angles)."
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }], temperature=0)
    txt = (resp["choices"][0]["message"]["content"] or "")
    return [_parse_num(t) for t in _NUMTOK.findall(txt)]


def _qwen_extract_block_dims_no_crops(llm, full_img) -> dict | None:
    """
    One-shot extraction: let Qwen identify views and return strict JSON.
    Falls back to numeric heuristics if the JSON response isn't plausible.
    """

    rules = (
        "You are reading a CAD drawing of a rectangular PLATE (two views: plan + side). "
        "Identify the OVERALL STOCK SIZE used to purchase raw material, not feature dimensions.\n"
        "Guidelines:\n"
        "• LENGTH & WIDTH: choose the two dimensions that span the OUTER edges of the plan view "
        "  (baseline arrows touching the outside rectangle). Ignore interior features (.03×45°, counterbores, taps) "
        "  and all tables/title blocks.\n"
        "• THICKNESS: pick the slab thickness from the SIDE view; it is usually a small value (e.g., 0.5005) "
        "  compared with length/width.\n"
        "• Do NOT infer from prior context. Use only this image.\n"
        "Return STRICT JSON ONLY (no prose): "
        '{"length": <number>, "width": <number>, "thickness": <number>, "units": "in"|"mm"}'
    )
    b64 = _img_b64(full_img)
    resp = llm.create_chat_completion(messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": rules},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }], temperature=0)
    txt = (resp["choices"][0]["message"]["content"] or "").strip()
    m = re.search(r"\{.*\}", txt, flags=re.S)
    data = None
    if m:
        try:
            data = json.loads(m.group(0))
        except Exception:
            data = None

    try:
        L = float(data.get("length", 0)) if data else 0
        W = float(data.get("width", 0)) if data else 0
        T = float(data.get("thickness", 0)) if data else 0
    except Exception:
        L = W = T = 0

    def plausible(L, W, T):
        # Dynamic sanity: thickness must be smallish, L/W must be > T by a margin
        if T <= 0 or max(L, W) <= 0:
            return False
        # Plates are typically thin vs plan dims
        if not (0.05 <= T <= 6.0):   # 0.050" .. 6.000"
            return False
        # L/W should clearly exceed T (≥3×), and not be microscopic
        if min(L, W) <= T * 3.0:
            return False
        return True

    # --- numeric evidence over the FULL image (no crops) ---
    if not plausible(L, W, T):
        nums = _extract_numbers_from_img(llm, full_img)

        # Filter out obvious feature dims and tiny notes
        tiny_bad = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.10, 0.12, 0.19, 0.25, 0.375}
        nums = [x for x in nums if x not in tiny_bad]

        # Step 1: choose thickness first (largest in a plausible thin range)
        t_candidates = [x for x in nums if 0.05 <= x <= 6.0]
        if t_candidates:
            # prefer "roundish" two-decimal values (2.00, 0.50, 0.5005, etc.)
            T = max(t_candidates, key=lambda v: (-abs(v - round(v, 3)), v))
        else:
            T = 0.5 if nums else 0.5  # weak fallback

        # Step 2: choose L/W as the two largest dimensions that are clearly > T
        # require at least ~3× thicker (blocks/plates) OR >= 1.0 in absolute
        lw_floor = max(T * 3.0, 1.0)   # clearly bigger than thickness
        lw_candidates = sorted([x for x in nums if x >= lw_floor], reverse=True)

        # If nothing qualifies (very small parts), just take top-2 numbers > T
        if len(lw_candidates) < 2:
            lw_candidates = sorted([x for x in nums if x > T], reverse=True)

        if len(lw_candidates) >= 2:
            L, W = lw_candidates[0], lw_candidates[1]

        # As a last resort, just pick top-2 from all numbers and the best T we had
        if (not L or not W) and len(nums) >= 2:
            L, W = sorted(nums, reverse=True)[:2]

        if L and W and T:
            data = {"length": float(L), "width": float(W), "thickness": float(T), "units": "in"}

    return data


def _emit_rendered_image(emit_image: str, full_img) -> None:
    if not emit_image:
        return
    try:
        path = emit_image if emit_image.lower().endswith(".png") else emit_image + ".png"
        full_img.save(path)
    except Exception:
        pass


def _emit_ocr_text(emit_ocr: str, text: str, input_path: str) -> None:
    if not emit_ocr or not text:
        return
    try:
        _ensure_parent_dir(emit_ocr)
        header = f"[INPUT] {input_path}\n[STAMP] {time.ctime()}\n\n"
        with open(emit_ocr, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(text)
        print(f"[dims-ocr] saved OCR/VLM text to {emit_ocr}")
    except Exception:
        pass


def _vlm_local_process_render(img_path: str,
                              vlm_local_model: str,
                              vlm_local_mmproj: str,
                              emit_image: str,
                              emit_ocr: str,
                              verbose: bool,
                              source_input: str) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    if Image is None:
        raise RuntimeError("Pillow is required for VLM local rendering.")

    llm = _load_qwen_local(vlm_local_model, vlm_local_mmproj)
    try:
        with Image.open(img_path) as img:
            img.load()
            full_img = img.convert("RGB")

        _emit_rendered_image(emit_image, full_img)

        full_text = _vlm_local_transcribe_image(llm, img_path)
        if verbose and full_text:
            preview = _norm(full_text)[:200]
            print("[dims-ocr] VLM transcription preview ->", preview)
        _emit_ocr_text(emit_ocr, full_text, source_input)

        raw_text = full_text or ""
        clean_text = _remove_fastener_numbers(_strip_tables(raw_text))
        text_nums = _extract_numeric_tokens(clean_text)

        data = _qwen_extract_block_dims_no_crops(llm, full_img)
        if data:
            L, W, T, units, source, debug_block = _gate_vlm_dims(data, clean_text, text_nums)

            if L is not None:
                data["length"] = float(L)
            if W is not None:
                data["width"] = float(W)
            if T is not None:
                data["thickness"] = float(T)
            data["units"] = units
            data["source"] = source
            data["debug"] = debug_block

            length_val = data.get("length")
            width_val = data.get("width")
            thickness_val = data.get("thickness")
            if length_val is not None and width_val is not None and thickness_val is not None:
                text_line = f"SIZE: {length_val} x {width_val} x {thickness_val} {units}"
            else:
                text_line = _json_dims_to_synthetic_line(data) or ""

            if verbose:
                print("[dims-ocr] VLM JSON ->", text_line, f"[{source}]")

            if emit_ocr:
                try:
                    json_blob = json.dumps(data, indent=2)
                except Exception:
                    json_blob = json.dumps(data, default=str)
                combined = (full_text or "").rstrip()
                if combined:
                    combined = f"{combined}\n\n[vlm_json]\n{json_blob}"
                else:
                    combined = json_blob
                _emit_ocr_text(emit_ocr, combined, source_input)

            return text_line, warnings

        warnings.append("VLM extraction failed; falling back to transcription.")
        if not full_text:
            full_text = _vlm_local_transcribe_image(llm, img_path)
        _emit_ocr_text(emit_ocr, full_text, source_input)
        return full_text, warnings
    finally:
        try:
            del llm
        except Exception:
            pass

def _detect_default_input() -> Optional[str]:
    """Best-effort to locate a sample input so CLI can run without flags."""

    env_path = os.getenv("PART_DIMS_OCR_INPUT")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return str(candidate)

    hardwired = Path("D:/CAD_Quoting_Tool/Cad Files/301_redacted.dwg")
    if hardwired.exists():
        return str(hardwired)

    return None

def _detect_default_oda_exe() -> Optional[str]:
    """Locate a suitable default ODA File Converter executable."""

    env_path = os.getenv("ODA_FILE_CONVERTER")
    if env_path:
        exe = Path(env_path).expanduser()
        if exe.is_file():
            return str(exe)

    candidates = [
        Path("C:/Program Files/ODA/OdaFileConverter.exe"),
        Path("C:/Program Files (x86)/ODA/OdaFileConverter.exe"),
    ]
    for exe in candidates:
        if exe.is_file():
            return str(exe)

    return None

def _detect_default_vlm_local_assets() -> tuple[Optional[str], Optional[str]]:
    """Locate default local VLM model + mmproj artefacts if present."""

    env_model = os.getenv("VLM_LOCAL_MODEL")
    env_mmproj = os.getenv("VLM_LOCAL_MMPROJ")

    def _normalize(candidate: Optional[str]) -> Optional[str]:
        if not candidate:
            return None
        path = Path(candidate).expanduser()
        try:
            return str(path.resolve())
        except Exception:
            return str(path)

    model_path = _normalize(env_model)
    mmproj_path = _normalize(env_mmproj)

    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models"

    def _first_existing(paths: Iterable[Path]) -> Optional[str]:
        for cand in paths:
            if cand.is_file():
                return str(cand.resolve())
        return None

    if not model_path and models_dir.is_dir():
        preferred_models = [
            models_dir / "qwen2.5-vl-7b-instruct-q4_k_m.gguf",
            models_dir / "Qwen2.5-VL-7B-Instruct-q4_k_m.gguf",
            models_dir / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        ]
        model_path = _first_existing(preferred_models)
        if not model_path:
            for cand in sorted(models_dir.glob("*.gguf")):
                name_lower = cand.name.lower()
                if "mmproj" in name_lower:
                    continue
                if "vl" in name_lower:
                    model_path = str(cand.resolve())
                    break

    if not mmproj_path and models_dir.is_dir():
        preferred_mmproj = [
            models_dir / "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf",
            models_dir / "Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf",
            models_dir / "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
            models_dir / "Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf",
        ]
        mmproj_path = _first_existing(preferred_mmproj)
        if not mmproj_path:
            for cand in sorted(models_dir.glob("*.gguf")):
                if "mmproj" in cand.name.lower():
                    mmproj_path = str(cand.resolve())
                    break

    return model_path, mmproj_path


def _iter_model_candidates() -> List[Path]:
    """Return possible GGUF model paths (env + ./models directory)."""

    candidates: List[Path] = []
    env_path = os.getenv("QWEN_GGUF_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            candidates.append(p)

    repo_models = Path(__file__).resolve().parent.parent / "models"
    if repo_models.is_dir():
        ggufs = sorted(repo_models.glob("*.gguf"))
        candidates.extend(ggufs)

    # Deduplicate while preserving order
    seen: set[Path] = set()
    uniq: List[Path] = []
    for cand in candidates:
        cand = cand.resolve()
        if cand not in seen:
            seen.add(cand)
            uniq.append(cand)
    return uniq


def _detect_default_debug_outputs() -> tuple[str, str, str]:
    """Return default paths for emitted debug artifacts (image, OCR text, JSON)."""

    base_dir = Path("D:/CAD_Quoting_Tool/debug")
    image_path = base_dir / "dwg_render.png"
    text_path = base_dir / "dwg_text.txt"
    json_path = base_dir / "dims.json"
    return str(image_path), str(text_path), str(json_path)


def _ensure_parent_dir(path: str) -> None:
    """Ensure parent directory exists for the given file path."""

    if not path:
        return
    try:
        Path(path).expanduser().resolve(strict=False).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        try:
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def _load_local_llm():
    """Load llama-cpp model from models/ folder if available; return Llama instance or None."""

    global _LOCAL_LLM, _LOCAL_LLM_PATH
    if _LOCAL_LLM is not None:
        return _LOCAL_LLM

    try:
        from llama_cpp import Llama  # type: ignore
    except Exception:
        return None

    for model_path in _iter_model_candidates():
        try:
            kwargs: Dict[str, Any] = {
                "model_path": str(model_path),
                "chat_format": "qwen",
                "n_ctx": int(os.getenv("QWEN_CONTEXT", "4096") or "4096"),
                "logits_all": False,
                "seed": 0,
            }
            n_threads = int(os.getenv("QWEN_N_THREADS", "0") or "0")
            if n_threads > 0:
                kwargs["n_threads"] = n_threads
            n_gpu_layers = int(os.getenv("QWEN_N_GPU_LAYERS", "0") or "0")
            if n_gpu_layers > 0:
                kwargs["n_gpu_layers"] = n_gpu_layers
            llm = Llama(**kwargs)
        except Exception as exc:
            print(f"[dims-ocr] failed to load local LLM {model_path}: {exc}", file=sys.stderr)
            continue

        _LOCAL_LLM = llm
        _LOCAL_LLM_PATH = model_path
        print(f"[dims-ocr] using local GGUF model: {model_path}")
        return _LOCAL_LLM

    return None


def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _vlm_local_transcribe_image(llm, image_path: str) -> str:
    """
    Ask the local VLM to transcribe all text from the image.
    Uses OpenAI-style chat with data:image/png;base64 payload (supported by llama.cpp).
    """

    b64 = _img_to_b64(image_path)
    prompt = "Transcribe ALL legible text exactly as printed in the drawing. Keep line breaks. No commentary."
    resp = llm.create_chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        temperature=0,
    )
    return (resp["choices"][0]["message"]["content"] or "").strip()


def _vlm_local_extract_dims(llm, image_path: str):
    """
    Ask the local VLM to return STRICT JSON:
    { "length": <num>, "width": <num>, "thickness": <num>, "units": "in|mm",
      "candidates": [ ... ] }
    """

    import json as _json
    import re as _re

    b64 = _img_to_b64(image_path)
    prompt = (
        "You are reading a mechanical drawing. Identify the OVERALL STOCK SIZE / SIZE / BLANK dimensions "
        "used for raw material selection (not feature dimensions). "
        "Return STRICT JSON with keys: length, width, thickness, units, candidates. "
        "Units must be 'in' or 'mm'. Numbers only. No extra text."
    )
    resp = llm.create_chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }],
        temperature=0,
    )
    txt = (resp["choices"][0]["message"]["content"] or "").strip()
    m = _re.search(r"\{.*\}", txt, flags=_re.S)
    if not m:
        return None
    try:
        return _json.loads(m.group(0))
    except Exception:
        return None


def _json_dims_to_synthetic_line(data: Dict[str, Any]) -> Optional[str]:
    try:
        units = str(data.get("units", "in") or "in").strip() or "in"
        length = float(data.get("length", 0) or 0)
        width = float(data.get("width", 0) or 0)
        thickness = float(data.get("thickness", 0) or 0)
    except (TypeError, ValueError):
        return None
    return f"SIZE: {length} x {width} x {thickness} {units}"


def _build_openai_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def _http_post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int = 60) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"VLM HTTP error {e.code}: {err or e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"VLM request failed: {e.reason}") from e
    return json.loads(body.decode("utf-8"))


def _vlm_remote_chat(endpoint: str, model: str, prompt: str, image_path: str) -> str:
    url = _build_openai_url(endpoint, "chat/completions")
    b64 = _img_to_b64(image_path)
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ],
        }],
        "temperature": 0,
        "max_tokens": 512,
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = _http_post_json(url, payload, headers)
    try:
        return (resp["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("Unexpected VLM response structure.")


def _vlm_remote_transcribe_image(endpoint: str, model: str, image_path: str) -> str:
    prompt = "Transcribe ALL legible text exactly as printed in the drawing. Keep line breaks. No commentary."
    return _vlm_remote_chat(endpoint, model, prompt, image_path)


def _vlm_remote_extract_dims(endpoint: str, model: str, image_path: str) -> Optional[Dict[str, Any]]:
    prompt = (
        "You are reading a mechanical drawing. Identify the OVERALL STOCK SIZE / SIZE / BLANK dimensions "
        "used for raw material selection (not feature dimensions). "
        "Return STRICT JSON with keys: length, width, thickness, units, candidates. "
        "Units must be 'in' or 'mm'. Numbers only. No extra text."
    )
    txt = _vlm_remote_chat(endpoint, model, prompt, image_path)
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
# ---------- Data model ----------

@dataclass
class DimensionTriple:
    length: float
    width: float
    thickness: float
    units: str  # "in" or "mm"
    source: str
    score: float  # 0..1

@dataclass
class ExtractionResult:
    best: Optional[DimensionTriple]
    candidates: List[DimensionTriple]
    ocr_text: str
    warnings: List[str]

# ---------- Utilities ----------

def _norm(s: str) -> str:
    """
    Normalize OCR quirks:
      - Replace multiplication glyphs (×, x, X) with 'x'
      - Normalize Ø-like or unicode characters (but we aren't using diameters here)
      - Collapse whitespace; strip weird punctuation pairs
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("×", "x").replace("X", "x")
    s = re.sub(r"[‒–—−]", "-", s)  # dashes
    s = re.sub(r"[“”]", '"', s).replace("’", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 12, 12.00, .62, 1 3/8
_NUM = r"(?:\d+(?:[.,]\d+)?|\.\d+|\d+\s*\d+/\d+)"

def _parse_num(n: str) -> float:
    n = n.strip().replace(",", "")
    # support mixed fractions: "1 3/8"
    m = re.match(r"^(\d+)\s+(\d+)/(\d+)$", n)
    if m:
        return float(m.group(1)) + (float(m.group(2)) / float(m.group(3)))
    # pure fraction "13/32"
    m = re.match(r"^(\d+)/(\d+)$", n)
    if m:
        return float(m.group(1)) / float(m.group(2))
    return float(n)

def _to_inches(val: float, units: str) -> float:
    return val / 25.4 if units == "mm" else val

def _to_mm(val: float, units: str) -> float:
    return val * 25.4 if units == "in" else val

def _score_reasonableness(L: float, W: float, T: float) -> float:
    """
    Heuristic confidence:
      - positive sizes
      - L>=W>=T typical
      - thickness relatively small vs planar dims
    """
    if min(L, W, T) <= 0:
        return 0.05
    score = 0.4
    if L >= W >= T:
        score += 0.3
    # Penalize absurd aspect ratios lightly
    ratio = max(L, W) / max(T, 1e-6)
    if 3 <= ratio <= 500:
        score += 0.2
    # Small bonus if T <= min(L, W)/5
    if T <= min(L, W) / 5:
        score += 0.1
    return min(score, 0.98)

# ---------- OCR ----------

def _render_pdf_to_images(path: str, dpi: int = 300) -> List["Image.Image"]:
    if convert_from_path is None:
        raise RuntimeError("pdf2image not installed or Poppler missing.")
    return convert_from_path(path, dpi=dpi)

def _preprocess_image(img: "Image.Image") -> "Image.Image":
    # Simple, fast pre-processing: grayscale + slight sharpen + adaptive thresholding
    g = ImageOps.grayscale(img)
    g = g.filter(ImageFilter.UnsharpMask(radius=1, percent=60, threshold=3))
    # Binarize with a light threshold
    return g.point(lambda p: 255 if p > 190 else 0)

def _ocr_image(img: "Image.Image") -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed. Install it or supply --use-llm with text.")
    cfg = "--psm 6"  # block of text
    txt = pytesseract.image_to_string(img, config=cfg)
    return _norm(txt)

# ---------- ODA File Converter (DWG/DXF -> PDF/TIFF/PNG) ----------

def _which_oda(oda_exe_cli: Optional[str] = None) -> Optional[str]:
    """
    Resolve path to ODA File Converter executable.
    Priority:
      1) --oda-exe CLI argument
      2) ODA_CONVERTER_EXE env var
      3) PATH (OdaFileConverter.exe / ODAFileConverter)
    """
    # CLI flag wins
    if oda_exe_cli and os.path.exists(oda_exe_cli):
        return oda_exe_cli

    # Env
    env_path = os.getenv("ODA_CONVERTER_EXE")
    if env_path and os.path.exists(env_path):
        return env_path

    # PATH common names
    for name in ("OdaFileConverter.exe", "ODAFileConverter.exe", "ODAFileConverter", "OdaFileConverter"):
        p = shutil.which(name)
        if p:
            return p

    return None


def _run_oda_to_dxf(oda_exe: str, in_path: str, out_dir: str,
                    out_ver: str = "ACAD2018",
                    recurse: int = 0,
                    audit: int = 0) -> str:
    if not os.path.exists(oda_exe):
        raise FileNotFoundError(f"ODA File Converter not found: {oda_exe}")

    tmp_in = os.path.join(out_dir, "_oda_in")
    os.makedirs(tmp_in, exist_ok=True)
    base = os.path.basename(in_path)
    src = os.path.join(tmp_in, base)
    shutil.copy2(in_path, src)

    tmp_out = os.path.join(out_dir, "_oda_out")
    os.makedirs(tmp_out, exist_ok=True)

    cmd = [
        oda_exe,
        tmp_in,
        tmp_out,
        out_ver,          # Output_version (e.g., ACAD2018)
        "DXF",            # Output File type (DWG|DXF|DXB) -- images NOT supported here
        str(int(bool(recurse))),
        str(int(bool(audit))),
        base,             # filter: render this file only
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    name_wo, _ = os.path.splitext(base)
    out_dxf = os.path.join(tmp_out, name_wo + ".dxf")
    if not os.path.exists(out_dxf):
        # Fallback: find any DXF with same base
        cands = [os.path.join(tmp_out, f) for f in os.listdir(tmp_out)
                 if f.lower().startswith(name_wo.lower()) and f.lower().endswith(".dxf")]
        if not cands:
            raise RuntimeError("ODA conversion produced no DXF.")
        out_dxf = cands[0]
    return out_dxf


# ---------- DXF -> PNG/TIFF rendering via ezdxf ----------

def _render_dxf_to_image(dxf_path: str, out_png: str, dpi: int = 800):
    """
    High-contrast DXF->PNG render:
      - white background
      - monochrome (all geometry black)
      - boosted lineweights so the outer profile shows
      - accurate lineweight policy
    """
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    from ezdxf.addons.drawing.config import Configuration, LinePolicy

    import ezdxf, matplotlib.pyplot as plt

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # High-contrast config (no layer colors)
    cfg = Configuration(
        background_color="#FFFFFF",
        monochrome=True,              # <-- all entities forced to black
        default_color="#000000",
        line_policy=LinePolicy.ACCURATE,
        min_lineweight=0.35,          # bump thin lines
        lineweight_scaling=1.8,       # scale CAD LWs up
        show_hatches=True,
        show_text=True,
    )

    ctx = RenderContext(doc)
    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    backend = MatplotlibBackend(ax, adjust_figure=True)
    Frontend(ctx, backend, config=cfg).draw_layout(msp, finalize=True)

    fig.savefig(out_png, dpi=dpi, facecolor="#FFFFFF")
    plt.close(fig)
    return out_png


def _perform_ocr_on_image_path(
    img_path: str,
    ocr_backend: str,
    vlm_mode: str,
    warnings: List[str],
    vlm_endpoint: str,
    vlm_model: str,
    vlm_local_model: str,
    vlm_local_mmproj: str,
) -> str:
    backend = (ocr_backend or "tesseract").lower()
    if backend == "vlm_local":
        llm = _load_qwen_local(vlm_local_model, vlm_local_mmproj)
        try:
            full_text = _vlm_local_transcribe_image(llm, img_path) or ""
            if vlm_mode == "extract":
                raw_text = full_text
                clean_text = _remove_fastener_numbers(_strip_tables(raw_text))
                text_nums = _extract_numeric_tokens(clean_text)
                data = _vlm_local_extract_dims(llm, img_path)
                if data:
                    L, W, T, units, source, debug_block = _gate_vlm_dims(data, clean_text, text_nums)
                    if L is not None:
                        data["length"] = float(L)
                    if W is not None:
                        data["width"] = float(W)
                    if T is not None:
                        data["thickness"] = float(T)
                    data["units"] = units
                    data["source"] = source
                    data["debug"] = debug_block

                    length_val = data.get("length")
                    width_val = data.get("width")
                    thickness_val = data.get("thickness")
                    if length_val is not None and width_val is not None and thickness_val is not None:
                        text_line = f"SIZE: {length_val} x {width_val} x {thickness_val} {units}"
                    else:
                        text_line = _json_dims_to_synthetic_line(data) or ""

                    try:
                        json_blob = json.dumps(data, indent=2)
                    except Exception:
                        json_blob = json.dumps(data, default=str)

                    sections: List[str] = []
                    if full_text.strip():
                        sections.append(full_text)
                    if text_line:
                        sections.append(text_line)
                    if json_blob:
                        sections.append("[vlm_json]")
                        sections.append(json_blob)
                    combined_text = "\n\n".join(section for section in sections if section)
                    return combined_text or text_line or full_text
                warnings.append("vlm_local extract failed; falling back to transcription.")
            return full_text
        finally:
            try:
                del llm
            except Exception:
                pass
    if backend == "vlm":
        if not vlm_endpoint:
            raise RuntimeError("Provide --vlm-endpoint for vlm backend.")
        if vlm_mode == "extract":
            full_text = _vlm_remote_transcribe_image(vlm_endpoint, vlm_model, img_path) or ""
            raw_text = full_text
            clean_text = _remove_fastener_numbers(_strip_tables(raw_text))
            text_nums = _extract_numeric_tokens(clean_text)
            data = _vlm_remote_extract_dims(vlm_endpoint, vlm_model, img_path)
            if data:
                L, W, T, units, source, debug_block = _gate_vlm_dims(data, clean_text, text_nums)
                if L is not None:
                    data["length"] = float(L)
                if W is not None:
                    data["width"] = float(W)
                if T is not None:
                    data["thickness"] = float(T)
                data["units"] = units
                data["source"] = source
                data["debug"] = debug_block

                length_val = data.get("length")
                width_val = data.get("width")
                thickness_val = data.get("thickness")
                if length_val is not None and width_val is not None and thickness_val is not None:
                    text_line = f"SIZE: {length_val} x {width_val} x {thickness_val} {units}"
                else:
                    text_line = _json_dims_to_synthetic_line(data) or ""

                try:
                    json_blob = json.dumps(data, indent=2)
                except Exception:
                    json_blob = json.dumps(data, default=str)

                sections: List[str] = []
                if full_text.strip():
                    sections.append(full_text)
                if text_line:
                    sections.append(text_line)
                if json_blob:
                    sections.append("[vlm_json]")
                    sections.append(json_blob)
                combined_text = "\n\n".join(section for section in sections if section)
                return combined_text or text_line or full_text
            warnings.append("vlm extract failed; falling back to transcription.")
            return full_text
        text = _vlm_remote_transcribe_image(vlm_endpoint, vlm_model, img_path)
        return text

    if Image is None or pytesseract is None:
        raise RuntimeError("Pillow + pytesseract required for OCR.")
    img = Image.open(img_path)
    img = _preprocess_image(img)
    return _ocr_image(img)


def _get_text_from_input(path: str,
                         oda_exe_cli: Optional[str] = None,
                         oda_format: str = "PDF",
                         ocr_backend: str = "tesseract",
                         vlm_endpoint: str = "http://127.0.0.1:8080/v1",
                         vlm_model: str = "qwen2.5-vl-7b",
                         vlm_mode: str = "transcribe",
                         vlm_local_model: str = "",
                         vlm_local_mmproj: str = "",
                         emit_image: str = "",
                         emit_ocr: str = "",
                         verbose: bool = False) -> Tuple[str, List[str]]:
    """Return OCR text + warnings for CAD/PDF/image inputs."""
    warnings: List[str] = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()

    # DWG: DWG -> DXF (ODA) -> PNG -> OCR
    if ext == ".dwg":
        oda_exe = _which_oda(oda_exe_cli)
        if not oda_exe:
            raise RuntimeError(
                "ODA File Converter not found.\n"
                "Provide --oda-exe path, or set ODA_CONVERTER_EXE env var, "
                "or add OdaFileConverter.exe to PATH."
            )
        with tempfile.TemporaryDirectory(prefix="oda_img_") as tmpdir:
            dxf_out = _run_oda_to_dxf(oda_exe, path, tmpdir, out_ver="ACAD2018")
            img_path = os.path.join(tmpdir, "page.png")
            _render_dxf_to_image(dxf_out, img_path)
            backend = (ocr_backend or "tesseract").lower()
            if backend == "vlm_local":
                text, local_warnings = _vlm_local_process_render(
                    img_path,
                    vlm_local_model,
                    vlm_local_mmproj,
                    emit_image,
                    emit_ocr,
                    verbose,
                    path,
                )
                warnings.extend(local_warnings)
                return (text, warnings)

            if emit_image and Image is not None:
                with Image.open(img_path) as img:
                    img.load()
                    _emit_rendered_image(emit_image, img)

            txt = _perform_ocr_on_image_path(
                img_path,
                ocr_backend,
                vlm_mode,
                warnings,
                vlm_endpoint,
                vlm_model,
                vlm_local_model,
                vlm_local_mmproj,
            )
            if emit_ocr:
                _emit_ocr_text(emit_ocr, txt, path)
            if verbose:
                print("[dims-ocr] OCR preview:", _norm(txt)[:500].replace("\n", " ⏎ "))
            return (txt, warnings)

    # DXF: DXF -> PNG -> OCR (same renderer)
    if ext == ".dxf":
        with tempfile.TemporaryDirectory(prefix="dxf_img_") as tmpdir:
            img_path = os.path.join(tmpdir, "page.png")
            _render_dxf_to_image(path, img_path)
            backend = (ocr_backend or "tesseract").lower()
            if backend == "vlm_local":
                text, local_warnings = _vlm_local_process_render(
                    img_path,
                    vlm_local_model,
                    vlm_local_mmproj,
                    emit_image,
                    emit_ocr,
                    verbose,
                    path,
                )
                warnings.extend(local_warnings)
                return (text, warnings)

            if emit_image and Image is not None:
                with Image.open(img_path) as img:
                    img.load()
                    _emit_rendered_image(emit_image, img)

            txt = _perform_ocr_on_image_path(
                img_path,
                ocr_backend,
                vlm_mode,
                warnings,
                vlm_endpoint,
                vlm_model,
                vlm_local_model,
                vlm_local_mmproj,
            )
            if emit_ocr:
                _emit_ocr_text(emit_ocr, txt, path)
            if verbose:
                print("[dims-ocr] OCR preview:", _norm(txt)[:500].replace("\n", " ⏎ "))
            return (txt, warnings)

    # Native PDF
    if ext == ".pdf":
        if convert_from_path is None:
            raise RuntimeError("pdf2image not available to render PDFs.")
        pages = _render_pdf_to_images(path)
        if not pages:
            warnings.append("No pages rendered from PDF.")
        texts: List[str] = []
        backend = (ocr_backend or "tesseract").lower()
        if backend == "tesseract":
            if Image is None or pytesseract is None:
                raise RuntimeError("Pillow + pytesseract required for OCR.")
            for p in pages:
                img = _preprocess_image(p) if Image else p
                txt = _ocr_image(img)
                texts.append(txt)
        else:
            with tempfile.TemporaryDirectory(prefix="pdf_img_") as tmpdir:
                for idx, p in enumerate(pages):
                    img_path = os.path.join(tmpdir, f"page_{idx}.png")
                    p.save(img_path, format="PNG")
                    txt = _perform_ocr_on_image_path(
                        img_path,
                        ocr_backend,
                        vlm_mode,
                        warnings,
                        vlm_endpoint,
                        vlm_model,
                        vlm_local_model,
                        vlm_local_mmproj,
                    )
                    texts.append(txt)
        combined = "\n".join(texts)
        if emit_ocr:
            _emit_ocr_text(emit_ocr, combined, path)
        if verbose and combined:
            print("[dims-ocr] OCR preview:", _norm(combined)[:500].replace("\n", " ⏎ "))
        return (combined, warnings)

    # Raster images
    backend = (ocr_backend or "tesseract").lower()
    if backend == "tesseract":
        if Image is None:
            raise RuntimeError("Pillow not installed to load images.")
        img = Image.open(path)
        img = _preprocess_image(img)
        txt = _ocr_image(img)
    else:
        txt = _perform_ocr_on_image_path(
            path,
            ocr_backend,
            vlm_mode,
            warnings,
            vlm_endpoint,
            vlm_model,
            vlm_local_model,
            vlm_local_mmproj,
        )
    if emit_ocr and txt:
        _emit_ocr_text(emit_ocr, txt, path)
    if verbose and txt:
        print("[dims-ocr] OCR preview:", _norm(txt)[:500].replace("\n", " ⏎ "))
    return (txt, warnings)

# ---------- Regex candidates ----------

# allow up to ~40 non-digit chars between captured numbers (labels, line breaks)
_GAP = r"(?:[^\d]{0,40})"
_PATTERNS = [
    # SIZE: 12.00 x 14.00 x 2.00 in  (robust gaps)
    re.compile(fr"\bSIZE\s*[:\-]?{_GAP}({_NUM}){_GAP}[x×X]{_GAP}({_NUM}){_GAP}[x×X]{_GAP}({_NUM}){_GAP}(in|inch|inches|mm|millimeters)?\b", re.I|re.S),
    # STOCK: 18 x 18 x 3.5 IN
    re.compile(fr"\bSTOCK\s*[:\-]?{_GAP}({_NUM}){_GAP}[x×X]{_GAP}({_NUM}){_GAP}[x×X]{_GAP}({_NUM}){_GAP}(in|inch|inches|mm|millimeters)?\b", re.I|re.S),
    # L=12  W=14  T=2  (units optional)
    re.compile(fr"\bL\s*[:=]\s*({_NUM}).*?\bW\s*[:=]\s*({_NUM}).*?\bT\s*[:=]\s*({_NUM})(?:\s*(in|inch|inches|mm|millimeters))?", re.I|re.S),
    # Plain triplet possibly followed by (INCH) / (MM)
    re.compile(fr"\b({_NUM}){_GAP}[x×X]{_GAP}({_NUM}){_GAP}[x×X]{_GAP}({_NUM}){_GAP}(?:\((in|inch|inches|mm|millimeters)\))?\b", re.I|re.S),
    # Title-block "MATERIAL / SIZE ... 12 x 14 x 2"
    re.compile(fr"\bSIZE\b.*?({_NUM}){_GAP}[x×X]{_GAP}({_NUM}){_GAP}[x×X]{_GAP}({_NUM})(?:{_GAP}(in|inch|inches|mm|millimeters))?", re.I|re.S),
]

_UNIT_CANON = {
    None: "auto",
    "in": "in", "inch": "in", "inches": "in",
    "mm": "mm", "millimeters": "mm",
}

def _find_candidates(text: str, prefer_stock: bool) -> List[DimensionTriple]:
    textN = _norm(text)
    cands: List[DimensionTriple] = []
    for pat in _PATTERNS:
        for m in pat.finditer(textN):
            L = _parse_num(m.group(1))
            W = _parse_num(m.group(2))
            T = _parse_num(m.group(3))
            units_raw = _UNIT_CANON.get((m.group(4) or "").lower(), "auto")
            # Guess units if omitted: bias inches if decimals with 2 places or typical US title blocks
            units = "in" if units_raw in ("auto", "in") else "mm"
            src = m.group(0)
            score = _score_reasonableness(L, W, T)
            # STOCK bias if requested
            if prefer_stock and src.upper().startswith("STOCK"):
                score += 0.05
            cands.append(DimensionTriple(L, W, T, units, src, min(score, 0.99)))
    # Deduplicate by near-equality
    uniq: List[DimensionTriple] = []
    def _near(a: float, b: float) -> bool: return abs(a-b) <= 1e-3
    for c in cands:
        if any(_near(c.length,u.length) and _near(c.width,u.width) and _near(c.thickness,u.thickness) and c.units==u.units for u in uniq):
            continue
        uniq.append(c)
    return uniq

# ---------- LLM arbitration (optional) ----------

def _llm_pick_best(cands: List[DimensionTriple], ocr_text: str) -> Optional[int]:
    """
    Optional: ask an LLM (local or cloud) to choose the best triple.
    Prefers local llama.cpp GGUF models in ./models; falls back to OpenAI if configured.
    """

    # Prepare prompt payload once
    options = [
        {
            "index": idx,
            "length": c.length,
            "width": c.width,
            "thickness": c.thickness,
            "units": c.units,
            "source": c.source[:200],
        }
        for idx, c in enumerate(cands)
    ]

    base_instruction = (
        "You are selecting overall part dimensions from OCR text.\n"
        "Pick the most likely Length, Width, Thickness triple (in the drawing units) used for stock selection.\n"
        "Prefer values explicitly labeled as SIZE or STOCK; avoid feature dimensions.\n"
        "Return ONLY the integer index of the best option."
    )
    user_payload = (
        f"Options: {json.dumps(options)}\n\n"
        f"OCR Text (truncated): {ocr_text[:3000]}"
    )

    # --- Try local llama.cpp model first ---
    llm = _load_local_llm()
    if llm is not None:
        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": base_instruction},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.0,
                max_tokens=32,
            )
            content = response["choices"][0]["message"]["content"]  # type: ignore[index]
            m = re.search(r"(\d+)", content or "")
            if m:
                return int(m.group(1))
        except Exception as exc:
            print(f"[dims-ocr] local LLM arbitration failed: {exc}", file=sys.stderr)

    # --- Fallback to OpenAI (if API key present) ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            input=f"{base_instruction}\n\n{user_payload}",
        )
        content = resp.output[0].content[0].text.strip()  # type: ignore[index]
    except Exception:
        try:
            import openai  # type: ignore

            openai.api_key = api_key
            content = openai.ChatCompletion.create(  # type: ignore[attr-defined]
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": base_instruction},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0,
            )["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            print(f"[dims-ocr] OpenAI arbitration failed: {exc}", file=sys.stderr)
            return None

    m = re.search(r"(\d+)", content or "")
    return int(m.group(1)) if m else None

# ---------- Public API ----------

def extract_part_dims(input_path: str,
                      units: str = "auto",
                      prefer_stock: bool = False,
                      use_llm: bool = False,
                      oda_exe: Optional[str] = None,
                      oda_format: str = "PDF",
                      ocr_backend: str = "tesseract",
                      vlm_endpoint: str = "http://127.0.0.1:8080/v1",
                      vlm_model: str = "qwen2.5-vl-7b",
                      vlm_mode: str = "transcribe",
                      vlm_local_model: str = "",
                      vlm_local_mmproj: str = "",
                      emit_image: str = "",
                      emit_ocr: str = "",
                      verbose: bool = False) -> ExtractionResult:
    """
    Extract (L, W, T) from a PDF/image drawing.

    Args:
        input_path: path to the source PDF/image.
        units: "in", "mm", or "auto" (auto = detect; inches-biased).
        prefer_stock: small score bump to lines starting with "STOCK".
        use_llm: if True, and multiple candidates exist, ask LLM to pick best.
        oda_exe: optional explicit path to ODA File Converter for DWG/DXF inputs.
        oda_format: ODA output format prior to OCR (PDF/TIFF/PNG).
        ocr_backend: "tesseract", "vlm", or "vlm_local".
        vlm_endpoint: HTTP endpoint for OpenAI-compatible VLM server.
        vlm_model: Model name for remote VLM.
        vlm_mode: "transcribe" for raw text or "extract" for JSON dimension parsing.
        vlm_local_model: Path to local GGUF vision model.
        vlm_local_mmproj: Path to local mmproj GGUF for the model.
        emit_image: Optional path/prefix to save the rendered CAD image for debugging.
        emit_ocr: Optional path to save OCR/VLM text output for debugging.
        verbose: If True, print a short preview of OCR text.

    Returns:
        ExtractionResult with best triple (if any), all candidates, raw OCR text, warnings.
    """
    warnings: List[str] = []
    text, w = _get_text_from_input(
        input_path,
        oda_exe_cli=oda_exe,
        oda_format=oda_format,
        ocr_backend=ocr_backend,
        vlm_endpoint=vlm_endpoint,
        vlm_model=vlm_model,
        vlm_mode=vlm_mode,
        vlm_local_model=vlm_local_model,
        vlm_local_mmproj=vlm_local_mmproj,
        emit_image=emit_image,
        emit_ocr=emit_ocr,
        verbose=verbose,
    )
    warnings += w
    if not text.strip():
        return ExtractionResult(best=None, candidates=[], ocr_text=text, warnings=warnings+["Empty OCR text."])

    cands = _find_candidates(text, prefer_stock)

    # Post-filter by requested units
    if units in ("in","mm"):
        cands = [c for c in cands if c.units == units] or cands

    if not cands:
        return ExtractionResult(best=None, candidates=[], ocr_text=text, warnings=warnings+["No dimension triples found."])

    # If multiple, try LLM if allowed; else pick highest score
    best_idx: Optional[int] = None
    if use_llm and len(cands) > 1:
        best_idx = _llm_pick_best(cands, text)
    if best_idx is None or best_idx < 0 or best_idx >= len(cands):
        # fall back to score
        best_idx = max(range(len(cands)), key=lambda i: cands[i].score)

    best = cands[best_idx]
    # Normalize ordering (L>=W); swap if needed
    if best.width > best.length:
        best = DimensionTriple(best.width, best.length, best.thickness, best.units, best.source, best.score)

    return ExtractionResult(best=best, candidates=cands, ocr_text=text, warnings=warnings)

# ---------- CLI ----------

def _cli():
    hardwired_input = str(Path("D:/CAD_Quoting_Tool/Cad Files/301_redacted.dwg"))
    default_input = _detect_default_input() or hardwired_input

    hardwired_oda_exe = str(Path("C:/Program Files/ODA/OdaFileConverter.exe"))
    default_oda_exe = _detect_default_oda_exe() or hardwired_oda_exe

    detected_model_path, detected_mmproj_path = _detect_default_vlm_local_assets()
    default_model_path = detected_model_path or str(
        Path("D:/CAD_Quoting_Tool/models/qwen2.5-vl-7b-instruct-q4_k_m.gguf")
    )
    default_mmproj_path = detected_mmproj_path or str(
        Path("D:/CAD_Quoting_Tool/models/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf")
    )

    default_emit_image, default_emit_ocr, default_json_out = _detect_default_debug_outputs()
    default_ocr_backend = "vlm_local"

    p = argparse.ArgumentParser(
        description="Extract part L/W/T from drawing via OCR (with optional LLM arbitration)."
    )

    input_help = "Path to PDF or image."
    if default_input:
        input_help += f" (default: {default_input})"
    else:
        input_help += " (default: first PDF/image found in 'Cad Files' or 'debug')"
    json_out_help = (
        "Optional: override the dims JSON output path. "
        f"Default: debug/<input>_<fingerprint>_dims.json (e.g., {default_json_out})."
    )
    if default_oda_exe:
        oda_help = (
            f"Path to ODA File Converter executable (default: {default_oda_exe}). "
            "Alternatively set ODA_CONVERTER_EXE env var or put it on PATH."
        )
    else:
        oda_help = "Path to ODA File Converter executable. Alternatively set ODA_CONVERTER_EXE env var or put it on PATH."
    ocr_backend_help = (
        "OCR backend: tesseract (pytesseract), vlm (remote OpenAI-compatible endpoint), "
        "or vlm_local (local llama.cpp). Defaults to vlm_local so you can just run the script."
    )
    emit_image_help = (
        "If set, override the rendered PNG output path. "
        f"Default: debug/<input>_<fingerprint>_render.png (e.g., {default_emit_image})."
    )
    emit_ocr_help = (
        "If set, override the OCR/VLM transcript path. "
        f"Default: debug/<input>_<fingerprint>_text.txt (e.g., {default_emit_ocr})."
    )
    verbose_help = "Print a short preview of OCR text (default: enabled; use --no-verbose to disable)."

    p.add_argument("--input", default=default_input, help=input_help)
    p.add_argument("--units", choices=["auto", "in", "mm"], default="auto", help="Units preference (default auto).")
    p.add_argument("--prefer-stock", action="store_true", help="Bias lines starting with STOCK.")
    p.add_argument("--use-llm", action="store_true", help="If multiple candidates, ask LLM to pick best (requires OPENAI_API_KEY/local model).")
    p.add_argument("--json-out", default=None, help=json_out_help)
    p.add_argument("--oda-exe", default=default_oda_exe or "", help=oda_help)
    p.add_argument("--oda-format", choices=["PDF","TIFF","PNG"], default="PDF",
                   help="ODA output format before OCR (default PDF).")
    p.add_argument("--ocr-backend", choices=["tesseract", "vlm", "vlm_local"], default=default_ocr_backend, help=ocr_backend_help)
    p.add_argument("--vlm-endpoint", default="http://127.0.0.1:8080/v1", help="Only for --ocr-backend vlm.")
    p.add_argument("--vlm-model", default="qwen2.5-vl-7b", help="Only for --ocr-backend vlm.")
    p.add_argument("--vlm-local-model", default=default_model_path or "", help="Path to qwen2.5-vl-*.gguf")
    p.add_argument("--vlm-local-mmproj", default=default_mmproj_path or "", help="Path to mmproj-*.gguf")
    p.add_argument("--vlm-mode", choices=["transcribe", "extract"], default="transcribe",
                   help="Return raw text (transcribe) or JSON dims (extract). Works for vlm and vlm_local.")
    p.add_argument("--emit-image", default=None, help=emit_image_help)
    p.add_argument("--emit-ocr", default=None, help=emit_ocr_help)
    p.add_argument("--no-cache", action="store_true", help="Ignore any previous outputs; overwrite.")
    p.add_argument("--out-dir", default=None, help="Output folder (default: debug).")
    p.add_argument("--verbose", action="store_true", dest="verbose", help=verbose_help)
    p.add_argument("--no-verbose", action="store_false", dest="verbose", help="Disable OCR preview output.")
    p.set_defaults(verbose=True)
    args = p.parse_args()

    input_path = args.input or default_input
    if not input_path:
        p.error("No input file provided; pass --input or set PART_DIMS_OCR_INPUT.")

    input_path = str(Path(input_path).expanduser())
    if default_input:
        default_norm = str(Path(default_input).expanduser().resolve(strict=False))
        input_norm = str(Path(input_path).expanduser().resolve(strict=False))
        if os.path.normcase(default_norm) == os.path.normcase(input_norm):
            print(f"[dims-ocr] defaulting to input: {input_path}")

    debug_dir = args.out_dir or r"D:\CAD_Quoting_Tool\debug"
    os.makedirs(debug_dir, exist_ok=True)

    fp = _file_fingerprint(input_path)
    base = pathlib.Path(input_path).stem

    img_path = args.emit_image or os.path.join(debug_dir, f"{base}_{fp}_render.png")
    txt_path = args.emit_ocr or os.path.join(debug_dir, f"{base}_{fp}_text.txt")
    json_path = args.json_out or os.path.join(debug_dir, f"{base}_{fp}_dims.json")

    for candidate in (img_path, txt_path, json_path):
        _ensure_parent_dir(candidate)

    if args.no_cache:
        for pth in (img_path, txt_path, json_path):
            try:
                os.remove(pth)
            except FileNotFoundError:
                pass

    try:
        res = extract_part_dims(
            input_path,
            units=args.units,
            prefer_stock=args.prefer_stock,
            use_llm=args.use_llm,
            oda_exe=(args.oda_exe or None),
            oda_format=args.oda_format,
            ocr_backend=args.ocr_backend,
            vlm_endpoint=args.vlm_endpoint,
            vlm_model=args.vlm_model,
            vlm_mode=args.vlm_mode,
            vlm_local_model=args.vlm_local_model,
            vlm_local_mmproj=args.vlm_local_mmproj,
            emit_image=img_path,
            emit_ocr=txt_path,
            verbose=args.verbose,
        )
        out: Dict[str, Any] = {
            "best": None if res.best is None else {
                "length": res.best.length,
                "width": res.best.width,
                "thickness": res.best.thickness,
                "units": res.best.units,
                "confidence": res.best.score,
                "source": res.best.source,
            },
            "candidates": [
                {
                    "length": c.length, "width": c.width, "thickness": c.thickness,
                    "units": c.units, "confidence": c.score, "source": c.source
                } for c in res.candidates
            ],
            "warnings": res.warnings,
        }
        if res.best is None:
            raise RuntimeError("Failed to extract dimensions; refusing to reuse old dims.json.")
        best = res.best
        if not (best.length and best.width and best.thickness):
            raise RuntimeError("Failed to extract dimensions; refusing to reuse old dims.json.")

        dims = {
            "length": best.length,
            "width": best.width,
            "thickness": best.thickness,
            "units": best.units,
        }
        _atomic_write_json(dims, json_path)
        print(f"[dims] wrote {json_path}")
        print(json.dumps(out, indent=2))
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    _cli()
