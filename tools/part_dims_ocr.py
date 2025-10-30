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
import io
import json
import math
import os
import re
from pathlib import Path
import shutil, subprocess
import sys
import tempfile
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
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

# Local LLM (llama.cpp) support ----------------
_LOCAL_LLM = None
_LOCAL_LLM_PATH: Optional[Path] = None

_VLM_LOCAL_INSTANCE = None
_VLM_LOCAL_KEY: Optional[Tuple[str, str, int]] = None


def _load_qwen_local(model_path: str, mmproj_path: str, n_ctx: int = 4096):
    from llama_cpp import Llama  # pip install llama-cpp-python

    global _VLM_LOCAL_INSTANCE, _VLM_LOCAL_KEY

    if not model_path or not mmproj_path:
        raise RuntimeError("Provide --vlm-local-model and --vlm-local-mmproj for vlm_local backend.")

    key = (str(Path(model_path).resolve()), str(Path(mmproj_path).resolve()), int(n_ctx))
    if _VLM_LOCAL_INSTANCE is not None and _VLM_LOCAL_KEY == key:
        return _VLM_LOCAL_INSTANCE

    llm = Llama(model_path=key[0], mmproj=key[1], n_ctx=key[2], verbose=False)
    _VLM_LOCAL_INSTANCE = llm
    _VLM_LOCAL_KEY = key
    return llm


def _crop_frac(img, box_str: str):
    # box_str like "0,0,0.72,1.0"
    l, t, r, b = [float(x) for x in box_str.split(",")]
    W, H = img.size
    return img.crop((int(l * W), int(t * H), int(r * W), int(b * H)))


def _img_b64(img):
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
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
    """Ask the VLM to list numeric dimensions only, then parse to floats."""

    def _img_b64(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    b64 = _img_b64(pil_img)
    resp = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "From this CAD view, output ONLY numeric dimensions you can read, separated by spaces. "
                            "No words, no units, no angles, no parentheses."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0,
    )
    txt = (resp["choices"][0]["message"]["content"] or "")
    return [_parse_num(t) for t in _NUMTOK.findall(txt)]


def _pick_dims(plan_nums: list[float], side_nums: list[float]) -> tuple[float, float, float] | None:
    P = sorted([x for x in plan_nums if x >= 10.0], reverse=True)
    if len(P) < 2:
        P = sorted(plan_nums, reverse=True)[:2]
    if len(P) < 2:
        return None
    L, W = P[0], P[1]

    Tcand = [x for x in side_nums if 0.25 <= x <= 3.00]
    T = max(Tcand) if Tcand else (sorted(side_nums)[0] if side_nums else None)
    if T is None:
        return None
    return (L, W, T)


def _qwen_extract_block_dims_local(llm, full_img, plan_img, side_img) -> dict | None:
    """
    Ask Qwen2.5-VL (local) to output strict JSON for overall block dimensions.
    Uses explicit rules so it selects outer baseline dimensions on plan view,
    and overall slab thickness on side view.
    """

    def _img_b64(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    b_full = _img_b64(full_img)
    b_plan = _img_b64(plan_img)
    b_side = _img_b64(side_img)

    system = (
        "You analyze mechanical drawings. Return overall stock size only."
    )
    rules = (
        "Identify the OVERALL block dimensions of a rectangular plate:\n"
        "• Use the PLAN/TOP view to pick LENGTH and WIDTH = the two LARGEST linear dimensions that span the outer edges. "
        "  Prefer baseline dimensions with extension lines/arrows touching the outer rectangle. Ignore interior callouts.\n"
        "• Use the SIDE view to pick THICKNESS = the overall plate thickness. Ignore counterbores, chamfers (.03 x 45°), taps, and small offsets.\n"
        "• Prefer whole-inch or 2-decimal values when close candidates exist (e.g., 15.50 over 15.49 if both appear).\n"
        "• Units are almost always inches if numbers look like 12.00, 15.50, 2.00; otherwise detect mm if clearly labeled.\n"
        "• Return STRICT JSON ONLY:\n"
        '{"length": <number>, "width": <number>, "thickness": <number>, "units": "in"|"mm"}\n'
        "No commentary."
    )

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": rules},
                {"type": "text", "text": "Full sheet (context only):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b_full}"}},
                {"type": "text", "text": "Plan/top view crop (choose LENGTH & WIDTH from this view):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b_plan}"}},
                {"type": "text", "text": "Side view crop (choose THICKNESS from this view):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b_side}"}},
            ],
        },
    ]

    resp = llm.create_chat_completion(messages=messages, temperature=0)
    txt = (resp["choices"][0]["message"]["content"] or "").strip()
    m = re.search(r"\{.*\}", txt, flags=re.S)
    data = None
    if m:
        try:
            data = json.loads(m.group(0))
        except Exception:
            data = None

    # Post-check: if JSON missing or obviously wrong, fall back to numeric evidence.
    # Plan: take two largest >= 10.0 (sheet shows things like 12.00, 13.50, 15.50).
    # Side: take the largest in [0.25, 3.50] as thickness (common plate range).
    if not data or any(k not in data for k in ("length", "width", "thickness")):
        data = {}

    try:
        L, W, T = float(data.get("length", 0)), float(data.get("width", 0)), float(data.get("thickness", 0))
    except Exception:
        L, W, T = 0.0, 0.0, 0.0

    def _looks_plausible(L, W, T):
        return (max(L, W) >= 12.0 and min(L, W) >= 8.0 and 0.25 <= T <= 3.5)

    if not _looks_plausible(L, W, T):
        plan_nums = _extract_numbers_from_img(llm, plan_img)
        side_nums = _extract_numbers_from_img(llm, side_img)

        big = sorted([x for x in plan_nums if x >= 10.0], reverse=True)
        if len(big) < 2:
            big = sorted(plan_nums, reverse=True)[:2]
        if len(big) >= 2:
            L, W = big[0], big[1]

        t_candidates = [x for x in side_nums if 0.25 <= x <= 3.50]
        if t_candidates:
            T = max(t_candidates)
        elif side_nums:
            T = sorted(side_nums)[0]

        if L and W and T:
            data = {"length": float(L), "width": float(W), "thickness": float(T), "units": "in"}

    # final sanity
    if not data or any(k not in data for k in ("length", "width", "thickness", "units")):
        return None
    return data


def _vlm_local_transcribe_all(llm, image):
    b64 = _img_b64(image)
    resp = llm.create_chat_completion(messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": (
                "Transcribe all visible numeric dimensions from this drawing (just the numbers with decimals or fractions). "
                "List them separated by spaces. No words, no comments."
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }], temperature=0)
    return (resp["choices"][0]["message"]["content"] or "")


def _emit_images_with_crops(emit_image: str, full_img, plan_img, side_img) -> None:
    if not emit_image:
        return
    try:
        full_path = emit_image if emit_image.lower().endswith(".png") else emit_image + "_full.png"
        full_img.save(full_path)
        plan_img.save(emit_image + "_plan.png")
        side_img.save(emit_image + "_side.png")
    except Exception:
        pass


def _emit_ocr_text(emit_ocr: str, text: str) -> None:
    if not emit_ocr or not text:
        return
    try:
        with open(emit_ocr, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[dims-ocr] saved OCR/VLM text to {emit_ocr}")
    except Exception:
        pass


def _fallback_dims_from_text(nums: List[float]) -> Optional[Tuple[float, float, float]]:
    if len(nums) < 3:
        return None

    uniq: List[float] = []
    for val in sorted(nums, reverse=True):
        if not any(abs(val - existing) < 1e-3 for existing in uniq):
            uniq.append(val)
        if len(uniq) >= 3:
            break

    if len(uniq) < 3:
        return None

    L, W, T = uniq[0], uniq[1], uniq[2]
    if W < T:
        W, T = T, W
    return (L, W, T)


def _vlm_local_process_render(img_path: str,
                              plan_crop: str,
                              side_crop: str,
                              vlm_local_model: str,
                              vlm_local_mmproj: str,
                              emit_image: str,
                              emit_ocr: str,
                              verbose: bool) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    if Image is None:
        raise RuntimeError("Pillow is required for VLM local rendering.")

    with Image.open(img_path) as img:
        img.load()
        plan_img = _crop_frac(img, plan_crop).copy()
        side_img = _crop_frac(img, side_crop).copy()

        _emit_images_with_crops(emit_image, img, plan_img, side_img)

        llm = _load_qwen_local(vlm_local_model, vlm_local_mmproj)
        data = _qwen_extract_block_dims_local(llm, img, plan_img, side_img)

        def _looks_like_default(triple: Tuple[float, float, float]) -> bool:
            L, W, T = triple
            return (
                abs(L - 12.0) < 1e-3
                and abs(W - 8.0) < 1e-3
                and abs(T - 2.0) < 1e-3
            )

        need_heuristic = not data
        parsed_dims: Optional[Tuple[float, float, float]] = None
        if data:
            try:
                parsed_dims = (float(data["length"]), float(data["width"]), float(data["thickness"]))
            except Exception:
                parsed_dims = None
                need_heuristic = True
            else:
                if _looks_like_default(parsed_dims):
                    need_heuristic = True

        guess: Optional[Tuple[float, float, float]] = None
        if need_heuristic:
            plan_nums = _extract_numbers_from_img(llm, plan_img)
            side_nums = _extract_numbers_from_img(llm, side_img)
            guess = _pick_dims(plan_nums, side_nums)
            if guess and verbose:
                L, W, T = guess
                print(f"[dims-ocr] heuristic -> SIZE: {L} x {W} x {T} in")

        if guess:
            L, W, T = guess
            text = f"SIZE: {L} x {W} x {T} in"
            _emit_ocr_text(emit_ocr, text)
            return text, warnings

        if data and parsed_dims:
            text = f"SIZE: {data['length']} x {data['width']} x {data['thickness']} {data['units']}"
            if verbose:
                print("[dims-ocr] VLM JSON ->", text)
            _emit_ocr_text(emit_ocr, text)
            return text, warnings

        txt_nums = _vlm_local_transcribe_all(llm, img)
        if verbose:
            print("[dims-ocr] OCR nums preview:", txt_nums[:200])
        nums: List[float] = []
        for token in re.findall(r"\d+\s*\d+/\d+|\d+[.,]\d+|\.\d+|\d+", txt_nums):
            token = token.replace(",", "")
            if re.match(r"^\d+\s+\d+/\d+$", token):
                a, b, c = re.match(r"^(\d+)\s+(\d+)/(\d+)$", token).groups()
                val = float(a) + float(b) / float(c)
            elif re.match(r"^\d+/\d+$", token):
                a, b = token.split("/")
                val = float(a) / float(b)
            else:
                val = float(token)
            nums.append(val)

        guess = _fallback_dims_from_text(nums)
        if guess:
            L, W, T = guess
            units = "in"
            text = f"SIZE: {L} x {W} x {T} {units}"
            _emit_ocr_text(emit_ocr, text)
            return text, warnings

        if txt_nums:
            _emit_ocr_text(emit_ocr, txt_nums)
        warnings.append("No dimension triples found.")
        return "", warnings
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

def _render_dxf_to_image(dxf_path: str, out_path: str, dpi: int = 600) -> str:
    """
    Render a DXF modelspace to a raster image (PNG/TIFF) using ezdxf's drawing add-on.
    Returns the path to the written image.
    """
    try:
        import ezdxf
        from ezdxf.addons.drawing import RenderContext, Frontend
        from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    except Exception as e:
        raise RuntimeError("ezdxf (and its drawing add-on) + matplotlib are required. pip install ezdxf matplotlib") from e

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Create a borderless figure
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    backend = MatplotlibBackend(ax)
    Frontend(ctx, backend).draw_layout(msp, finalize=True)

    # Fit the drawing nicely
    ax.autoscale(True)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


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
        if vlm_mode == "extract":
            data = _vlm_local_extract_dims(llm, img_path)
            if data:
                synthetic = _json_dims_to_synthetic_line(data)
                if synthetic:
                    return _norm(synthetic)
            warnings.append("vlm_local extract failed; falling back to transcription.")
        text = _vlm_local_transcribe_image(llm, img_path)
        return _norm(text)
    if backend == "vlm":
        if not vlm_endpoint:
            raise RuntimeError("Provide --vlm-endpoint for vlm backend.")
        if vlm_mode == "extract":
            data = _vlm_remote_extract_dims(vlm_endpoint, vlm_model, img_path)
            if data:
                synthetic = _json_dims_to_synthetic_line(data)
                if synthetic:
                    return _norm(synthetic)
            warnings.append("vlm extract failed; falling back to transcription.")
        text = _vlm_remote_transcribe_image(vlm_endpoint, vlm_model, img_path)
        return _norm(text)

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
                         plan_crop: str = "0,0,0.72,1.0",
                         side_crop: str = "0.72,0,1.0,1.0",
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
            _render_dxf_to_image(dxf_out, img_path, dpi=600)
            backend = (ocr_backend or "tesseract").lower()
            if backend == "vlm_local":
                text, local_warnings = _vlm_local_process_render(
                    img_path,
                    plan_crop,
                    side_crop,
                    vlm_local_model,
                    vlm_local_mmproj,
                    emit_image,
                    emit_ocr,
                    verbose,
                )
                warnings.extend(local_warnings)
                return (text, warnings)

            if emit_image and Image is not None:
                with Image.open(img_path) as img:
                    img.load()
                    plan_img = _crop_frac(img, plan_crop).copy()
                    side_img = _crop_frac(img, side_crop).copy()
                    _emit_images_with_crops(emit_image, img, plan_img, side_img)

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
                _emit_ocr_text(emit_ocr, txt)
            if verbose:
                print("[dims-ocr] OCR preview:", _norm(txt)[:500].replace("\n", " ⏎ "))
            return (txt, warnings)

    # DXF: DXF -> PNG -> OCR (same renderer)
    if ext == ".dxf":
        with tempfile.TemporaryDirectory(prefix="dxf_img_") as tmpdir:
            img_path = os.path.join(tmpdir, "page.png")
            _render_dxf_to_image(path, img_path, dpi=600)
            backend = (ocr_backend or "tesseract").lower()
            if backend == "vlm_local":
                text, local_warnings = _vlm_local_process_render(
                    img_path,
                    plan_crop,
                    side_crop,
                    vlm_local_model,
                    vlm_local_mmproj,
                    emit_image,
                    emit_ocr,
                    verbose,
                )
                warnings.extend(local_warnings)
                return (text, warnings)

            if emit_image and Image is not None:
                with Image.open(img_path) as img:
                    img.load()
                    plan_img = _crop_frac(img, plan_crop).copy()
                    side_img = _crop_frac(img, side_crop).copy()
                    _emit_images_with_crops(emit_image, img, plan_img, side_img)

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
                _emit_ocr_text(emit_ocr, txt)
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
            try:
                _ensure_parent_dir(emit_ocr)
                with open(emit_ocr, "w", encoding="utf-8") as f:
                    f.write(combined)
                print(f"[dims-ocr] saved OCR/VLM text to {emit_ocr}")
            except Exception:
                pass
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
        try:
            _ensure_parent_dir(emit_ocr)
            with open(emit_ocr, "w", encoding="utf-8") as f:
                f.write(txt)
            print(f"[dims-ocr] saved OCR/VLM text to {emit_ocr}")
        except Exception:
            pass
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
                      plan_crop: str = "0,0,0.72,1.0",
                      side_crop: str = "0.72,0,1.0,1.0",
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
        plan_crop: Fractional crop for the plan/top view (left,top,right,bottom).
        side_crop: Fractional crop for the side view (left,top,right,bottom).
        emit_image: Optional prefix to save rendered CAD image + crops for debugging.
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
        plan_crop=plan_crop,
        side_crop=side_crop,
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
    json_out_help = f"Optional: write result JSON here (default: {default_json_out})."
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
        f"If set, save the rendered PNG/TIFF here for debugging (default: {default_emit_image})."
    )
    emit_ocr_help = (
        f"If set, save the OCR/VLM text here for debugging (default: {default_emit_ocr})."
    )
    verbose_help = "Print a short preview of OCR text (default: enabled; use --no-verbose to disable)."

    p.add_argument("--input", default=default_input, help=input_help)
    p.add_argument("--units", choices=["auto", "in", "mm"], default="auto", help="Units preference (default auto).")
    p.add_argument("--prefer-stock", action="store_true", help="Bias lines starting with STOCK.")
    p.add_argument("--use-llm", action="store_true", help="If multiple candidates, ask LLM to pick best (requires OPENAI_API_KEY/local model).")
    p.add_argument("--json-out", default=default_json_out, help=json_out_help)
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
    p.add_argument("--emit-image", default=default_emit_image, help=emit_image_help)
    p.add_argument("--emit-ocr", default=default_emit_ocr, help=emit_ocr_help)
    p.add_argument("--verbose", action="store_true", dest="verbose", help=verbose_help)
    p.add_argument("--no-verbose", action="store_false", dest="verbose", help="Disable OCR preview output.")
    p.add_argument("--plan-crop", default="0,0,0.72,1.0", help="left,top,right,bottom as fractions (plan/top view)")
    p.add_argument("--side-crop", default="0.72,0,1.0,1.0", help="left,top,right,bottom as fractions (side view)")
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
            plan_crop=args.plan_crop,
            side_crop=args.side_crop,
            emit_image=args.emit_image,
            emit_ocr=args.emit_ocr,
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
        if args.json_out:
            _ensure_parent_dir(args.json_out)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"[dims] wrote {args.json_out}")
        else:
            print(json.dumps(out, indent=2))
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    _cli()
