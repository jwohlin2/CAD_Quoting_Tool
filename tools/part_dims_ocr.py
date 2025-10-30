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
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

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

_NUM = r"(?:\d+(?:[.,]\d+)?|\d+\s*\d+/\d+)"  # 12, 12.00, 1 3/8
_SEP = r"[xX×]"

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


def _run_oda_convert(oda_exe: str, in_path: str, out_dir: str,
                     out_format: str = "PDF",
                     in_ver: str = "ACAD2018",
                     out_ver: str = "ACAD2018",
                     audit: int = 0,
                     recover: int = 0) -> str:
    """
    Call ODA File Converter. ODA expects *directories* for in/out.
    We put the single DWG/DXF into a temp input dir, run the converter,
    and return the resulting file path (PDF/TIFF/PNG).
    """
    if not os.path.exists(oda_exe):
        raise FileNotFoundError(f"ODA File Converter not found: {oda_exe}")

    # Prepare temp input dir containing our single file
    tmp_in = os.path.join(out_dir, "_oda_in")
    os.makedirs(tmp_in, exist_ok=True)
    base = os.path.basename(in_path)
    src = os.path.join(tmp_in, base)
    shutil.copy2(in_path, src)

    # Output directory (ODA writes converted files here)
    tmp_out = os.path.join(out_dir, "_oda_out")
    os.makedirs(tmp_out, exist_ok=True)

    # Build command:
    # ODAFileConverter <in_dir> <out_dir> <in_ver> <out_ver> <audit> <recover> <out_format>
    # out_format: PDF, BMP, TIFF, etc. PNG often appears as “BMP” family; TIFF is reliable for OCR.
    cmd = [
        oda_exe,
        tmp_in,
        tmp_out,
        in_ver,
        out_ver,
        str(int(bool(audit))),
        str(int(bool(recover))),
        out_format.upper(),
    ]

    # Run
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ODA conversion failed: {e.stderr.decode(errors='ignore') or e.stdout.decode(errors='ignore')}")

    # Find produced file
    name_wo_ext, _ = os.path.splitext(base)
    # ODA often appends extension based on chosen out_format
    candidates = []
    if out_format.upper() == "PDF":
        candidates.append(os.path.join(tmp_out, name_wo_ext + ".pdf"))
    elif out_format.upper() in ("TIFF", "TIF"):
        # Could be multi-page: name_wo_ext.tif or name_wo_ext_1.tif, etc.
        for fname in os.listdir(tmp_out):
            if fname.lower().startswith(name_wo_ext.lower()) and fname.lower().endswith((".tif", ".tiff")):
                candidates.append(os.path.join(tmp_out, fname))
    elif out_format.upper() in ("PNG", "BMP", "JPG", "JPEG"):
        for fname in os.listdir(tmp_out):
            if fname.lower().startswith(name_wo_ext.lower()) and fname.lower().endswith(("."+out_format.lower(),)):
                candidates.append(os.path.join(tmp_out, fname))
    else:
        # Default: scan for anything with our base name
        for fname in os.listdir(tmp_out):
            if fname.lower().startswith(name_wo_ext.lower()):
                candidates.append(os.path.join(tmp_out, fname))

    if not candidates:
        # Fallback: pick the first file in out dir
        outs = [os.path.join(tmp_out, f) for f in os.listdir(tmp_out)]
        if not outs:
            raise RuntimeError("ODA conversion produced no files.")
        candidates = outs

    # Return first candidate (PDF preferred if multiple)
    candidates.sort()
    return candidates[0]


def _get_text_from_input(path: str,
                         oda_exe_cli: Optional[str] = None,
                         oda_format: str = "PDF") -> Tuple[str, List[str]]:
    """
    Returns (text, warnings). If input is DWG/DXF, uses ODA File Converter to render to
    PDF or image first, then OCRs the result. For PDF/image inputs, OCRs directly.
    """
    warnings: List[str] = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    texts: List[str] = []

    # DWG/DXF: render via ODA to PDF/TIFF/PNG, then OCR that artifact
    if ext in (".dwg", ".dxf"):
        oda_exe = _which_oda(oda_exe_cli)
        if not oda_exe:
            raise RuntimeError(
                "ODA File Converter not found.\n"
                "Provide --oda-exe path, or set ODA_CONVERTER_EXE env var, "
                "or add OdaFileConverter.exe to your PATH."
            )
        with tempfile.TemporaryDirectory(prefix="oda_ocr_") as tmpdir:
            out_path = _run_oda_convert(oda_exe, path, tmpdir, out_format=oda_format)
            out_ext = os.path.splitext(out_path)[1].lower()
            if out_ext == ".pdf":
                if convert_from_path is None:
                    raise RuntimeError("pdf2image not available to render ODA PDF.")
                pages = convert_from_path(out_path, dpi=300)
                if not pages:
                    warnings.append("No pages rendered from ODA PDF.")
                for p in pages:
                    img = _preprocess_image(p) if Image else p
                    txt = _ocr_image(img)
                    texts.append(txt)
            else:
                # Single raster (TIFF/PNG/BMP/JPG)
                if Image is None:
                    raise RuntimeError("Pillow not installed to load ODA-rendered image.")
                img = Image.open(out_path)
                img = _preprocess_image(img)
                texts.append(_ocr_image(img))
        return ("\n".join(texts), warnings)

    # Native PDF
    if ext == ".pdf":
        if convert_from_path is None:
            raise RuntimeError("pdf2image not available to render PDFs.")
        pages = _render_pdf_to_images(path)
        if not pages:
            warnings.append("No pages rendered from PDF.")
        for p in pages:
            img = _preprocess_image(p) if Image else p
            txt = _ocr_image(img)
            texts.append(txt)
        return ("\n".join(texts), warnings)

    # Raster images
    if Image is None:
        raise RuntimeError("Pillow not installed to load images.")
    img = Image.open(path)
    img = _preprocess_image(img)
    texts.append(_ocr_image(img))
    return ("\n".join(texts), warnings)

# ---------- Regex candidates ----------

_PATTERNS = [
    # SIZE: 12.00 x 14.00 x 2.00 in
    re.compile(fr"\bSIZE\s*[:\-]?\s*({_NUM})\s*{_SEP}\s*({_NUM})\s*{_SEP}\s*({_NUM})\s*(in|inch|inches|mm|millimeters)?\b", re.I),
    # STOCK: 18 x 18 x 3.5 IN
    re.compile(fr"\bSTOCK\s*[:\-]?\s*({_NUM})\s*{_SEP}\s*({_NUM})\s*{_SEP}\s*({_NUM})\s*(in|inch|inches|mm|millimeters)?\b", re.I),
    # L=12  W=14  T=2  (units optional nearby)
    re.compile(fr"\bL\s*[:=]\s*({_NUM}).*?\bW\s*[:=]\s*({_NUM}).*?\bT\s*[:=]\s*({_NUM})(?:\s*(in|inch|inches|mm|millimeters))?", re.I),
    # Plain triplet: 12 x 14 x 2 (INCH)
    re.compile(fr"\b({_NUM})\s*{_SEP}\s*({_NUM})\s*{_SEP}\s*({_NUM})\s*(?:\((in|inch|inches|mm|millimeters)\))?\b", re.I),
    # Title block "MATERIAL / SIZE 12 x 14 x 2"
    re.compile(fr"\bSIZE\b.*?\b({_NUM})\s*{_SEP}\s*({_NUM})\s*{_SEP}\s*({_NUM})\b(?:\s*(in|inch|inches|mm|millimeters))?", re.I),
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
    Optional: ask an LLM to choose the best triple.
    Requires OPENAI_API_KEY and openai>=1.x installed; if not available, returns None.
    """
    try:
        import openai  # type: ignore
    except Exception:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    openai.api_key = api_key

    # Build a compact prompt
    options = []
    for idx, c in enumerate(cands):
        options.append({
            "index": idx,
            "length": c.length,
            "width": c.width,
            "thickness": c.thickness,
            "units": c.units,
            "source": c.source[:200]
        })

    try:
        # Use the new Responses API if available; fall back to chat otherwise
        # We keep it simple: ask for the index of the best option.
        prompt = (
            "You are selecting overall part dimensions from OCR text.\n"
            "Pick the most likely Length, Width, Thickness triple (in the drawing units) used for stock selection.\n"
            "Prefer values explicitly labeled as SIZE or STOCK; avoid feature dims.\n"
            "Return ONLY the integer index of the best option.\n\n"
            f"Options: {json.dumps(options)}\n\n"
            f"OCR Text (truncated): {ocr_text[:3000]}"
        )
        try:
            # Modern client (responses)
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            resp = client.responses.create(
                model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                input=prompt,
            )
            content = resp.output[0].content[0].text.strip()  # type: ignore
        except Exception:
            # Legacy chat completion
            content = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                messages=[{"role":"user","content":prompt}],
                temperature=0,
            )["choices"][0]["message"]["content"].strip()
        m = re.search(r"(\d+)", content)
        return int(m.group(1)) if m else None
    except Exception:
        return None

# ---------- Public API ----------

def extract_part_dims(input_path: str,
                      units: str = "auto",
                      prefer_stock: bool = False,
                      use_llm: bool = False,
                      oda_exe: Optional[str] = None,
                      oda_format: str = "PDF") -> ExtractionResult:
    """
    Extract (L, W, T) from a PDF/image drawing.

    Args:
        input_path: path to the source PDF/image.
        units: "in", "mm", or "auto" (auto = detect; inches-biased).
        prefer_stock: small score bump to lines starting with "STOCK".
        use_llm: if True, and multiple candidates exist, ask LLM to pick best.
        oda_exe: optional explicit path to ODA File Converter for DWG/DXF inputs.
        oda_format: ODA output format prior to OCR (PDF/TIFF/PNG).

    Returns:
        ExtractionResult with best triple (if any), all candidates, raw OCR text, warnings.
    """
    warnings: List[str] = []
    text, w = _get_text_from_input(input_path, oda_exe_cli=oda_exe, oda_format=oda_format)
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
    p = argparse.ArgumentParser(description="Extract part L/W/T from drawing via OCR (with optional LLM arbitration).")
    p.add_argument("--input", required=True, help="Path to PDF or image.")
    p.add_argument("--units", choices=["auto","in","mm"], default="auto", help="Units preference (default auto).")
    p.add_argument("--prefer-stock", action="store_true", help="Bias lines starting with STOCK.")
    p.add_argument("--use-llm", action="store_true", help="If multiple candidates, ask LLM to pick best (requires OPENAI_API_KEY).")
    p.add_argument("--json-out", default="", help="Optional: write result JSON here.")
    p.add_argument("--oda-exe", default="", help="Path to ODA File Converter executable. "
                                                "Alternatively set ODA_CONVERTER_EXE env var or put it on PATH.")
    p.add_argument("--oda-format", choices=["PDF","TIFF","PNG"], default="PDF",
                   help="ODA output format before OCR (default PDF).")
    args = p.parse_args()

    try:
        res = extract_part_dims(
            args.input,
            units=args.units,
            prefer_stock=args.prefer_stock,
            use_llm=args.use_llm,
            oda_exe=(args.oda_exe or None),
            oda_format=args.oda_format
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
