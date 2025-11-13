"""
vlm_dims_extractor.py
=====================
Simplified VLM-based dimension extraction for CAD drawings.

Features:
- DWG/DXF -> PNG conversion via ODA + ezdxf
- Local VLM (llama.cpp) for dimension extraction
- Multi-stage extraction with fallbacks
- Clear validation and error reporting

Usage:
    python vlm_dims_extractor.py --input "path/to/drawing.dwg"
"""

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Default configuration paths
DEFAULT_MODEL_PATH = r"D:\CAD_Quoting_Tool\models\qwen2.5-vl-7b-instruct-q4_k_m.gguf"
DEFAULT_MMPROJ_PATH = r"D:\CAD_Quoting_Tool\models\mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
DEFAULT_ODA_EXE = r"D:\ODA\ODAFileConverter 26.8.0\ODAFileConverter.exe"
DEFAULT_OUTPUT_DIR = r"D:\CAD_Quoting_Tool\debug"

# Check for required libraries
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import ezdxf
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    from ezdxf.addons.drawing.config import Configuration, ColorPolicy, BackgroundPolicy, LinePolicy
except ImportError:
    ezdxf = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None


@dataclass
class Dimensions:
    """Represents extracted part dimensions."""
    length: float
    width: float
    thickness: float
    units: str
    confidence: str
    method: str


class VLMDimensionExtractor:
    """Extracts dimensions from CAD drawings using a local VLM."""
    
    def __init__(self, model_path: str, mmproj_path: str, verbose: bool = False):
        """Initialize the extractor with model paths."""
        self.model_path = Path(model_path)
        self.mmproj_path = Path(mmproj_path)
        self.verbose = verbose
        self.llm = None
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not self.mmproj_path.exists():
            raise FileNotFoundError(f"MMProj not found: {mmproj_path}")
        
        if Llama is None:
            raise ImportError("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        
        self._load_model()
    
    def _load_model(self):
        """Load the VLM model."""
        if self.verbose:
            print(f"[VLM] Loading model: {self.model_path.name}")
        
        self.llm = Llama(
            model_path=str(self.model_path),
            mmproj=str(self.mmproj_path),
            n_ctx=4096,
            verbose=False
        )
        
        if self.verbose:
            print("[VLM] Model loaded successfully")
    
    def _img_to_base64(self, pil_image: Image.Image) -> str:
        """Convert PIL image to base64 string."""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    
    def _query_vlm(self, pil_image: Image.Image, prompt: str, max_tokens: int = 512) -> str:
        """Query the VLM with an image and prompt."""
        import hashlib

        b64_image = self._img_to_base64(pil_image)

        # Print image fingerprint for debugging cache issues
        if self.verbose:
            img_hash = hashlib.md5(b64_image.encode()).hexdigest()[:12]
            print(f"[VLM] Image fingerprint: {img_hash}, size: {pil_image.size}")

        response = self.llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }],
            temperature=0.1,  # Slight randomness to prevent caching
            max_tokens=max_tokens,
            seed=-1  # Random seed each time
        )

        return (response["choices"][0]["message"]["content"] or "").strip()
    
    def _extract_dimensions_structured(self, pil_image: Image.Image) -> Optional[Dict[str, Any]]:
        """Extract dimensions using structured JSON prompt."""
        prompt = """You are analyzing a mechanical/CAD drawing. Your task is to find the OUTER BOUNDING BOX dimensions of the MAIN part shown.

WHERE TO LOOK:
1. Focus ONLY on the main technical drawing (usually on the LEFT side)
2. COMPLETELY IGNORE any parts tables, revision tables, or text blocks on the right side
3. Look for dimension lines with arrows pointing to the OUTER EDGES of the part outline
4. These dimension lines usually have the LARGEST numbers

WHAT TO FIND:
- Find the dimension line along the BOTTOM edge (this is usually length or width)
- Find the dimension line along the LEFT or RIGHT edge (this is the other planar dimension)
- Find the thickness from a SIDE VIEW or SECTION VIEW (usually 0.05-6.0 inches)

READ THE EXACT NUMBERS from these dimension lines. Be precise with decimals.

COMPLETELY IGNORE:
- Parts tables (tables with hole coordinates, part lists, etc.)
- Hole diameters (Ø symbol)
- Angles (45°, 90°, etc.)
- Internal feature dimensions
- Revision numbers
- Scale ratios

Return ONLY valid JSON with exact numbers you can read:
{
  "length": <number from bottom or top edge>,
  "width": <number from left or right edge>,
  "thickness": <number from side view>,
  "units": "in"
}

If you cannot read a dimension clearly, use null for that value."""

        response = self._query_vlm(pil_image, prompt, max_tokens=256)
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if self.verbose:
                    print(f"[VLM] Structured response: {json.dumps(data, indent=2)}")
                return data
            except json.JSONDecodeError as e:
                if self.verbose:
                    print(f"[VLM] JSON parse error: {e}")
        
        if self.verbose:
            print(f"[VLM] Raw response: {response[:200]}")
        
        return None
    
    def _extract_all_numbers(self, pil_image: Image.Image) -> List[float]:
        """Extract all visible numbers from the drawing."""
        prompt = """List ALL numeric dimensions you can read in this mechanical drawing, ESPECIALLY the LARGEST numbers.

PRIORITY: Focus on dimensions at the OUTER EDGES of the part (these are usually the largest numbers like 15.5, 19.0, 12.0).
Include ALL decimal values with full precision.

Output ONLY numbers separated by spaces (e.g., "19.00 15.5 12.0 11.5 2.0 0.5005").

Do NOT include:
- Angles (like 45° or 90°)
- Hole diameters with Ø symbol
- Coordinate table numbers
- Scale factors (like 1:1 or 2:1)
- Part numbers or revision numbers"""

        response = self._query_vlm(pil_image, prompt, max_tokens=512)

        if self.verbose:
            print(f"[VLM] All numbers response: {response[:300]}")

        # Extract all numbers from response (including decimals)
        number_pattern = r'\d+\.?\d*'
        all_numbers = [float(m.group()) for m in re.finditer(number_pattern, response)]

        if self.verbose:
            print(f"[VLM] Raw numbers found: {sorted(set(all_numbers), reverse=True)}")

        # Filter out common angles
        common_angles = {0, 15, 30, 45, 60, 90, 120, 135, 180, 270, 360}
        numbers = [n for n in all_numbers if n not in common_angles]

        if self.verbose:
            print(f"[VLM] After filtering angles: {sorted(set(numbers), reverse=True)}")

        return sorted(set(numbers), reverse=True)
    
    def _infer_from_numbers(self, numbers: List[float]) -> Optional[Tuple[float, float, float]]:
        """Infer L, W, T from a list of numbers using heuristics."""
        if len(numbers) < 3:
            return None

        # Filter out obviously wrong values
        # Typical plate: thickness 0.05-6.0, L/W typically 1.0-100.0
        valid_thickness = [n for n in numbers if 0.05 <= n <= 6.0]
        valid_planar = [n for n in numbers if 1.0 <= n <= 100.0]

        if self.verbose:
            print(f"[VLM] Valid thickness candidates: {sorted(valid_thickness, reverse=True)}")
            print(f"[VLM] Valid planar candidates: {sorted(valid_planar, reverse=True)}")

        if not valid_thickness or len(valid_planar) < 2:
            if self.verbose:
                print(f"[VLM] Insufficient valid candidates")
            return None

        # Pick thickness: smallest value in valid range, prefer values with more decimals
        def thickness_score(t: float) -> Tuple[float, int]:
            t_str = f"{t:.6f}".rstrip('0').rstrip('.')
            decimals = len(t_str.split('.')[-1]) if '.' in t_str else 0
            # Prefer smaller values (thickness is usually smallest) and more precision
            return (t, -decimals)  # Negative decimals so more decimals = lower score

        thickness = min(valid_thickness, key=thickness_score)

        if self.verbose:
            print(f"[VLM] Selected thickness: {thickness}")

        # Pick L and W: Find the two largest values that are reasonable stock dimensions
        # Filter candidates: must be significantly larger than thickness
        candidates = [n for n in valid_planar if n >= thickness * 3]

        # Further filter: remove values that are too large to be stock dimensions
        # Most stock plates are under 50 inches in any dimension
        candidates = [n for n in candidates if n <= 50.0]

        if len(candidates) < 2:
            # Fallback to all valid planar dimensions
            candidates = [n for n in valid_planar if n <= 50.0]

        if len(candidates) >= 2:
            # Strategy: For stock dimensions, prefer the LARGEST valid pair
            # Stock dimensions are the outer envelope, so bigger is better

            # Pick the two largest values
            sorted_candidates = sorted(candidates, reverse=True)
            length = sorted_candidates[0]
            width = sorted_candidates[1]

            # Sanity check: make sure they're reasonable
            if length > 0 and width > 0 and length >= width:
                if self.verbose:
                    print(f"[VLM] Selected L={length}, W={width} from candidates: {sorted_candidates[:5]}")
                return length, width, thickness

        return None
    
    def _validate_dimensions(self, L: float, W: float, T: float) -> bool:
        """Validate that dimensions are physically plausible for a plate."""
        if any(v is None or v <= 0 for v in [L, W, T]):
            return False
        
        # Thickness should be reasonable for a plate
        if not (0.05 <= T <= 6.0):
            return False
        
        # Length and width should be significantly larger than thickness
        if min(L, W) < T * 2.5:
            return False
        
        # Dimensions should be reasonable in general
        if max(L, W) > 500 or T > 20:
            return False
        
        return True
    
    def _query_envelope_dimensions(self, pil_image: Image.Image) -> Optional[Tuple[float, float]]:
        """Ask specifically about the outer envelope dimensions."""
        prompt = """Look at this CAD drawing and find the MAXIMUM OUTER DIMENSIONS.

What are the TWO LARGEST dimension numbers that define the outer bounding box of this part?

Look for dimension lines at the VERY OUTER EDGES pointing to the part outline.
These are typically the LARGEST numbers on the drawing (e.g., 15.5, 19.0, 12.0).

Answer with ONLY the two largest dimension numbers separated by a space (e.g., "19.0 15.5").
Do NOT include: angles, hole sizes, thickness, or internal features."""

        response = self._query_vlm(pil_image, prompt, max_tokens=64)

        # Extract exactly 2 numbers from response
        number_pattern = r'\d+\.?\d*'
        numbers = [float(m.group()) for m in re.finditer(number_pattern, response)]

        if len(numbers) >= 2:
            if self.verbose:
                print(f"[VLM] Envelope query returned: {numbers[:2]}")
            return (numbers[0], numbers[1])

        return None

    def _crop_to_main_drawing(self, pil_image: Image.Image) -> Image.Image:
        """Crop image to focus on main drawing, removing parts tables."""
        width, height = pil_image.size

        # Most CAD drawings have parts tables on the right side
        # Crop to left 65% to focus on the main technical drawing
        crop_width = int(width * 0.65)

        if self.verbose:
            print(f"[VLM] Cropping from {width}x{height} to {crop_width}x{height} (removing parts table)")

        return pil_image.crop((0, 0, crop_width, height))

    def extract(self, pil_image: Image.Image) -> Optional[Dimensions]:
        """Extract dimensions from a rendered CAD drawing image."""
        if self.verbose:
            print("[VLM] Starting dimension extraction...")

        # Crop to main drawing area (remove parts tables)
        cropped_image = self._crop_to_main_drawing(pil_image)

        # Method 1: Structured extraction
        structured_data = self._extract_dimensions_structured(cropped_image)

        if structured_data:
            try:
                L = float(structured_data.get("length", 0))
                W = float(structured_data.get("width", 0))
                T = float(structured_data.get("thickness", 0))
                units = structured_data.get("units", "in")

                if self._validate_dimensions(L, W, T):
                    if self.verbose:
                        print(f"[VLM] [OK] Structured method: {L} x {W} x {T} {units}")
                    return Dimensions(L, W, T, units, "high", "structured_json")
                else:
                    if self.verbose:
                        print(f"[VLM] [FAIL] Structured values invalid: {L} x {W} x {T}")
            except (TypeError, ValueError) as e:
                if self.verbose:
                    print(f"[VLM] [FAIL] Structured method error: {e}")

        # Method 2: Extract all numbers and infer
        if self.verbose:
            print("[VLM] Trying numeric inference method...")

        numbers = self._extract_all_numbers(cropped_image)
        inferred = self._infer_from_numbers(numbers)

        if inferred:
            L, W, T = inferred

            # Method 2a: Refine with envelope query if we have multiple candidates
            if self.verbose:
                print("[VLM] Refining with envelope query...")

            envelope = self._query_envelope_dimensions(cropped_image)
            if envelope:
                env_l, env_w = envelope
                # Use envelope dimensions if they're in our extracted numbers
                if env_l in numbers and env_w in numbers:
                    L = max(env_l, env_w)
                    W = min(env_l, env_w)
                    if self.verbose:
                        print(f"[VLM] [REFINED] Using envelope dims: {L} x {W}")

            if self._validate_dimensions(L, W, T):
                if self.verbose:
                    print(f"[VLM] [OK] Inference method: {L} x {W} x {T} in")
                return Dimensions(L, W, T, "in", "medium", "numeric_inference")

        if self.verbose:
            print("[VLM] [FAIL] All extraction methods failed")

        return None
    
    def __del__(self):
        """Cleanup."""
        if self.llm is not None:
            try:
                del self.llm
            except:
                pass


class DrawingRenderer:
    """Renders CAD drawings (DWG/DXF) to PNG images."""
    
    def __init__(self, oda_exe: Optional[str] = None, verbose: bool = False):
        """Initialize renderer with ODA converter path."""
        self.verbose = verbose
        self.oda_exe = oda_exe or self._find_oda_converter()
        
        if ezdxf is None or matplotlib is None:
            raise ImportError("ezdxf and matplotlib required. Run: pip install ezdxf matplotlib")
        
        if Image is None:
            raise ImportError("Pillow required. Run: pip install pillow")
    
    def _find_oda_converter(self) -> Optional[str]:
        """Try to locate ODA File Converter."""
        # Check environment variable
        env_path = os.getenv("ODA_FILE_CONVERTER")
        if env_path and Path(env_path).exists():
            return env_path
        
        # Check common install locations
        common_paths = [
            r"C:\Program Files\ODA\OdaFileConverter.exe",
            r"C:\Program Files (x86)\ODA\OdaFileConverter.exe",
        ]
        
        for path in common_paths:
            if Path(path).exists():
                return path
        
        # Check PATH
        oda_path = shutil.which("OdaFileConverter.exe")
        if oda_path:
            return oda_path
        
        return None
    
    def _dwg_to_dxf(self, dwg_path: str, output_dir: str) -> str:
        """Convert DWG to DXF using ODA File Converter."""
        if not self.oda_exe:
            raise RuntimeError("ODA File Converter not found. Install from: https://www.opendesign.com/guestfiles/oda_file_converter")
        
        if not Path(self.oda_exe).exists():
            raise FileNotFoundError(f"ODA converter not found: {self.oda_exe}")
        
        # Create input directory and copy file
        input_dir = Path(output_dir) / "_oda_input"
        input_dir.mkdir(exist_ok=True)
        
        basename = Path(dwg_path).name
        input_file = input_dir / basename
        shutil.copy2(dwg_path, input_file)
        
        # Create output directory
        dxf_dir = Path(output_dir) / "_oda_output"
        dxf_dir.mkdir(exist_ok=True)
        
        # Run ODA converter
        cmd = [
            self.oda_exe,
            str(input_dir),
            str(dxf_dir),
            "ACAD2018",  # Output version
            "DXF",       # Output format
            "0",         # Recurse subdirectories
            "1",         # Audit
            basename     # File filter
        ]
        
        if self.verbose:
            print(f"[ODA] Converting: {basename}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"ODA conversion failed: {result.stderr}")
        
        # Find output DXF
        stem = Path(dwg_path).stem
        dxf_candidates = list(dxf_dir.glob(f"{stem}*.dxf"))
        
        if not dxf_candidates:
            raise RuntimeError(f"No DXF output found for {basename}")
        
        dxf_path = str(dxf_candidates[0])
        
        if self.verbose:
            print(f"[ODA] Created: {Path(dxf_path).name}")
        
        return dxf_path
    
    def _dxf_to_png(self, dxf_path: str, png_path: str, dpi: int = 800, max_dimension: int = 7000):
        """Render DXF to high-contrast PNG with size constraints."""
        if self.verbose:
            print(f"[DXF] Rendering to PNG (DPI={dpi})...")

        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        # High-contrast configuration
        config = Configuration(
            color_policy=ColorPolicy.BLACK,
            background_policy=BackgroundPolicy.WHITE,
            line_policy=LinePolicy.ACCURATE,
            min_lineweight=0.35,
            lineweight_scaling=1.8,
        )

        ctx = RenderContext(doc)

        fig = plt.figure(figsize=(14, 10), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        backend = MatplotlibBackend(ax, adjust_figure=True)
        Frontend(ctx, backend, config=config).draw_layout(msp, finalize=True)

        fig.savefig(png_path, dpi=dpi, facecolor="#FFFFFF")
        plt.close(fig)

        # Check and resize if needed to stay under max_dimension
        with Image.open(png_path) as img:
            width, height = img.size

            if width > max_dimension or height > max_dimension:
                # Calculate scale factor to fit within max_dimension
                scale = min(max_dimension / width, max_dimension / height)
                new_width = int(width * scale)
                new_height = int(height * scale)

                if self.verbose:
                    print(f"[DXF] Resizing from {width}x{height} to {new_width}x{new_height} (max={max_dimension})")

                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img_resized.save(png_path, "PNG")
            elif self.verbose:
                print(f"[DXF] Image size OK: {width}x{height}")

        if self.verbose:
            print(f"[DXF] Saved: {Path(png_path).name}")
    
    def render(self, input_path: str, output_png: str) -> str:
        """Render CAD file to PNG. Returns path to PNG."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        ext = input_path.suffix.lower()
        
        # If already an image, just copy it
        if ext in ['.png', '.jpg', '.jpeg']:
            shutil.copy2(input_path, output_png)
            return output_png
        
        # DXF: render directly
        if ext == '.dxf':
            self._dxf_to_png(str(input_path), output_png)
            return output_png
        
        # DWG: convert to DXF first
        if ext == '.dwg':
            with tempfile.TemporaryDirectory(prefix="dwg_render_") as tmpdir:
                dxf_path = self._dwg_to_dxf(str(input_path), tmpdir)
                self._dxf_to_png(dxf_path, output_png)
            return output_png
        
        raise ValueError(f"Unsupported file type: {ext}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract part dimensions from CAD drawings using VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Default Configuration:
  Edit the DEFAULT_* constants at the top of this script to change default paths.
"""
    )

    # Input/output
    parser.add_argument("--input", required=True, help="Path to CAD file (DWG/DXF/PNG)")
    parser.add_argument("--output-json", help=f"Output JSON file path (default: {DEFAULT_OUTPUT_DIR}/<input>_dims.json)")
    parser.add_argument("--output-png", help=f"Save rendered PNG (default: {DEFAULT_OUTPUT_DIR}/<input>_render.png)")

    # VLM model paths
    parser.add_argument("--model", help=f"Path to VLM model (.gguf) (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--mmproj", help=f"Path to mmproj file (.gguf) (default: {DEFAULT_MMPROJ_PATH})")

    # ODA converter
    parser.add_argument("--oda-exe", help=f"Path to ODA File Converter executable (default: {DEFAULT_ODA_EXE})")

    # Options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Determine output paths
    input_path = Path(args.input)

    if args.output_json:
        json_path = args.output_json
    else:
        json_path = str(Path(DEFAULT_OUTPUT_DIR) / f"{input_path.stem}_dims.json")

    if args.output_png:
        png_path = args.output_png
    else:
        png_path = str(Path(DEFAULT_OUTPUT_DIR) / f"{input_path.stem}_render.png")

    # Find model paths (priority: CLI args > defaults > auto-detect)
    model_path = args.model or DEFAULT_MODEL_PATH
    mmproj_path = args.mmproj or DEFAULT_MMPROJ_PATH

    # Verify paths exist, or try auto-detect
    if not Path(model_path).exists() or not Path(mmproj_path).exists():
        # Try to find in ./models directory
        models_dir = Path(__file__).parent / "models"

        if not Path(model_path).exists():
            candidates = list(models_dir.glob("*qwen*vl*.gguf"))
            candidates = [c for c in candidates if "mmproj" not in c.name.lower()]
            if candidates:
                model_path = str(candidates[0])

        if not Path(mmproj_path).exists():
            candidates = list(models_dir.glob("*mmproj*.gguf"))
            if candidates:
                mmproj_path = str(candidates[0])

    if not Path(model_path).exists() or not Path(mmproj_path).exists():
        print("Error: VLM model paths not found.", file=sys.stderr)
        print("Specify --model and --mmproj, or edit DEFAULT_MODEL_PATH and DEFAULT_MMPROJ_PATH in the script.", file=sys.stderr)
        sys.exit(1)

    # Get ODA exe path (priority: CLI args > default)
    oda_exe = args.oda_exe or DEFAULT_ODA_EXE
    
    try:
        # Step 1: Render drawing to PNG
        print(f"[1/3] Rendering: {input_path.name}")
        renderer = DrawingRenderer(oda_exe=oda_exe, verbose=args.verbose)
        renderer.render(str(input_path), png_path)
        
        # Step 2: Load VLM
        print(f"[2/3] Loading VLM...")
        extractor = VLMDimensionExtractor(model_path, mmproj_path, verbose=args.verbose)
        
        # Step 3: Extract dimensions
        print(f"[3/3] Extracting dimensions...")
        with Image.open(png_path) as img:
            img = img.convert("RGB")
            dims = extractor.extract(img)
        
        if dims is None:
            print("\n[ERROR] Failed to extract dimensions", file=sys.stderr)
            sys.exit(1)
        
        # Save results
        result = {
            "length": dims.length,
            "width": dims.width,
            "thickness": dims.thickness,
            "units": dims.units,
            "confidence": dims.confidence,
            "extraction_method": dims.method,
            "source_file": str(input_path),
            "rendered_image": png_path
        }
        
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Print results
        print(f"\n[SUCCESS] Dimensions extracted successfully:")
        print(f"  Length:    {dims.length} {dims.units}")
        print(f"  Width:     {dims.width} {dims.units}")
        print(f"  Thickness: {dims.thickness} {dims.units}")
        print(f"  Method:    {dims.method}")
        print(f"  Confidence: {dims.confidence}")
        print(f"\n[JSON] Saved to: {json_path}")
        print(f"[PNG]  Render:  {png_path}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()