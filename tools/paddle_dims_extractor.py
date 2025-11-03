"""
paddle_dims_extractor.py
========================
PaddleOCR-based dimension extraction for CAD drawings.

Features:
- DWG/DXF -> PNG conversion via ODA + ezdxf
- PaddleOCR for reliable text extraction
- Smart dimension inference with heuristics
- Clear validation and error reporting

Installation:
    pip install paddlepaddle paddleocr pillow ezdxf matplotlib

Usage:
    python paddle_dims_extractor.py --input "path/to/drawing.dwg"
"""

import argparse
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
DEFAULT_ODA_EXE = r"D:\ODA\ODAFileConverter 26.8.0\ODAFileConverter.exe"
DEFAULT_OUTPUT_DIR = r"D:\CAD_Quoting_Tool\debug"

# Check for required libraries
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

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
    all_numbers: List[float] = None


class PaddleOCRDimensionExtractor:
    """Extracts dimensions from CAD drawings using PaddleOCR."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the extractor with PaddleOCR."""
        self.verbose = verbose
        self.ocr = None
        self._all_ocr_detections = []
        self._all_ocr_detections_vized = []
        
        if PaddleOCR is None:
            raise ImportError(
                "PaddleOCR not installed. Run:\n"
                "  pip install paddlepaddle paddleocr\n"
                "Or for GPU: pip install paddlepaddle-gpu paddleocr"
            )
        
        self._load_ocr()
    
    def _load_ocr(self):
        """Load PaddleOCR model."""
        if self.verbose:
            print("[OCR] Loading PaddleOCR model...")
        
        # Initialize PaddleOCR
        # use_textline_orientation=True helps with rotated text
        # lang='en' for English (CAD drawings are typically in English)
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='en'
        )
        
        if self.verbose:
            print("[OCR] PaddleOCR loaded successfully")
    
    def _extract_text_from_image(self, pil_image: Image.Image) -> List[Tuple[str, float]]:
        """
        Extract all text from image using PaddleOCR.
        Returns list of (text, confidence) tuples.
        """
        # Convert PIL image to file-like object or save temporarily
        import numpy as np
        img_array = np.array(pil_image)

        if self.verbose:
            print(f"[OCR] Processing image: {pil_image.size}")

        # Run OCR
        result = self.ocr.ocr(img_array)

        if not result or len(result) == 0:
            if self.verbose:
                print("[OCR] No text detected")
            return []

        # --- Normalize boxes/texts/scores for downstream consumers ---
        self._norm_boxes = []
        self._norm_texts = []
        self._norm_scores = []

        ocr_result = result[0]
        self._last_ocr_result = ocr_result
        self._last_ocr_image = pil_image

        # Handle dict-like (new) format
        if isinstance(ocr_result, dict):
            if self.verbose:
                print(f"[OCR] Using dict-like OCRResult format")
                print(f"[OCR] OCRResult keys: {list(ocr_result.keys()) if hasattr(ocr_result, 'keys') else 'N/A'}")

            # Prefer boxes already in original-image coordinates
            # rec_boxes/rec_polys are recognition-time boxes mapped to original coordinates
            # dt_polys are detection-time boxes which may be in preprocessed coordinate space
            if 'rec_boxes' in ocr_result and 'rec_texts' in ocr_result:
                self._norm_boxes = ocr_result['rec_boxes'] if ocr_result['rec_boxes'] is not None else []
                self._norm_texts = ocr_result['rec_texts'] if ocr_result['rec_texts'] is not None else []
                self._norm_scores = ocr_result.get('rec_scores', [1.0]*len(self._norm_texts))
                if self.verbose:
                    print(f"[OCR] Using rec_boxes (recognition-time boxes)")
            elif 'rec_polys' in ocr_result and 'rec_texts' in ocr_result:
                self._norm_boxes = ocr_result['rec_polys'] if ocr_result['rec_polys'] is not None else []
                self._norm_texts = ocr_result['rec_texts'] if ocr_result['rec_texts'] is not None else []
                self._norm_scores = ocr_result.get('rec_scores', [1.0]*len(self._norm_texts))
                if self.verbose:
                    print(f"[OCR] Using rec_polys (recognition-time polygons)")
            elif 'dt_polys' in ocr_result and 'rec_texts' in ocr_result:
                # Fallback (may be misaligned if preprocessor resized/padded)
                self._norm_boxes = ocr_result['dt_polys'] if ocr_result['dt_polys'] is not None else []
                self._norm_texts = ocr_result['rec_texts'] if ocr_result['rec_texts'] is not None else []
                self._norm_scores = ocr_result.get('rec_scores', [1.0]*len(self._norm_texts))
                if self.verbose:
                    print(f"[OCR] Using dt_polys (detection-time boxes - may be misaligned)")
            else:
                # Fallback: try to iterate generic dict entries
                for item in ocr_result.get('lines', []):
                    box = item.get('points') or item.get('box')
                    txt = item.get('text') or item.get('rec_text', '')
                    sc  = item.get('score') or item.get('rec_score', 0.9)
                    if box and txt:
                        self._norm_boxes.append(box)
                        self._norm_texts.append(txt)
                        self._norm_scores.append(float(sc))

        # Handle classic list format: [[box, (text, score)], ...] or [box, text, score]
        elif isinstance(ocr_result, list):
            if self.verbose:
                print(f"[OCR] Using classic list format")
            for line in ocr_result:
                try:
                    box = line[0]
                    if isinstance(line[1], tuple) and len(line[1]) == 2:
                        txt, sc = line[1]
                    else:
                        txt = line[1]
                        sc  = float(line[2]) if len(line) >= 3 else 0.9
                    self._norm_boxes.append(box)
                    self._norm_texts.append(txt)
                    self._norm_scores.append(float(sc))
                except Exception:
                    if self.verbose:
                        print(f"[OCR] Skipping malformed result row: {line}")
        else:
            if self.verbose:
                print(f"[OCR] Unknown OCRResult type: {type(ocr_result)}")

        # Build the (text,score) list to return
        text_list = list(zip(self._norm_texts, self._norm_scores))

        # Save raw detections for downstream export
        self._all_ocr_detections = []
        for box, text, score in zip(self._norm_boxes, self._norm_texts, self._norm_scores):
            box_payload = box.tolist() if hasattr(box, "tolist") else box
            self._all_ocr_detections.append({
                "box": box_payload,
                "text": text,
                "score": float(score)
            })
        self._all_ocr_detections_vized = []

        if self.verbose:
            print(f"[OCR] Detected {len(text_list)} text regions")
            for text, conf in text_list[:10]:  # Show first 10
                print(f"  '{text}' (conf: {conf:.3f})")
            # Store text list for later debug file saving
            self._last_text_list = text_list

        return text_list
    
    def _is_in_table_region(self, box, table_regions: List[Tuple[int, int, int, int]]) -> bool:
        """
        Check if a text bounding box overlaps with any table region.
        """
        import numpy as np

        if not table_regions:
            return False

        # Get center point of the box - handle different formats
        box_array = np.array(box)
        if box_array.shape == (4,):
            # Bounding box format: [xmin, ymin, xmax, ymax]
            center_x = (box_array[0] + box_array[2]) / 2
            center_y = (box_array[1] + box_array[3]) / 2
        else:
            # Polygon format: [[x1,y1], [x2,y2], ...]
            center_x = np.mean(box_array[:, 0])
            center_y = np.mean(box_array[:, 1])

        # Check if center is in any table region
        for table_x, table_y, table_w, table_h in table_regions:
            if (table_x <= center_x <= table_x + table_w and
                table_y <= center_y <= table_y + table_h):
                return True

        return False

    def _extract_numbers_from_text(self, text_list: List[Tuple[str, float]], table_regions: Optional[List[Tuple[int, int, int, int]]] = None) -> List[float]:
        """
        Extract all numbers from OCR text results.
        Handles various formats: decimals, fractions, dimensions with symbols.
        Filters out detail callouts (numbers with letter prefixes like D51, H03, etc.)
        Also filters out numbers from detected table regions.
        """
        import numpy as np
        numbers = []

        # Track spatial locations of key numbers for debugging
        key_number_locations = {}

        # Track bounding box coordinates for all numbers (for debug output)
        # Maps number -> (xmin_top_left, ymin_top_left)
        self._number_to_bbox = {}

        # Get bounding boxes from normalized data
        boxes = getattr(self, "_norm_boxes", None)

        for idx, (text, confidence) in enumerate(text_list):
            # Check if this text is in a table region
            if boxes is not None and table_regions and idx < len(boxes):
                if self._is_in_table_region(boxes[idx], table_regions):
                    if self.verbose:
                        print(f"[OCR] Skipping text in table region: '{text}'")
                    continue

            # Skip low confidence text
            if confidence < 0.5:
                continue

            # Skip detail callouts and reference numbers (letter followed by number)
            # Examples: D51, H03, A60, R12, N31, K75, _H23, )H25, 0H20, etc.
            # Use search instead of match to find pattern anywhere in text (handles OCR errors with leading chars)
            if re.search(r'[A-Z]\d{1,3}(?:\D|$)', text.strip()):
                # But allow if it's clearly a dimension with decimal (e.g., "25.0" not "H25")
                if not re.search(r'\d+\.\d+', text):
                    if self.verbose:
                        print(f"[OCR] Skipping detail callout: '{text}'")
                    continue

            # Skip threading/hole specifications (TAP, THRU, DEEP, etc.)
            if re.search(r'\b(TAP|THRU|DEEP|DRILL)\b', text, re.IGNORECASE):
                if self.verbose:
                    print(f"[OCR] Skipping hole/thread spec: '{text}'")
                continue

            # Skip coordinate table entries and headers
            if re.search(r'\b(HOLE|COORDINATE|LIST)\b', text, re.IGNORECASE):
                if self.verbose:
                    print(f"[OCR] Skipping coordinate table text: '{text}'")
                continue

            # Skip dimensions with lowercase letter suffixes (e.g., "13.00 q", "3.75 q")
            # These are often detail/feature dimensions, not overall part dimensions
            if re.search(r'\d+\.?\d*\s+[a-z]', text):
                if self.verbose:
                    print(f"[OCR] Skipping detail dimension: '{text}'")
                continue

            # Skip compound reference callouts (e.g., "N31 ON32", "N40 N38", "F9 H99")
            # These contain reference numbers mixed with text
            if re.search(r'[A-Z]\d+', text) and re.search(r'[A-Z]{2,}|O[A-Z]', text):
                if self.verbose:
                    print(f"[OCR] Skipping compound callout: '{text}'")
                continue

            # Skip if it looks like a table cell or list item (e.g., "1.", "2.", etc.)
            if re.match(r'^\d+\.$', text.strip()):
                continue

            # Skip text containing feet/inches markers (title block dimensions)
            # OCR often misreads tolerance notation "-.01" as "10'-"
            if "'" in text or '"' in text:
                if self.verbose:
                    print(f"[OCR] Skipping feet/inches notation: '{text}'")
                continue

            # Skip tolerance notation patterns (e.g., "-.01", "+.00", "±.005")
            # These start with +/- and have only decimal values
            if re.match(r'^[+\-±]\s*\.?\d+', text.strip()):
                if self.verbose:
                    print(f"[OCR] Skipping tolerance notation: '{text}'")
                continue

            # Remove common non-numeric symbols BEFORE checking letter/digit ratio
            text = text.replace('Ø', '')  # Diameter symbol
            text = text.replace('°', '')  # Degree symbol
            text = text.replace('R', '')  # Radius (remove R first so "OVER R" becomes "OVER")
            text = text.replace('X', ' ')  # X separator (e.g., "12X4" -> "12 4")
            text = text.replace('x', ' ')
            text = text.replace('OVER', '')  # Common suffix for tolerances (e.g., ".7811 OVER R")
            text = text.strip()

            # Skip text that's mostly letters (likely labels, not dimensions)
            # Now this check happens AFTER removing symbols
            letter_count = sum(1 for c in text if c.isalpha())
            digit_count = sum(1 for c in text if c.isdigit())
            if letter_count > digit_count:  # More letters than digits
                continue

            # Extract decimal numbers (including leading/trailing decimals)
            # Pattern matches: 12, 12.5, .5, 0.5, etc.
            number_pattern = r'\d+\.?\d*|\.\d+'
            matches = re.finditer(number_pattern, text)

            for match in matches:
                try:
                    num = float(match.group())
                    # Filter out obviously wrong values
                    if 0.01 <= num <= 1000:  # Reasonable range for dimensions
                        numbers.append(num)
                        if self.verbose and 0.4 <= num <= 0.6:
                            print(f"[OCR] Extracted thickness-range number {num} from '{text}'")

                        # Store bounding box coordinates (top-left corner) for this number
                        if boxes is not None and idx < len(boxes):
                            box_array = np.array(boxes[idx])
                            # Handle different box formats
                            if box_array.shape == (4,):
                                # Bounding box format: [xmin, ymin, xmax, ymax]
                                xmin_top_left = int(box_array[0])
                                ymin_top_left = int(box_array[1])
                                center_x = int((box_array[0] + box_array[2]) / 2)
                                center_y = int((box_array[1] + box_array[3]) / 2)
                            else:
                                # Polygon format: [[x1,y1], [x2,y2], ...]
                                xmin_top_left = int(np.min(box_array[:, 0]))
                                ymin_top_left = int(np.min(box_array[:, 1]))
                                center_x = int(np.mean(box_array[:, 0]))
                                center_y = int(np.mean(box_array[:, 1]))

                            # Store the first occurrence if multiple instances of same number
                            if num not in self._number_to_bbox:
                                self._number_to_bbox[num] = (xmin_top_left, ymin_top_left)

                            # Track spatial location of key numbers
                            if num in [11.5, 19.0, 20.0, 25.0, 1.125]:
                                key_number_locations[num] = (center_x, center_y, text)
                except ValueError:
                    continue

        if self.verbose:
            print(f"[OCR] Extracted {len(numbers)} numbers: {sorted(set(numbers), reverse=True)[:20]}")

            # Log spatial locations of key numbers
            if key_number_locations:
                print(f"\n[SPATIAL] Key number locations:")
                for num in sorted(key_number_locations.keys(), reverse=True):
                    x, y, orig_text = key_number_locations[num]
                    print(f"  {num:6.3f} at ({x:4d}, {y:4d}) from text '{orig_text}'")
                print()

        return numbers
    
    def _filter_dimension_candidates(self, numbers: List[float]) -> Dict[str, List[float]]:
        """
        Separate numbers into likely thickness, length/width, and other categories.
        Filters out part numbers and other non-dimension numbers.
        """
        # Remove duplicates and sort
        unique_numbers = sorted(set(numbers), reverse=True)

        # Filter out common angles that might be misread as dimensions
        common_angles = {0, 15, 30, 45, 60, 90, 120, 135, 180, 270, 360}
        filtered = [n for n in unique_numbers if n not in common_angles]

        # Filter out large integers that are likely part numbers (50-200 range)
        # Part numbers in tables are typically whole numbers > 50
        filtered = [n for n in filtered if not (n >= 40.0 and n == int(n))]

        # Categorize by typical dimension ranges
        thickness_candidates = [n for n in filtered if 0.05 <= n <= 6.0]

        # Planar dimensions: 0.2" to 50"
        # Lowered minimum from 1.0 to 0.2 to support small parts like 0.665 x 0.2758
        planar_candidates = [n for n in filtered if 0.2 <= n <= 50.0]

        # Remove only very small thickness values from planar candidates
        # Values >= 0.2 could be width for small parts, so keep them in planar
        # Only remove values < 0.15 (typical drill/hole sizes) from planar candidates
        very_small_thickness = [t for t in thickness_candidates if t < 0.15]
        planar_candidates = [n for n in planar_candidates if n not in very_small_thickness]

        if self.verbose:
            print(f"[OCR] Thickness candidates ({len(thickness_candidates)}): {thickness_candidates[:10]}")
            print(f"[OCR] Planar candidates ({len(planar_candidates)}): {planar_candidates[:20]}")

        return {
            'thickness': thickness_candidates,
            'planar': planar_candidates,
            'all': filtered
        }
    
    def _select_dimensions(self, candidates: Dict[str, List[float]]) -> Optional[Tuple[float, float, float]]:
        """
        Select the most likely L, W, T from candidates using heuristics.
        Prioritizes common plate thicknesses and realistic stock dimensions.
        """
        thickness_candidates = candidates['thickness']
        planar_candidates = candidates['planar']

        if not thickness_candidates:
            if self.verbose:
                print("[OCR] No valid thickness candidates found")
            return None

        if len(planar_candidates) < 2:
            if self.verbose:
                print("[OCR] Insufficient planar dimension candidates")
            return None

        # STRATEGY: Select L/W first, then choose thickness that creates typical aspect ratios
        # This works better than choosing thickness first

        # Step 1: Select length and width from the two largest planar dimensions
        # Filter for reasonable stock dimensions
        valid_planar = [n for n in planar_candidates if n <= 50.0]

        if len(valid_planar) < 2:
            if self.verbose:
                print("[OCR] Insufficient planar candidates after filtering")
            return None

        if len(valid_planar) >= 2:
            # Pick the two largest values, strongly preferring "rounder" numbers
            # Overall part dimensions tend to be whole numbers or simple fractions
            def is_round_number(val: float) -> bool:
                decimal_part = val - int(val)
                return (decimal_part == 0.0 or  # Whole number
                       abs(decimal_part - 0.25) < 0.01 or
                       abs(decimal_part - 0.5) < 0.01 or
                       abs(decimal_part - 0.75) < 0.01 or
                       abs(decimal_part - 0.125) < 0.01 or
                       abs(decimal_part - 0.375) < 0.01 or
                       abs(decimal_part - 0.625) < 0.01 or
                       abs(decimal_part - 0.875) < 0.01)

            def roundness_score(val: float) -> float:
                """Return how 'round' a number is (lower = rounder)"""
                decimal_part = val - int(val)
                # Check distance to common fractions
                common_fractions = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
                min_dist = min(abs(decimal_part - frac) for frac in common_fractions)
                return min_dist

            # Strategy: Prefer round numbers from all candidates
            # Filter all planar candidates for round numbers
            round_candidates = [v for v in valid_planar if is_round_number(v)]

            sorted_all = sorted(valid_planar, reverse=True)

            # Strategy: Prefer round numbers, but use largest if significantly bigger
            # This handles cases where length is non-round (like 8.72) but width is round (2.5)

            if len(round_candidates) >= 2:
                # Check if largest overall value is much bigger than largest round value
                largest_overall = sorted_all[0]
                largest_round = round_candidates[0]

                # If largest is 5%+ bigger than largest round, it's probably the actual length
                if largest_overall > largest_round * 1.05:
                    length = largest_overall
                    # Width is largest round number
                    width = largest_round
                    if self.verbose:
                        print(f"[OCR] Length (largest, non-round): {length}")
                        print(f"[OCR] Width (largest round): {width}")
                else:
                    # Use top 2 round numbers, prioritizing exactness then size
                    # Sort by: roundness (more exact first), then by value (larger first)
                    round_sorted = sorted(round_candidates, key=lambda v: (roundness_score(v), -v))
                    length = round_sorted[0]
                    width = round_sorted[1]
                    if self.verbose:
                        print(f"[OCR] Length (largest round): {length}")
                        print(f"[OCR] Width (2nd largest round): {width}")
            else:
                # Fallback: use largest 2 values
                length = sorted_all[0]
                width = sorted_all[1] if len(sorted_all) > 1 else sorted_all[0]
                if self.verbose:
                    print(f"[OCR] Fallback - Length: {length}, Width: {width}")

            # Step 2: Now select thickness based on L/W
            # IMPORTANT: Exclude values already used for length/width from thickness selection
            available_thickness = [t for t in thickness_candidates if t not in {length, width}]

            if not available_thickness:
                # Edge case: all thickness values were used for L/W, use original list
                available_thickness = thickness_candidates

            # Filter thickness candidates to reasonable range: typically W/20 to W/5
            # This handles typical plate aspect ratios of 5:1 to 20:1
            smaller_dim = min(length, width)
            min_reasonable_t = smaller_dim / 20.0
            max_reasonable_t = smaller_dim / 5.0

            reasonable_thickness = [t for t in available_thickness
                                   if min_reasonable_t <= t <= max_reasonable_t]

            # If filtering is too strict, fall back to available candidates (excluding L/W)
            if not reasonable_thickness:
                reasonable_thickness = available_thickness

            if self.verbose:
                print(f"[OCR] Reasonable thickness range: {min_reasonable_t:.3f} to {max_reasonable_t:.3f}")
                print(f"[OCR] Reasonable thickness candidates: {reasonable_thickness[:10]}")

            # Choose standard thickness, preferring larger values in the reasonable range
            def thickness_priority(t: float) -> Tuple[float, float]:
                common_thicknesses = [0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.125, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]

                # Calculate "standardness" - how close to a standard thickness (lower is better)
                min_dist = min(abs(t - ct) for ct in common_thicknesses)

                # Priority:
                # 1. Must be standard or very close (exact match has min_dist=0)
                # 2. Among equally standard values, prefer LARGER thickness
                return (min_dist, -t)

            thickness = min(reasonable_thickness, key=thickness_priority)

            if self.verbose:
                print(f"[OCR] Selected dimensions:")
                print(f"  Length: {length}")
                # Print bounding box for length
                if hasattr(self, '_number_to_bbox') and length in self._number_to_bbox:
                    xmin, ymin = self._number_to_bbox[length]
                    print(f"    Bounding Box Top-Left: ({xmin}, {ymin})")
                print(f"  Width: {width}")
                # Print bounding box for width
                if hasattr(self, '_number_to_bbox') and width in self._number_to_bbox:
                    xmin, ymin = self._number_to_bbox[width]
                    print(f"    Bounding Box Top-Left: ({xmin}, {ymin})")
                print(f"  Thickness: {thickness}")
                # Print bounding box for thickness
                if hasattr(self, '_number_to_bbox') and thickness in self._number_to_bbox:
                    xmin, ymin = self._number_to_bbox[thickness]
                    print(f"    Bounding Box Top-Left: ({xmin}, {ymin})")

            return length, width, thickness

        return None
    
    def _validate_dimensions(self, L: float, W: float, T: float) -> bool:
        """Validate that dimensions are physically plausible.

        Note: This validation is relaxed to support both plate and block geometries.
        For blocks, thickness can be >= width (e.g., 1.148 x 0.445 x 2.0).
        """
        if any(v is None or v <= 0 for v in [L, W, T]):
            return False

        # Thickness should be reasonable
        if not (0.05 <= T <= 6.0):
            if self.verbose:
                print(f"[OCR] Validation failed: thickness {T} out of range [0.05, 6.0]")
            return False

        # Relaxed validation: For plates, L/W >> T. For blocks, T can be >= W.
        # Just ensure that the three dimensions aren't all identical (that would be weird)
        if L == W == T:
            if self.verbose:
                print(f"[OCR] Validation failed: all dimensions are identical")
            return False

        # Dimensions should be reasonable in general
        if max(L, W, T) > 500:
            if self.verbose:
                print(f"[OCR] Validation failed: dimensions too large (>500)")
            return False

        return True

    def _fix_box_coords_for_pil(self, box, img_height: int, assume_top_left: Optional[bool] = None):
        """
        Ensure box coords are in PIL's top-left origin system.
        If assume_top_left is None, auto-detect: if most y's are in the bottom
        40% of the image, treat coords as bottom-left origin and flip.
        """
        import numpy as np
        box_array = np.array(box)

        # Normalize to polygon points
        if box_array.shape == (4,):
            # [xmin, ymin, xmax, ymax]
            pts = [(float(box[0]), float(box[1])),
                   (float(box[2]), float(box[1])),
                   (float(box[2]), float(box[3])),
                   (float(box[0]), float(box[3]))]
        else:
            pts = [(float(p[0]), float(p[1])) for p in box]

        ys = [p[1] for p in pts]

        # Auto-detect if not forced: "too many" y near the bottom
        flip_needed = False
        if assume_top_left is None:
            # share of points in bottom 40% of the image
            bottom_share = sum(y >= 0.6 * img_height for y in ys) / max(1, len(ys))
            # and in top 40%
            top_share = sum(y <= 0.4 * img_height for y in ys) / max(1, len(ys))
            # If we overwhelmingly sit near the bottom, flip
            flip_needed = (bottom_share > 0.7 and top_share < 0.3)
        else:
            flip_needed = (assume_top_left is False)

        if flip_needed:
            fixed = [(int(x), int(img_height - y)) for x, y in pts]
        else:
            fixed = [(int(x), int(y)) for x, y in pts]

        return fixed

    def _rotate_points(self, pts, w, h, mode: str):
        """
        Rotate polygon points by 90/180/270 degrees within a w×h canvas.
        mode in {"cw90","ccw90","180"}.
        Returns integer point list.
        """
        if mode == "cw90":      # clockwise 90
            return [(int(h - y - 1), int(x)) for (x, y) in pts]
        elif mode == "ccw90":   # counter-clockwise 90
            return [(int(y), int(w - x - 1)) for (x, y) in pts]
        elif mode == "180":
            return [(int(w - x - 1), int(h - y - 1)) for (x, y) in pts]
        return [(int(x), int(y)) for (x, y) in pts]

    def _to_poly(self, box):
        """
        Normalize a rectangle or polygon representation to a list of (x, y) points.
        """
        try:
            import numpy as np
        except ImportError:
            np = None

        if np is not None:
            if isinstance(box, np.ndarray):
                arr = box
            else:
                try:
                    arr = np.array(box, dtype=float)
                except Exception:
                    arr = None
        else:
            arr = None

        if arr is not None:
            if arr.ndim == 1 and arr.size == 4:
                x0, y0, x1, y1 = map(float, arr.tolist())
                return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return [(float(x), float(y)) for x, y in arr[:, :2]]

        if isinstance(box, (list, tuple)) and len(box) == 4 and isinstance(box[0], (int, float)):
            return [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]

        if isinstance(box, (list, tuple)) and len(box) >= 4 and isinstance(box[0], (list, tuple)):
            return [(float(p[0]), float(p[1])) for p in box]

        return None

    def _poly_top_left(self, pts):
        """Return the integer top-left corner from a list of points."""
        if not pts:
            return (0, 0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (int(min(xs)), int(min(ys)))

    def _roi_offset_from_doc_preproc(self, ocr_result):
        offx, offy = 0, 0
        pre = ocr_result.get("doc_preprocessor_res", {}) or {}

        # Try common fields used by Paddle doc preprocessor variants
        # Each gives a top-left (x0, y0) style offset for the cropped ROI.
        candidates = []

        # Rect-style boxes
        for k in ("roi_box", "crop_box", "doc_bbox", "page_box", "bbox"):
            v = pre.get(k)
            if isinstance(v, (list, tuple)) and len(v) == 4:
                # [x0, y0, x1, y1]
                candidates.append((int(v[0]), int(v[1])))

        # Polygon-style boxes
        for k in ("doc_points", "doc_polygon", "page_polygon", "roi_polygon"):
            v = pre.get(k)
            if isinstance(v, (list, tuple)) and len(v) >= 4 and isinstance(v[0], (list, tuple)):
                xs = [p[0] for p in v]
                ys = [p[1] for p in v]
                candidates.append((int(min(xs)), int(min(ys))))

        if candidates:
            # Pick the most conservative (largest) top-left; avoids under-correcting
            offx = max(c[0] for c in candidates)
            offy = max(c[1] for c in candidates)

        if self.verbose:
            print(f"[DEBUG] ROI offset guess from doc_preprocessor_res: ({offx}, {offy}) -- keys: {list(pre.keys())}")
        return offx, offy
    
    def _visualize_ocr_boxes(self, save_path: str, table_regions: Optional[List[Tuple[int, int, int, int]]] = None):
        """
        Visualize OCR detected text boxes on the image for debugging.
        Draws bounding boxes with detected text labels.
        Also visualizes detected table regions if provided.
        """
        if not hasattr(self, '_last_ocr_result') or not hasattr(self, '_last_ocr_image'):
            print("[DEBUG] No OCR result to visualize")
            return

        from PIL import Image, ImageDraw, ImageFont

        orc = getattr(self, "_last_ocr_result", {}) or {}
        pre = orc.get("doc_preprocessor_res", {}) or {}

        use_pre = pre.get("output_img", None) is not None
        rotate_mode = None
        pre_w = pre_h = None

        if use_pre:
            import numpy as np
            arr = pre["output_img"]
            base_img = Image.fromarray(arr.astype("uint8")).convert("RGB")
            pre_w, pre_h = base_img.size
            if pre_h > pre_w:
                rotate_mode = "cw90"
                base_img = base_img.transpose(Image.ROTATE_270)
            if self.verbose:
                rot_msg = f", rotated {rotate_mode}" if rotate_mode else ""
                print(f"[DEBUG] Drawing on doc_preprocessor_res output_img{rot_msg}")
        else:
            base_img = self._last_ocr_image.copy()

        draw = ImageDraw.Draw(base_img)
        img_w, img_h = base_img.size

        # Canary overlay to confirm drawing happened
        draw.rectangle([5, 5, 45, 45], outline="red", width=5)
        draw.line([(0, 0), (min(200, img_w - 1), min(200, img_h - 1))], fill="red", width=5)

        # First draw table regions in semi-transparent red rectangles
        if table_regions:
            print(f"[DEBUG] Drawing {len(table_regions)} table regions...")
            for table_x, table_y, table_w, table_h in table_regions:
                # Draw filled semi-transparent rectangle
                rect_points = [
                    (table_x, table_y),
                    (table_x + table_w, table_y),
                    (table_x + table_w, table_y + table_h),
                    (table_x, table_y + table_h)
                ]
                draw.polygon(rect_points, outline='red', width=5)
                # Add label
                try:
                    font = ImageFont.truetype("arial.ttf", 30)
                except Exception:
                    font = ImageFont.load_default()
                draw.text((table_x + 10, table_y + 10), "TABLE REGION", fill='red', font=font)

        # Choose boxes/texts (prefer recognition outputs)
        boxes: List[Any] = []
        texts: List[str] = []
        scores: Optional[Any] = None
        if isinstance(orc, dict):
            if "rec_boxes" in orc and "rec_texts" in orc:
                boxes = orc.get("rec_boxes", [])
                texts = orc.get("rec_texts", [])
                scores = orc.get("rec_scores")
            elif "rec_polys" in orc and "rec_texts" in orc:
                boxes = orc.get("rec_polys", [])
                texts = orc.get("rec_texts", [])
                scores = orc.get("rec_scores")
            else:
                boxes = orc.get("dt_polys", [])
                texts = orc.get("rec_texts", [])
                scores = orc.get("rec_scores")

        # Fallback to normalized data if needed
        if boxes is None or texts is None or len(boxes) == 0 or len(texts) == 0:
            boxes = getattr(self, "_norm_boxes", [])
            texts = getattr(self, "_norm_texts", [])
            scores = getattr(self, "_norm_scores", None)

        if boxes is None or texts is None or len(boxes) == 0 or len(texts) == 0:
            print("[DEBUG] No boxes/texts available to draw")
            base_img.save(save_path)
            print(f"[DEBUG] Saved OCR visualization to: {save_path}")
            return

        try:
            boxes = list(boxes)
        except TypeError:
            boxes = [boxes]

        try:
            texts = list(texts)
        except TypeError:
            texts = [texts]

        if scores is None:
            score_values = [1.0] * len(texts)
        else:
            try:
                score_values = [float(s) for s in list(scores)]
            except TypeError:
                score_values = [float(scores)]

        if len(score_values) < len(texts):
            score_values.extend([1.0] * (len(texts) - len(score_values)))
        elif len(score_values) > len(texts):
            score_values = score_values[:len(texts)]

        offx, offy = (0, 0)
        if not use_pre:
            offx, offy = self._roi_offset_from_doc_preproc(orc if isinstance(orc, dict) else {})

        print(f"[DEBUG] Drawing {len(boxes)} detected text boxes...")
        print(f"[DEBUG] Image size for visualization: {base_img.size}")

        self._all_ocr_detections_vized = []

        for idx, box in enumerate(boxes):
            if idx >= len(texts):
                break
            text = texts[idx]
            score = score_values[idx] if idx < len(score_values) else 1.0

            pts = self._to_poly(box)
            if not pts:
                continue

            if use_pre:
                if rotate_mode:
                    pts = self._rotate_points(pts, pre_w, pre_h, rotate_mode)
            else:
                pts = self._fix_box_coords_for_pil(pts, img_h)
                pts = [(x + offx, y + offy) for (x, y) in pts]

            pts = [(int(x), int(y)) for x, y in pts]
            pts = [
                (
                    min(max(0, x), img_w - 1),
                    min(max(0, y), img_h - 1)
                )
                for x, y in pts
            ]

            color = 'green' if score > 0.8 else ('yellow' if score > 0.5 else 'red')
            draw.polygon(pts, outline=color, width=4)

            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except Exception:
                font = ImageFont.load_default()

            tl = self._poly_top_left(pts)
            self._all_ocr_detections_vized.append({
                "text": text,
                "score": float(score),
                "poly": [[int(x), int(y)] for x, y in pts],
                "top_left": [tl[0], tl[1]],
            })

            if idx < 3 and self.verbose:
                if use_pre:
                    print(f"[DEBUG] Box {idx}: first_pt={pts[0]} (preprocessor image)")
                else:
                    print(f"[DEBUG] Box {idx}: first_pt={pts[0]} (after y-fix + offset {offx},{offy})")

            lbl_pos = (pts[0][0], max(0, pts[0][1] - 25))
            draw.text(lbl_pos, f"{text} ({score:.2f})", fill=color, font=font)

        base_img.save(save_path)
        print(f"[DEBUG] Saved OCR visualization to: {save_path}")

    def _detect_table_regions(self, pil_image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Detect table regions in the image using spatial analysis of detected text.
        Returns list of (x, y, width, height) bounding boxes for detected tables.
        """
        import numpy as np

        # Use normalized boxes and texts
        boxes = getattr(self, "_norm_boxes", [])
        texts = getattr(self, "_norm_texts", [])

        if len(boxes) == 0 or len(texts) == 0:
            return []

        # Identify table regions by looking for table indicator text
        table_indicators = []
        for box, text in zip(boxes, texts):
            # Check if text indicates a table
            if any(keyword in text.upper() for keyword in ['HOLE', 'COORDINATE', 'LIST', 'TABLE']):
                # Get bounding box - handle different formats
                box_array = np.array(box)
                if box_array.shape == (4,):
                    # Bounding box format: [xmin, ymin, xmax, ymax]
                    min_x = int(box_array[0])
                    min_y = int(box_array[1])
                    max_x = int(box_array[2])
                    max_y = int(box_array[3])
                else:
                    # Polygon format: [[x1,y1], [x2,y2], ...]
                    min_x = int(np.min(box_array[:, 0]))
                    min_y = int(np.min(box_array[:, 1]))
                    max_x = int(np.max(box_array[:, 0]))
                    max_y = int(np.max(box_array[:, 1]))
                table_indicators.append((min_x, min_y, max_x, max_y))

        if not table_indicators:
            return []

        # For each table indicator, define the table region
        # Tables are typically on the right side and extend vertically
        table_regions = []
        width, height = pil_image.size

        for min_x, min_y, max_x, max_y in table_indicators:
            # Assume table extends from indicator to right edge
            # and vertically from top of indicator to bottom
            table_left = max(0, min_x - 50)  # Small margin
            table_right = width
            table_top = max(0, min_y - 50)
            table_bottom = height  # Extend to bottom

            table_regions.append((table_left, table_top, table_right - table_left, table_bottom - table_top))

            if self.verbose:
                print(f"[OCR] Detected table region at x={table_left}, y={table_top}, w={table_right - table_left}, h={table_bottom - table_top}")

        return table_regions

    def _crop_to_main_drawing(self, pil_image: Image.Image) -> Image.Image:
        """
        No cropping - use full image.
        Table regions will be filtered out using spatial analysis of OCR results.
        """
        if self.verbose:
            width, height = pil_image.size
            print(f"[OCR] Using full image: {width}x{height} (no cropping)")

        return pil_image
    
    def extract(self, pil_image: Image.Image, save_cropped_path: Optional[str] = None) -> Optional[Dimensions]:
        """Extract dimensions from a rendered CAD drawing image."""
        if self.verbose:
            print("[OCR] Starting dimension extraction...")

        # Crop to main drawing area
        cropped_image = self._crop_to_main_drawing(pil_image)

        # Optionally save cropped image for debugging
        if save_cropped_path and self.verbose:
            cropped_image.save(save_cropped_path)
            if self.verbose:
                print(f"[OCR] Saved cropped image to: {save_cropped_path}")
        
        # Extract all text using PaddleOCR
        text_list = self._extract_text_from_image(cropped_image)

        # Detect table regions using OCR results
        table_regions = self._detect_table_regions(cropped_image)

        # Save OCR visualization if in verbose mode
        if save_cropped_path and self.verbose:
            ocr_viz_path = save_cropped_path.replace('_cropped.png', '_ocr_boxes.png')
            self._visualize_ocr_boxes(ocr_viz_path, table_regions=table_regions)

            # Save all detected text to debug file
            if hasattr(self, '_last_text_list'):
                debug_text_path = save_cropped_path.replace('_cropped.png', '_all_text.txt')
                with open(debug_text_path, 'w', encoding='utf-8') as f:
                    for i, (text, conf) in enumerate(self._last_text_list):
                        f.write(f"{i+1:3d}. '{text}' (conf: {conf:.3f})\n")
                print(f"[DEBUG] Saved all {len(self._last_text_list)} detected texts to: {debug_text_path}")

        if not text_list:
            if self.verbose:
                print("[OCR] No text detected in image")
            return None

        # Extract numbers from text, excluding numbers in table regions
        numbers = self._extract_numbers_from_text(text_list, table_regions=table_regions)
        
        if not numbers:
            if self.verbose:
                print("[OCR] No numbers found in extracted text")
            return None
        
        # Categorize and filter candidates
        candidates = self._filter_dimension_candidates(numbers)
        
        # Select best dimensions
        dimensions = self._select_dimensions(candidates)
        
        if not dimensions:
            if self.verbose:
                print("[OCR] Failed to infer dimensions from numbers")
            return None
        
        L, W, T = dimensions
        
        # Validate
        if not self._validate_dimensions(L, W, T):
            if self.verbose:
                print("[OCR] Dimensions failed validation")
            return None
        
        if self.verbose:
            print(f"[OCR] Successfully extracted: {L} x {W} x {T} in")
        
        return Dimensions(
            length=L,
            width=W,
            thickness=T,
            units="in",
            confidence="high",
            method="paddleocr",
            all_numbers=sorted(set(numbers), reverse=True)
        )


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
    
    def _dxf_to_png(self, dxf_path: str, png_path: str, dpi: int = 800, max_dimension: int = 3000):
        """Render DXF to high-contrast PNG optimized for OCR.

        max_dimension set to 3000 to prevent PaddleOCR segmentation faults.
        Some drawings cause segfaults at higher resolutions.
        """
        if self.verbose:
            print(f"[DXF] Rendering to PNG (DPI={dpi})...")

        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        # High-contrast configuration optimized for OCR
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
        description="Extract part dimensions from CAD drawings using PaddleOCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paddle_dims_extractor.py --input drawing.dwg
  python paddle_dims_extractor.py --input drawing.dxf --verbose
  python paddle_dims_extractor.py --input drawing.png --output-json results.json

Installation:
  pip install paddlepaddle paddleocr pillow ezdxf matplotlib
"""
    )

    # Input/output
    parser.add_argument("--input", required=True, help="Path to CAD file (DWG/DXF/PNG)")
    parser.add_argument("--output-json", help=f"Output JSON file path (default: {DEFAULT_OUTPUT_DIR}/<input>_dims.json)")
    parser.add_argument("--output-png", help=f"Save rendered PNG (default: {DEFAULT_OUTPUT_DIR}/<input>_render.png)")

    # ODA converter
    parser.add_argument("--oda-exe", help=f"Path to ODA File Converter executable (default: {DEFAULT_ODA_EXE})")

    # Options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Determine output paths
    input_path = Path(args.input)

    # Create output directory if it doesn't exist
    Path(DEFAULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if args.output_json:
        json_path = args.output_json
    else:
        json_path = str(Path(DEFAULT_OUTPUT_DIR) / f"{input_path.stem}_dims.json")

    if args.output_png:
        png_path = args.output_png
    else:
        png_path = str(Path(DEFAULT_OUTPUT_DIR) / f"{input_path.stem}_render.png")

    # Get ODA exe path (priority: CLI args > default)
    oda_exe = args.oda_exe or DEFAULT_ODA_EXE
    
    try:
        # Step 1: Render drawing to PNG
        print(f"[1/3] Rendering: {input_path.name}")
        renderer = DrawingRenderer(oda_exe=oda_exe, verbose=args.verbose)
        renderer.render(str(input_path), png_path)
        
        # Step 2: Load OCR
        print(f"[2/3] Loading PaddleOCR...")
        extractor = PaddleOCRDimensionExtractor(verbose=args.verbose)
        
        # Step 3: Extract dimensions
        print(f"[3/3] Extracting dimensions...")
        cropped_debug_path = str(Path(DEFAULT_OUTPUT_DIR) / f"{input_path.stem}_cropped.png")
        with Image.open(png_path) as img:
            img = img.convert("RGB")
            dims = extractor.extract(img, save_cropped_path=cropped_debug_path if args.verbose else None)
        
        if dims is None:
            print("\n[ERROR] Failed to extract dimensions", file=sys.stderr)
            print("\nTroubleshooting tips:", file=sys.stderr)
            print("  1. Check if the rendered PNG contains clear, readable text", file=sys.stderr)
            print("  2. Try increasing DPI in DrawingRenderer._dxf_to_png()", file=sys.stderr)
            print("  3. Manually verify dimensions in the PNG file", file=sys.stderr)
            sys.exit(1)
        
        # Build per-number entries with overlay coordinates
        import re

        number_rows: List[Dict[str, Any]] = []
        num_re = re.compile(r"[-+]?\d+(?:\.\d+)?")

        for det in getattr(extractor, "_all_ocr_detections_vized", []):
            text = det.get("text", "")
            matches = num_re.findall(text or "")
            if not matches:
                continue
            top_left = det.get("top_left", [0, 0])
            poly = det.get("poly", [])
            score = float(det.get("score", 0.0))
            for match in matches:
                try:
                    val = float(match)
                except ValueError:
                    continue
                number_rows.append({
                    "number": val,
                    "text": text,
                    "score": score,
                    "top_left": top_left,
                    "poly": poly
                })

        # Save results
        result = {
            "length": dims.length,
            "width": dims.width,
            "thickness": dims.thickness,
            "units": dims.units,
            "confidence": dims.confidence,
            "extraction_method": dims.method,
            "source_file": str(input_path),
            "rendered_image": png_path,
            "all_detected_numbers": number_rows
        }
        
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"[SUCCESS] Dimensions extracted successfully")
        print(f"{'='*60}")
        print(f"  Length:    {dims.length} {dims.units}")
        print(f"  Width:     {dims.width} {dims.units}")
        print(f"  Thickness: {dims.thickness} {dims.units}")
        print(f"  Method:    {dims.method}")
        print(f"  Confidence: {dims.confidence}")
        
        if args.verbose and dims.all_numbers:
            print(f"\nAll detected numbers (top 20):")
            for i, num in enumerate(dims.all_numbers[:20], 1):
                print(f"  {i:2d}. {num}")
        
        print(f"\n{'='*60}")
        print(f"[JSON] Saved to: {json_path}")
        print(f"[PNG]  Render:  {png_path}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
