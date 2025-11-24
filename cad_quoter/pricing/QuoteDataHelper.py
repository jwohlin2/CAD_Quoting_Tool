"""
QuoteDataHelper - Unified data structure for all CAD extraction results.

This module provides a centralized data structure that holds all extracted information
from DirectCostHelper and ProcessPlanner, making it easy to cache, serialize, and
pass around quote data throughout the application.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

# Punch extraction imports
try:
    from cad_quoter.geometry.dxf_enrich import (
        extract_punch_features_from_dxf,
        PunchFeatureSummary,
    )
    from cad_quoter.planning.process_planner import create_punch_plan
    from cad_quoter.pricing.time_estimator import estimate_punch_times
    PUNCH_EXTRACTION_AVAILABLE = True
except ImportError:
    PUNCH_EXTRACTION_AVAILABLE = False


@dataclass
class PartDimensions:
    """Part dimensions extracted from CAD file."""
    length: float = 0.0  # inches
    width: float = 0.0   # inches
    thickness: float = 0.0  # inches
    diameter: float = 0.0  # inches (for cylindrical parts - primary diameter)
    diameter_2: float = 0.0  # inches (secondary diameter for tapered parts)
    volume: float = 0.0  # cubic inches
    area: float = 0.0    # square inches (L × W)
    is_cylindrical: bool = False  # True for guide posts, spring pins, etc.

    def __post_init__(self):
        """Calculate derived properties if not set."""
        if self.volume == 0.0:
            if self.is_cylindrical and self.diameter > 0 and self.length > 0:
                # Volume = π * r² * length
                import math
                radius = self.diameter / 2.0
                self.volume = math.pi * radius * radius * self.length
            else:
                self.volume = self.length * self.width * self.thickness
        if self.area == 0.0:
            if self.is_cylindrical and self.diameter > 0:
                # Cross-sectional area = π * r²
                import math
                radius = self.diameter / 2.0
                self.area = math.pi * radius * radius
            else:
                self.area = self.length * self.width


@dataclass
class MaterialInfo:
    """Material information and properties."""
    material_name: str = "GENERIC"
    material_family: str = "aluminum"  # aluminum, steel, stainless, etc.
    density: float = 0.098  # lb/in³
    detected_from_cad: bool = False
    is_default: bool = False


@dataclass
class StockInfo:
    """McMaster stock information."""
    # Desired starting stock (part + allowances)
    desired_length: float = 0.0
    desired_width: float = 0.0
    desired_thickness: float = 0.0
    desired_diameter: float = 0.0  # For cylindrical parts
    desired_volume: float = 0.0

    # McMaster catalog stock
    mcmaster_length: float = 0.0
    mcmaster_width: float = 0.0
    mcmaster_thickness: float = 0.0
    mcmaster_diameter: float = 0.0  # For cylindrical parts
    mcmaster_volume: float = 0.0
    mcmaster_part_number: Optional[str] = None
    mcmaster_price: Optional[float] = None  # Unit price
    price_is_estimated: bool = False  # True if price was estimated from volume

    # Weights
    mcmaster_weight: float = 0.0  # lbs
    final_part_weight: float = 0.0  # lbs


# High scrap threshold (80%) - warn user when exceeded
HIGH_SCRAP_THRESHOLD = 80.0  # percent


@dataclass
class ScrapInfo:
    """Scrap calculation results."""
    # Scrap breakdown by source
    stock_prep_scrap: float = 0.0  # McMaster → Desired (in³)
    face_milling_scrap: float = 0.0  # Desired → Part envelope (in³)
    hole_drilling_scrap: float = 0.0  # Holes removed from part (in³)
    total_scrap_volume: float = 0.0  # Total material removed (in³)
    total_scrap_weight: float = 0.0  # lbs

    # Scrap percentages
    scrap_percentage: float = 0.0  # Scrap as % of McMaster stock
    utilization_percentage: float = 0.0  # Final part as % of McMaster stock

    # Scrap value
    scrap_price_per_lb: Optional[float] = None  # $/lb
    scrap_value: float = 0.0  # Total scrap value ($)
    scrap_price_source: str = ""  # e.g., "Wieland aluminum scrap"

    # High scrap warning
    high_scrap_warning: bool = False  # True if scrap_percentage > HIGH_SCRAP_THRESHOLD


@dataclass
class DirectCostBreakdown:
    """Direct cost breakdown."""
    stock_cost: float = 0.0  # McMaster stock price
    tax: float = 0.0  # Tax on stock
    shipping: float = 0.0  # Shipping on stock
    scrap_credit: float = 0.0  # Credit for scrap value (negative cost)
    net_material_cost: float = 0.0  # Total direct cost


@dataclass
class HoleOperation:
    """Single hole operation details."""
    hole_id: str = ""
    diameter: float = 0.0  # inches
    depth: float = 0.0  # inches
    qty: int = 1
    operation_type: str = ""  # drill, tap, cbore, cdrill, jig_grind
    description: str = ""

    # Time estimates
    time_per_hole: float = 0.0  # minutes
    total_time: float = 0.0  # minutes (qty × time_per_hole)

    # Operation-specific parameters
    sfm: Optional[float] = None  # Surface feet per minute
    ipr: Optional[float] = None  # Inches per revolution
    tpi: Optional[int] = None  # Threads per inch (for taps)
    side: Optional[str] = None  # For C'BORE: "front", "back", or None


@dataclass
class MillingOperation:
    """Detailed milling operation breakdown."""
    op_name: str = ""  # e.g., "square_up_rough_faces", "square_up_rough_sides"
    op_description: str = ""  # Human-readable description

    # Geometry
    length: float = 0.0  # inches
    width: float = 0.0  # inches
    thickness: float = 0.0  # inches (finished thickness)
    perimeter: float = 0.0  # inches (for side ops)

    # Stock dimensions (for square-up operations)
    stock_length: float = 0.0  # inches
    stock_width: float = 0.0  # inches
    stock_thickness: float = 0.0  # inches

    # Tool parameters
    tool_diameter: float = 0.0  # inches

    # Cut parameters
    passes: int = 1
    stepover: Optional[float] = None  # inches
    radial_stock: Optional[float] = None  # inches
    axial_step: Optional[float] = None  # inches
    axial_passes: Optional[int] = None
    radial_passes: Optional[int] = None

    # Path and feed
    path_length: float = 0.0  # total inches
    feed_rate: float = 0.0  # IPM

    # Volume removal (for grinding operations)
    volume_removed_cuin: float = 0.0  # cubic inches
    volume_thickness: float = 0.0  # cubic inches (volume from thickness removal)
    volume_length_trim: float = 0.0  # cubic inches (volume from length trimming)
    volume_width_trim: float = 0.0  # cubic inches (volume from width trimming)

    # Material factor (for square-up operations)
    material_factor: float = 1.0

    # Additional debug fields for square-up operations
    sq_top_bottom_stock: Optional[float] = None  # inches - stock removed from top/bottom
    surface_area_sq_in: Optional[float] = None  # square inches - surface area

    # Time
    time_minutes: float = 0.0

    # Override tracking
    _used_override: bool = False
    override_time_minutes: Optional[float] = None


@dataclass
class GrindingOperation:
    """Detailed grinding operation breakdown."""
    op_name: str = ""  # e.g., "wet_grind_square_all"
    op_description: str = ""  # Human-readable description

    # Geometry
    length: float = 0.0  # inches
    width: float = 0.0  # inches
    area: float = 0.0  # square inches

    # Grind parameters
    stock_removed_total: float = 0.0  # inches
    faces: int = 2  # number of faces
    volume_removed: float = 0.0  # cubic inches

    # Time calculation
    min_per_cuin: float = 3.0
    material_factor: float = 1.0
    grind_material_factor: float = 1.0  # Alias for renderer display
    time_minutes: float = 0.0

    # Override tracking
    _used_override: bool = False


@dataclass
class PocketOperation:
    """Detailed pocket/profile milling operation breakdown."""
    op_name: str = ""  # e.g., "pocket_mill", "profile_mill"
    op_description: str = ""  # Human-readable description

    # Geometry
    pocket_area: float = 0.0  # square inches
    pocket_depth: float = 0.0  # inches
    pocket_width: float = 0.0  # inches (for rectangular pockets)
    pocket_length: float = 0.0  # inches (for rectangular pockets)

    # Tool parameters
    tool_diameter: float = 0.0  # inches

    # Cut parameters
    stepover: float = 0.0  # inches (radial step)
    stepdown: float = 0.0  # inches (axial step per pass)
    z_passes: int = 1  # number of depth passes

    # Path and feed
    pocket_path_length_in: float = 0.0  # total path length in inches
    feed_ipm: float = 0.0  # feed rate in inches per minute

    # Time
    pocket_time_min: float = 0.0  # total time in minutes

    # Override tracking
    _used_override: bool = False
    override_time_minutes: Optional[float] = None


@dataclass
class SlotOperation:
    """Detailed slot milling operation breakdown."""
    op_name: str = ""  # e.g., "slot_mill"
    op_description: str = ""  # Human-readable description

    # Geometry (two radii + straight sides)
    slot_length: float = 0.0  # inches (overall length)
    slot_width: float = 0.0  # inches (diameter of endmill / slot width)
    slot_depth: float = 0.0  # inches
    slot_radius: float = 0.0  # inches (end radius)

    # Tool parameters
    tool_diameter: float = 0.0  # inches

    # Cut parameters
    stepdown: float = 0.0  # inches (axial step per pass)
    z_passes: int = 1  # number of depth passes

    # Path and feed
    slot_path_length_in: float = 0.0  # total path length in inches
    feed_ipm: float = 0.0  # feed rate in inches per minute

    # Time calculation (using formula: base + k_len * length + k_dep * depth)
    base_slot_min: float = 0.0  # base time constant
    k_len: float = 0.0  # length coefficient
    k_dep: float = 0.0  # depth coefficient
    slot_mill_time_min: float = 0.0  # total time in minutes

    # Override tracking
    _used_override: bool = False
    override_time_minutes: Optional[float] = None


@dataclass
class MachineHoursBreakdown:
    """Machine hours estimation breakdown."""
    # Operations by type (hole operations)
    drill_operations: Optional[List[HoleOperation]] = None
    tap_operations: Optional[List[HoleOperation]] = None
    cbore_operations: Optional[List[HoleOperation]] = None
    cdrill_operations: Optional[List[HoleOperation]] = None
    jig_grind_operations: Optional[List[HoleOperation]] = None
    edm_operations: Optional[List[HoleOperation]] = None
    edge_break_operations: Optional[List[HoleOperation]] = None
    etch_operations: Optional[List[HoleOperation]] = None
    polish_operations: Optional[List[HoleOperation]] = None

    # Operations by type (plan operations)
    milling_operations: Optional[List[MillingOperation]] = None
    grinding_operations: Optional[List[GrindingOperation]] = None
    pocket_operations: Optional[List[PocketOperation]] = None
    slot_operations: Optional[List[SlotOperation]] = None
    waterjet_operations: Optional[List[Dict[str, Any]]] = None  # NEW: Waterjet operations (first-class)

    # Time totals by operation type (from hole table)
    total_drill_minutes: float = 0.0
    total_tap_minutes: float = 0.0
    total_cbore_minutes: float = 0.0
    total_cdrill_minutes: float = 0.0
    total_jig_grind_minutes: float = 0.0
    total_edge_break_minutes: float = 0.0
    total_etch_minutes: float = 0.0
    total_polish_minutes: float = 0.0

    # Time totals by operation category (from plan operations)
    total_milling_minutes: float = 0.0  # Includes squaring ops
    total_grinding_minutes: float = 0.0  # Includes wet grind squaring
    total_pocket_minutes: float = 0.0  # Pocket milling operations
    total_slot_minutes: float = 0.0  # Slot milling operations
    total_edm_minutes: float = 0.0
    total_other_minutes: float = 0.0
    total_waterjet_minutes: float = 0.0  # NEW: Waterjet operations (promoted to first-class)
    total_cmm_minutes: float = 0.0  # CMM checking time (machine only, setup is in labor)
    total_inspection_minutes: float = 0.0  # Non-CMM inspection time (e.g., in-process checks for punch parts)
    cmm_holes_checked: int = 0  # Number of holes inspected by CMM
    holes_total: int = 0  # Total number of holes from hole table (sum of QTY)
    hole_entries: int = 0  # Count of unique hole groups (A, B, C, etc.) from hole table

    # Other operations detail (NEW: detailed breakdown instead of single scalar)
    other_ops_detail: Optional[List[Dict[str, Any]]] = None  # Detailed "other operations" breakdown

    # Overall totals
    total_minutes: float = 0.0
    total_hours: float = 0.0
    machine_cost: float = 0.0  # Total cost at machine rate

    def __post_init__(self):
        """Initialize empty lists."""
        if self.drill_operations is None:
            self.drill_operations = []
        if self.tap_operations is None:
            self.tap_operations = []
        if self.cbore_operations is None:
            self.cbore_operations = []
        if self.cdrill_operations is None:
            self.cdrill_operations = []
        if self.jig_grind_operations is None:
            self.jig_grind_operations = []
        if self.edm_operations is None:
            self.edm_operations = []
        if self.edge_break_operations is None:
            self.edge_break_operations = []
        if self.etch_operations is None:
            self.etch_operations = []
        if self.polish_operations is None:
            self.polish_operations = []
        if self.milling_operations is None:
            self.milling_operations = []
        if self.grinding_operations is None:
            self.grinding_operations = []
        if self.pocket_operations is None:
            self.pocket_operations = []
        if self.slot_operations is None:
            self.slot_operations = []
        if self.waterjet_operations is None:
            self.waterjet_operations = []
        if self.other_ops_detail is None:
            self.other_ops_detail = []


@dataclass
class LaborHoursBreakdown:
    """Labor hours estimation breakdown."""
    # Labor by category (minutes)
    setup_minutes: float = 0.0
    programming_minutes: float = 0.0
    machining_steps_minutes: float = 0.0
    inspection_minutes: float = 0.0
    finishing_minutes: float = 0.0
    misc_overhead_minutes: float = 0.0  # Any additional overhead not in categories

    # Totals
    total_minutes: float = 0.0
    total_hours: float = 0.0
    labor_cost: float = 0.0  # Total cost at labor rate

    # Input parameters (for reference)
    ops_total: int = 0
    holes_total: int = 0
    tool_changes: int = 0
    fixturing_complexity: int = 1  # 0=none, 1=light, 2=moderate, 3=complex

    # Detailed breakdown of finishing operations
    finishing_detail: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        """Initialize empty lists."""
        if self.finishing_detail is None:
            self.finishing_detail = []

    def categories_sum(self) -> float:
        """Return sum of all category minutes."""
        return (
            self.setup_minutes +
            self.programming_minutes +
            self.machining_steps_minutes +
            self.inspection_minutes +
            self.finishing_minutes +
            self.misc_overhead_minutes
        )


@dataclass
class CostSummary:
    """Overall cost summary."""
    # Per-unit costs
    direct_cost: float = 0.0
    machine_cost: float = 0.0
    labor_cost: float = 0.0
    total_cost: float = 0.0

    # Total costs (for quantity > 1)
    total_direct_cost: float = 0.0
    total_machine_cost: float = 0.0
    total_labor_cost: float = 0.0
    total_total_cost: float = 0.0

    # Margin and final price
    margin_rate: float = 0.15
    margin_amount: float = 0.0
    final_price: float = 0.0  # Per-unit price
    total_final_price: float = 0.0  # Total price for all units


@dataclass
class QuoteData:
    """
    Complete quote data structure holding all extraction results.

    This is the main data structure that aggregates all information from
    DirectCostHelper and ProcessPlanner.
    """
    # Metadata
    cad_file_path: str = ""
    cad_file_name: str = ""
    extraction_timestamp: str = ""
    quantity: int = 1  # Number of parts to quote

    # Core data
    part_dimensions: PartDimensions = None
    material_info: MaterialInfo = None
    stock_info: StockInfo = None
    scrap_info: ScrapInfo = None
    direct_cost_breakdown: DirectCostBreakdown = None
    machine_hours: MachineHoursBreakdown = None
    labor_hours: LaborHoursBreakdown = None
    cost_summary: CostSummary = None

    # Raw plan data (optional, for reference)
    raw_plan: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize nested dataclasses."""
        if self.part_dimensions is None:
            self.part_dimensions = PartDimensions()
        if self.material_info is None:
            self.material_info = MaterialInfo()
        if self.stock_info is None:
            self.stock_info = StockInfo()
        if self.scrap_info is None:
            self.scrap_info = ScrapInfo()
        if self.direct_cost_breakdown is None:
            self.direct_cost_breakdown = DirectCostBreakdown()
        if self.machine_hours is None:
            self.machine_hours = MachineHoursBreakdown()
        if self.labor_hours is None:
            self.labor_hours = LaborHoursBreakdown()
        if self.cost_summary is None:
            self.cost_summary = CostSummary()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        return asdict(self)

    def to_json(self, filepath: Optional[str | Path] = None, indent: int = 2) -> str:
        """
        Convert to JSON string.

        Args:
            filepath: Optional path to save JSON file
            indent: JSON indentation level

        Returns:
            JSON string
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent)

        if filepath:
            Path(filepath).write_text(json_str)

        return json_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QuoteData:
        """Create QuoteData from dictionary."""
        # Convert nested dicts to dataclasses
        if 'part_dimensions' in data and isinstance(data['part_dimensions'], dict):
            data['part_dimensions'] = PartDimensions(**data['part_dimensions'])
        if 'material_info' in data and isinstance(data['material_info'], dict):
            data['material_info'] = MaterialInfo(**data['material_info'])
        if 'stock_info' in data and isinstance(data['stock_info'], dict):
            data['stock_info'] = StockInfo(**data['stock_info'])
        if 'scrap_info' in data and isinstance(data['scrap_info'], dict):
            data['scrap_info'] = ScrapInfo(**data['scrap_info'])
        if 'direct_cost_breakdown' in data and isinstance(data['direct_cost_breakdown'], dict):
            data['direct_cost_breakdown'] = DirectCostBreakdown(**data['direct_cost_breakdown'])
        if 'machine_hours' in data and isinstance(data['machine_hours'], dict):
            # Convert hole operations lists
            machine_data = data['machine_hours']
            for key in ['drill_operations', 'tap_operations', 'cbore_operations',
                       'cdrill_operations', 'jig_grind_operations', 'edm_operations']:
                if key in machine_data and isinstance(machine_data[key], list):
                    machine_data[key] = [HoleOperation(**op) if isinstance(op, dict) else op
                                        for op in machine_data[key]]
            data['machine_hours'] = MachineHoursBreakdown(**machine_data)
        if 'labor_hours' in data and isinstance(data['labor_hours'], dict):
            data['labor_hours'] = LaborHoursBreakdown(**data['labor_hours'])
        if 'cost_summary' in data and isinstance(data['cost_summary'], dict):
            data['cost_summary'] = CostSummary(**data['cost_summary'])

        return cls(**data)

    @classmethod
    def from_json(cls, json_str_or_path: str | Path) -> QuoteData:
        """
        Load QuoteData from JSON string or file.

        Args:
            json_str_or_path: JSON string or path to JSON file

        Returns:
            QuoteData instance
        """
        # Check if it's a file path
        try:
            path = Path(json_str_or_path)
            if path.exists():
                json_str = path.read_text()
            else:
                json_str = json_str_or_path
        except:
            json_str = json_str_or_path

        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class OrderData:
    """
    Order data structure that can hold multiple parts/files.

    Enables multi-file orders where each part has its own quote data,
    but shipping and totals are calculated at the order level.
    """
    # Order metadata
    order_id: str = ""
    order_name: str = ""
    order_timestamp: str = ""

    # Parts in this order (list of QuoteData objects)
    parts: List[QuoteData] = None

    # Order-level overrides (if any)
    notes: str = ""

    def __post_init__(self):
        """Initialize parts list if None."""
        if self.parts is None:
            self.parts = []
        if not self.order_timestamp:
            from datetime import datetime
            self.order_timestamp = datetime.now().isoformat()
        if not self.order_id:
            # Generate a simple order ID based on timestamp
            self.order_id = f"ORD_{self.order_timestamp.replace(':', '').replace('-', '').replace('.', '')[:14]}"

    def add_part(self, quote_data: QuoteData) -> int:
        """
        Add a part to the order.

        Args:
            quote_data: QuoteData for the part to add

        Returns:
            Index of the added part
        """
        self.parts.append(quote_data)
        return len(self.parts) - 1

    def remove_part(self, index: int) -> None:
        """
        Remove a part from the order.

        Args:
            index: Index of the part to remove
        """
        if 0 <= index < len(self.parts):
            del self.parts[index]

    def get_part(self, index: int) -> Optional[QuoteData]:
        """
        Get a part by index.

        Args:
            index: Index of the part

        Returns:
            QuoteData or None if index is invalid
        """
        if 0 <= index < len(self.parts):
            return self.parts[index]
        return None

    def get_total_weight_lb(self) -> float:
        """
        Calculate total order weight (for shipping).

        Returns sum of (part_weight * quantity) for all parts.
        """
        total = 0.0
        for part in self.parts:
            if part.stock_info and part.stock_info.mcmaster_weight:
                # Stock weight per piece * quantity for this part
                total += part.stock_info.mcmaster_weight * part.quantity
        return total

    def get_parts_subtotal(self) -> float:
        """
        Calculate total cost of all parts (before order-level shipping).

        Returns sum of final_price for all parts.
        Note: Each part's final_price already includes per-part shipping.
        For order-level shipping, we'll need to subtract per-part shipping
        and add order-level shipping instead.
        """
        subtotal = 0.0
        for part in self.parts:
            if part.cost_summary:
                # Use total_final_price if quantity > 1
                if part.quantity > 1:
                    subtotal += part.cost_summary.total_final_price
                else:
                    subtotal += part.cost_summary.final_price
        return subtotal

    def get_order_shipping_cost(self) -> float:
        """
        Calculate order-level shipping based on total weight.

        Uses the existing shipping estimator but applies it to total order weight.
        """
        from cad_quoter.pricing.mcmaster_helpers import estimate_mcmaster_shipping
        total_weight = self.get_total_weight_lb()
        return estimate_mcmaster_shipping(total_weight)

    def get_order_total(self) -> float:
        """
        Calculate final order total.

        This is the grand total the customer pays:
        - Sum of all part costs (excluding per-part shipping)
        - Plus order-level shipping

        Note: Current implementation has shipping embedded in each part.
        For true order-level shipping, we need to:
        1. Sum parts without their individual shipping
        2. Add single order shipping based on total weight
        """
        # For now, return parts subtotal + order shipping
        # (This will be refined when we update the cost calculation)
        parts_subtotal_no_shipping = 0.0
        for part in self.parts:
            if part.cost_summary:
                # Get the total cost without shipping
                # Each part's final price includes shipping, so we subtract it
                if part.quantity > 1:
                    part_total = part.cost_summary.total_final_price
                else:
                    part_total = part.cost_summary.final_price

                # Subtract the per-part shipping cost
                if part.direct_cost_breakdown:
                    part_shipping = part.direct_cost_breakdown.shipping
                    parts_subtotal_no_shipping += (part_total - part_shipping * part.quantity)
                else:
                    parts_subtotal_no_shipping += part_total

        order_shipping = self.get_order_shipping_cost()
        return parts_subtotal_no_shipping + order_shipping

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        return {
            'order_id': self.order_id,
            'order_name': self.order_name,
            'order_timestamp': self.order_timestamp,
            'notes': self.notes,
            'parts': [part.to_dict() for part in self.parts]
        }

    def to_json(self, filepath: Optional[str | Path] = None, indent: int = 2) -> str:
        """
        Convert to JSON string.

        Args:
            filepath: Optional path to save JSON file
            indent: JSON indentation level

        Returns:
            JSON string
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent)

        if filepath:
            Path(filepath).write_text(json_str)

        return json_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderData':
        """Create OrderData from dictionary."""
        # Convert parts list
        if 'parts' in data and isinstance(data['parts'], list):
            data['parts'] = [
                QuoteData.from_dict(part) if isinstance(part, dict) else part
                for part in data['parts']
            ]

        return cls(**data)

    @classmethod
    def from_json(cls, json_str_or_path: str | Path) -> 'OrderData':
        """
        Load OrderData from JSON string or file.

        Args:
            json_str_or_path: JSON string or path to JSON file

        Returns:
            OrderData instance
        """
        # Check if it's a file path
        try:
            path = Path(json_str_or_path)
            if path.exists():
                json_str = path.read_text()
            else:
                json_str = json_str_or_path
        except:
            json_str = json_str_or_path

        data = json.loads(json_str)
        return cls.from_dict(data)


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================


def detect_punch_drawing(cad_file_path: Path, text_dump: str = None, plan: dict = None) -> bool:
    """
    Detect if a CAD file is a punch drawing (individual punch component).

    Detection uses multiple heuristics with confidence scoring:
    1. Text-based: Filename and drawing text containing punch keywords
    2. Geometry-based: Cylindrical shape with high aspect ratio (L/D > 2.5)
    3. Feature-based: Turned/ground operations, few holes, no windows

    Excludes die shoes, holders, and other tooling that references punches.

    Args:
        cad_file_path: Path to CAD file
        text_dump: Optional text dump from drawing (if already extracted)
        plan: Optional process plan dict with extracted_dims and ops

    Returns:
        True if file appears to be a punch drawing (confidence score >= 3)
    """
    import re

    confidence_score = 0
    debug_signals = []

    # ========================================================================
    # 1. TEXT-BASED DETECTION (existing logic + enhancements)
    # ========================================================================

    # Check filename
    filename = cad_file_path.stem.upper()

    # Exclusion patterns - these are NOT punches even if they reference punches
    exclusion_patterns = ["SHOE", "HOLDER", "BASE", "PLATE", "DIE SET", "BLOCK", "INSERT", "SPACER"]
    if any(excl in filename for excl in exclusion_patterns):
        debug_signals.append(f"EXCLUSION in filename: {filename}")
        return False

    # Punch indicators in filename
    filename_indicators = ["PUNCH", "PILOT", "PIN", "FORM", "GUIDE POST"]
    if any(ind in filename for ind in filename_indicators):
        confidence_score += 5  # Strong signal from filename
        debug_signals.append(f"Filename match (+5): {filename}")

    # Check text content if provided
    if text_dump:
        text_upper = text_dump.upper()

        # Check for exclusions first - die shoes, holders, etc.
        exclusion_indicators = [
            "DIE SHOE",
            "PUNCH SHOE",
            "PUNCH HOLDER",
            "DIE HOLDER",
            "DIE SET",
            "DIE BASE",
            "PUNCH PLATE",
            "DIE PLATE",
            "BACKING PLATE",
            "STRIPPER PLATE",
            "STRIPPER INSERT",
            "PUNCH BLOCK",
            "PUNCH SPACER",
        ]
        if any(excl in text_upper for excl in exclusion_indicators):
            debug_signals.append(f"EXCLUSION in text: found {[e for e in exclusion_indicators if e in text_upper]}")
            return False

        # Punch indicators in text
        text_indicators = [
            "FORM PUNCH",
            "DIE PUNCH",
            "PIERCING PUNCH",
            "PILOT PIN",
            "SPRING PIN",
            "PUNCH TIP",
            "GUIDE POST",
        ]
        found_indicators = [ind for ind in text_indicators if ind in text_upper]
        if found_indicators:
            confidence_score += 4  # Strong signal from text
            debug_signals.append(f"Text indicators (+4): {found_indicators}")

        # Check for standalone "PUNCH" - but only if no exclusion words nearby
        if "PUNCH" in text_upper and not found_indicators:
            punch_exclusion_suffixes = ["SHOE", "HOLDER", "PLATE", "POCKET", "CLEARANCE", "LOCATION", "BLOCK"]
            punch_pattern = r'\bPUNCH\b'
            for match in re.finditer(punch_pattern, text_upper):
                after_punch = text_upper[match.end():match.end()+15]
                if not any(suffix in after_punch for suffix in punch_exclusion_suffixes):
                    confidence_score += 3  # Moderate signal
                    debug_signals.append("Standalone PUNCH (+3)")
                    break

        # NEW: Check for part number callouts (e.g., "PART 2", "PART 6", "DETAIL 14")
        # Individual component callouts suggest punch parts in assembly drawings
        part_number_patterns = [
            r'\bPART\s*[#:]?\s*([0-9]{1,2})\b',
            r'\bDETAIL\s*[#:]?\s*([0-9]{1,2})\b',
            r'\bITEM\s*[#:]?\s*([0-9]{1,2})\b',
        ]
        for pattern in part_number_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                # Part numbers 1-20 are typically individual components (punches/pins)
                # Part numbers > 100 are typically assemblies/plates
                part_nums = [int(m) for m in matches if m.isdigit()]
                small_part_nums = [n for n in part_nums if 1 <= n <= 20]
                if small_part_nums:
                    confidence_score += 3  # Strong signal for individual components
                    debug_signals.append(f"Small part numbers (+3): {small_part_nums[:5]}")
                    break

    # ========================================================================
    # 2. GEOMETRY-BASED DETECTION (NEW)
    # ========================================================================

    if plan and 'extracted_dims' in plan:
        dims = plan['extracted_dims']
        L = dims.get('L', 0.0)
        W = dims.get('W', 0.0)
        T = dims.get('T', 0.0)

        if L > 0 and W > 0 and T > 0:
            # Check if part is cylindrical (two dimensions similar, one much larger)
            # Sort dimensions to get smallest, middle, largest
            sorted_dims = sorted([L, W, T])
            smallest = sorted_dims[0]
            middle = sorted_dims[1]
            largest = sorted_dims[2]

            # Cylindrical check: smallest ≈ middle (diameter) and largest >> diameter
            diameter_similarity_ratio = middle / smallest if smallest > 0 else 0
            aspect_ratio = largest / middle if middle > 0 else 0
            max_cross_section = max(W, T)

            # Signal 1: Two dimensions are similar (within 30%) → suggests round part
            if 0.7 <= diameter_similarity_ratio <= 1.3:
                confidence_score += 2
                debug_signals.append(f"Round geometry (+2): {smallest:.2f}\" ≈ {middle:.2f}\"")

            # Signal 2: High aspect ratio (L/D > 2.5) → suggests punch/pin shape
            if aspect_ratio > 2.5:
                confidence_score += 2
                debug_signals.append(f"High aspect ratio (+2): L/D = {aspect_ratio:.1f}")

            # Signal 3: Small cross-section (< 3") → punches are typically small
            if max_cross_section < 3.0:
                confidence_score += 1
                debug_signals.append(f"Small diameter (+1): {max_cross_section:.2f}\"")

    # ========================================================================
    # 3. FEATURE-BASED DETECTION (NEW)
    # ========================================================================

    if plan and 'ops' in plan:
        ops = plan.get('ops', [])
        op_types = [op.get('op', '') for op in ops]

        # Signal 1: Has wire EDM windows → NOT a punch (plates have windows)
        windows = plan.get('windows', [])
        if len(windows) > 0:
            confidence_score -= 3  # Strong negative signal
            debug_signals.append(f"EDM windows (-3): {len(windows)} windows")

        # Signal 2: Has grinding operations → suggests turned part
        grind_ops = [op for op in op_types if 'grind' in op.lower()]
        if grind_ops:
            confidence_score += 1
            debug_signals.append(f"Grinding ops (+1): {len(grind_ops)}")

        # Signal 3: Few holes (< 3) → punches typically have 0-2 holes
        hole_sets = plan.get('hole_sets', [])
        total_holes = sum(h.get('qty', 0) for h in hole_sets)
        if total_holes < 3:
            confidence_score += 1
            debug_signals.append(f"Few holes (+1): {total_holes} holes")

        # Signal 4: Many holes (> 10) → likely a plate
        if total_holes > 10:
            confidence_score -= 2
            debug_signals.append(f"Many holes (-2): {total_holes} holes")

    # ========================================================================
    # DECISION: Threshold-based classification
    # ========================================================================

    is_punch = confidence_score >= 3

    # Debug output (can be enabled with verbose flag in future)
    if False:  # Set to True to enable debug output
        print(f"\n=== PUNCH DETECTION: {cad_file_path.name} ===")
        for signal in debug_signals:
            print(f"  {signal}")
        print(f"  TOTAL SCORE: {confidence_score}")
        print(f"  DECISION: {'PUNCH' if is_punch else 'PLATE'}")

    return is_punch


def extract_punch_quote_data(
    cad_file_path: Path,
    quote_data: 'QuoteData',
    verbose: bool = False,
    plan: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract punch-specific data and estimate times.

    Args:
        cad_file_path: Path to DXF/DWG file
        quote_data: QuoteData object to populate
        verbose: Print progress messages
        plan: Optional plan dict with process planning data

    Returns:
        Dict with punch features, plan, and time estimates
    """
    if not PUNCH_EXTRACTION_AVAILABLE:
        return {"error": "Punch extraction modules not available"}

    try:
        # Extract text from DXF
        from cad_quoter.geo_extractor import open_doc, collect_all_text

        doc = open_doc(cad_file_path)
        text_records = list(collect_all_text(doc))
        text_lines = [rec["text"] for rec in text_records if rec.get("text")]
        text_dump = "\n".join(text_lines)

        if verbose:
            print(f"  Extracted {len(text_lines)} text lines from drawing")

        # Extract punch features
        punch_features = extract_punch_features_from_dxf(cad_file_path, text_dump)

        if verbose:
            print(f"  Punch features extracted:")
            print(f"    - Family: {punch_features.family}")
            print(f"    - Shape: {punch_features.shape_type}")
            print(f"    - Length: {punch_features.overall_length_in:.3f}\"")
            print(f"    - Max OD: {punch_features.max_od_or_width_in:.3f}\"")
            print(f"    - Ground diameters: {punch_features.num_ground_diams}")
            print(f"    - Confidence: {punch_features.confidence_score:.2f}")

        # Convert PunchFeatureSummary to dict
        from dataclasses import asdict
        features_dict = asdict(punch_features)

        # Create punch manufacturing plan
        punch_plan = create_punch_plan(features_dict)

        if verbose:
            print(f"  Punch plan created with {len(punch_plan.get('ops', []))} operations")

        # Estimate times
        time_estimates = estimate_punch_times(punch_plan, features_dict)

        if verbose:
            mh = time_estimates.get("machine_hours", {})
            lh = time_estimates.get("labor_hours", {})
            print(f"  Time estimates:")
            print(f"    - Machine hours: {mh.get('total_hours', 0):.2f}")
            print(f"    - Labor hours: {lh.get('total_hours', 0):.2f}")

        return {
            "is_punch": True,
            "punch_features": features_dict,
            "punch_plan": punch_plan,
            "time_estimates": time_estimates,
            "text_dump": text_dump,
        }

    except Exception as e:
        if verbose:
            import traceback
            print(f"  [PUNCH ERROR] Failed to extract punch features: {str(e)}")
            print(f"  {traceback.format_exc()}")
        return {
            "is_punch": True,
            "error": str(e),
        }


def extract_quote_data_from_cad(
    cad_file_path: str | Path,
    machine_rate: float = 45.0,
    labor_rate: float = 45.0,
    margin_rate: float = 0.15,
    material_override: Optional[str] = None,
    catalog_csv_path: Optional[str] = None,
    dimension_override: Optional[tuple[float, float, float]] = None,
    diameter_override: Optional[float] = None,  # Deprecated: use diameter_overrides instead
    diameter_overrides: Optional[tuple[Optional[float], Optional[float]]] = None,  # (diameter_1, diameter_2) for tapered parts
    mcmaster_price_override: Optional[float] = None,
    scrap_value_override: Optional[float] = None,
    quantity: int = 1,
    family_override: Optional[str] = None,
    cmm_inspection_level_override: Optional[str] = None,
    verbose: bool = False
) -> QuoteData:
    """
    Extract complete quote data from a CAD file.

    This is the main extraction function that pulls all data from DirectCostHelper
    and ProcessPlanner and packages it into a unified QuoteData structure.

    Args:
        cad_file_path: Path to CAD file (DXF/DWG)
        machine_rate: Machine hourly rate ($/hr)
        labor_rate: Labor hourly rate ($/hr)
        margin_rate: Profit margin rate (decimal, e.g., 0.15 = 15%)
        material_override: Optional material name override
        catalog_csv_path: Optional path to McMaster catalog CSV
        dimension_override: Optional (length, width, thickness) tuple to override OCR extraction
        diameter_override: Optional diameter override (inches) for cylindrical parts (deprecated - use diameter_overrides)
        diameter_overrides: Optional (diameter_1, diameter_2) tuple for cylindrical parts. Either value can be None.
                           Used for tapered punches where diameter varies along length.
        mcmaster_price_override: Optional manual stock price - skips McMaster API lookup
        scrap_value_override: Optional manual scrap value - skips automatic scrap value calculation
        quantity: Number of parts to quote (affects setup cost amortization and material pricing)
        family_override: Optional part family override (e.g., "Punches" to force punch pipeline)
        cmm_inspection_level_override: Optional CMM inspection level ("Full Inspection", "Critical Only", or "Spot Check")
        verbose: Print extraction progress

    Returns:
        QuoteData with all extraction results

    Example:
        >>> quote_data = extract_quote_data_from_cad("part.dxf", verbose=True)
        >>> print(f"Total cost: ${quote_data.cost_summary.total_cost:.2f}")

        >>> # With dimension override (when OCR fails)
        >>> quote_data = extract_quote_data_from_cad(
        ...     "part.dxf",
        ...     dimension_override=(10.0, 8.0, 0.5)
        ... )

        >>> # With diameter overrides for tapered punch
        >>> quote_data = extract_quote_data_from_cad(
        ...     "tapered_punch.dxf",
        ...     diameter_overrides=(1.5, 1.25)
        ... )
        >>> quote_data.to_json("quote_results.json")
    """
    from cad_quoter.planning import (
        plan_from_cad_file,
        extract_hole_operations_from_cad,
        estimate_hole_table_times
    )
    from cad_quoter.planning.process_planner import (
        LaborInputs,
        compute_labor_minutes,
        estimate_machine_hours_from_plan
    )
    from cad_quoter.pricing.DirectCostHelper import (
        extract_part_info_from_plan,
        get_mcmaster_part_number,
        get_mcmaster_price,
        calculate_total_scrap,
        calculate_scrap_value,
        get_material_density,
        DEFAULT_MATERIAL
    )
    from cad_quoter.pricing.KeywordDetector import detect_material_in_cad
    from cad_quoter.pricing.MaterialMapper import material_mapper
    from cad_quoter.pricing.mcmaster_helpers import (
        pick_mcmaster_plate_sku,
        pick_mcmaster_cylindrical_sku,
        load_mcmaster_catalog_rows,
        estimate_mcmaster_shipping
    )
    from cad_quoter.resources import default_catalog_csv

    cad_file_path = Path(cad_file_path)

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXTRACTING QUOTE DATA: {cad_file_path.name}")
        print(f"{'='*70}\n")

    # Initialize QuoteData
    quote_data = QuoteData(
        cad_file_path=str(cad_file_path),
        cad_file_name=cad_file_path.name,
        extraction_timestamp=datetime.now().isoformat(),
        quantity=quantity
    )

    # ========================================================================
    # STEP 1: Extract process plan (ODA + OCR)
    # ========================================================================
    if verbose:
        print("[1/5] Extracting process plan (ODA + OCR)...")

    # Skip expensive OCR if manual dimensions provided (saves ~43 seconds)
    use_ocr = dimension_override is None
    if not use_ocr and verbose:
        print("  Skipping OCR dimension extraction (manual dimensions provided)")

    plan = plan_from_cad_file(cad_file_path, use_paddle_ocr=use_ocr, verbose=verbose)
    quote_data.raw_plan = plan if verbose else None  # Only store if verbose

    # FALLBACK: If dimensions weren't extracted (use_ocr=False), force extraction for die sections
    # This ensures form operations have proper perimeter calculations
    extracted_dims = plan.get('extracted_dims', {})
    dims_missing = not extracted_dims or all(extracted_dims.get(k, 0.0) == 0.0 for k in ['L', 'W', 'T'])

    if dims_missing and not dimension_override:
        if verbose:
            print("  [FALLBACK] Dimensions missing - extracting with DimensionFinder...")

        from cad_quoter.planning import extract_dimensions_from_cad
        dims_tuple = extract_dimensions_from_cad(cad_file_path)

        if dims_tuple:
            L, W, T = dims_tuple
            if verbose:
                print(f"  [FALLBACK] Extracted dimensions: L={L:.3f}\", W={W:.3f}\", T={T:.3f}\"")

            # Update plan with extracted dimensions
            plan['extracted_dims'] = {'L': L, 'W': W, 'T': T}

            # For die sections, regenerate the plan with proper dimensions
            # This ensures form operations calculate correct perimeter
            planner_family = plan.get('planner', '')
            if 'Sections' in planner_family or 'die_section' in str(plan.get('meta', {}).get('sub_type', '')):
                if verbose:
                    print(f"  [FALLBACK] Regenerating die section plan with dimensions...")

                from cad_quoter.planning.process_planner import plan_job
                params = {
                    'plate_LxW': (L, W),
                    'T': T,
                    'material': plan.get('material', 'A2'),
                    'has_internal_form': True,  # Preserve form detection
                    'hole_sets': plan.get('hole_sets', []),
                }
                regenerated_plan = plan_job(planner_family, params)
                regenerated_plan['extracted_dims'] = {'L': L, 'W': W, 'T': T}
                regenerated_plan['source_file'] = plan.get('source_file')
                plan = regenerated_plan

                if verbose:
                    print(f"  [FALLBACK] Plan regenerated with dimensions")

    # Use extracted quantity from plan if available and user didn't specify a quantity
    extracted_qty = plan.get("extracted_part_quantity", 1)
    if quantity == 1 and extracted_qty > 1:
        quote_data.quantity = extracted_qty
        if verbose:
            print(f"  Using extracted part quantity: {extracted_qty}")

    # ========================================================================
    # CHECK: Is this a punch drawing?
    # ========================================================================
    is_punch = False
    punch_data = None

    if PUNCH_EXTRACTION_AVAILABLE:
        # First check for family override - this takes priority
        if family_override and family_override.lower() in ("punches", "punch"):
            is_punch = True
            if verbose:
                print(f"[PUNCH] Using family override: {family_override}")
        else:
            # Auto-detect using enhanced heuristics (text + geometry + features)
            # Pass plan to enable geometry-based and feature-based detection
            is_punch = detect_punch_drawing(cad_file_path, plan.get('text_dump'), plan)

        if is_punch:
            if verbose:
                print("[PUNCH] Detected punch drawing - using punch extraction pipeline")

            # Extract punch-specific data
            punch_data = extract_punch_quote_data(cad_file_path, quote_data, verbose, plan)

            if "error" not in punch_data:
                # Use punch features for part dimensions
                features = punch_data.get("punch_features", {})

                # Determine if part is cylindrical based on family
                family = features.get("family", "")
                is_cylindrical = family in ("guide_post", "round_punch", "pilot_pin", "bushing")

                # Set part dimensions from punch features
                # For rectangular shapes (plates), use body_thickness_in
                # For round shapes, use max_od_or_width_in as the "diameter"
                shape_type = features.get("shape_type", "round")
                if shape_type == "rectangular":
                    thickness = features.get("body_thickness_in", 0.0)
                    # Fallback if body_thickness_in not set
                    if thickness <= 0:
                        thickness = features.get("max_od_or_width_in", 0.0)
                else:
                    thickness = features.get("max_od_or_width_in", 0.0)  # OD for round

                # For cylindrical parts, the diameter is the max_od_or_width_in
                diameter = features.get("max_od_or_width_in", 0.0) if is_cylindrical else 0.0

                quote_data.part_dimensions = PartDimensions(
                    length=features.get("overall_length_in", 0.0),
                    width=features.get("max_od_or_width_in", 0.0),
                    thickness=thickness,
                    diameter=diameter,
                    is_cylindrical=is_cylindrical,
                )

                # If punch dimensions are zero, fall back to DimensionFinder
                if (quote_data.part_dimensions.length == 0.0 and
                    quote_data.part_dimensions.width == 0.0):
                    if verbose:
                        print("  [PUNCH] Punch dimensions are zero, falling back to DimensionFinder...")

                    # Extract dimensions from CAD file
                    from cad_quoter.planning.process_planner import extract_dimensions_from_cad
                    dims = extract_dimensions_from_cad(cad_file_path)
                    if dims:
                        L, W, T = dims
                        quote_data.part_dimensions = PartDimensions(
                            length=L,
                            width=W,
                            thickness=T,
                        )
                        if verbose:
                            print(f"  [PUNCH] DimensionFinder found: {L:.3f} x {W:.3f} x {T:.3f}")
                    else:
                        if verbose:
                            print("  [PUNCH] DimensionFinder also failed to extract dimensions")

                # Apply dimension override for punch parts
                if dimension_override:
                    dim1, dim2, dim3 = dimension_override
                    if verbose:
                        print(f"  [PUNCH] Dimension override input: {dim1} x {dim2} x {dim3}")

                    # For punch parts, auto-reorder dimensions:
                    # Users often enter dimensions as smallest×middle×largest (e.g., .148x.445x2)
                    # but punch features need: length=OAL (largest), width=OD, thickness
                    # Sort to get: thickness (smallest), width (middle), length (largest)
                    sorted_dims = sorted([dim1, dim2, dim3])
                    thickness = sorted_dims[0]  # smallest
                    width = sorted_dims[1]      # middle
                    length = sorted_dims[2]     # largest (OAL)

                    if verbose:
                        print(f"  [PUNCH] Reordered to: length={length}, width={width}, thickness={thickness}")

                    # For cylindrical parts, preserve the original detected diameter
                    # Dimension overrides (L×W×T) are intended for rectangular plates,
                    # not for overriding the diameter of round parts
                    if is_cylindrical:
                        # Keep original diameter from punch features (already set at line 912)
                        diameter = quote_data.part_dimensions.diameter
                        if verbose:
                            print(f"  [PUNCH] Cylindrical part - preserving detected diameter: {diameter:.3f}\" (use Diameter 1/2 fields to override)")
                            print(f"  [PUNCH] DEBUG: quote_data.part_dimensions before update = diameter:{quote_data.part_dimensions.diameter}, is_cyl:{quote_data.part_dimensions.is_cylindrical}")
                    else:
                        diameter = 0.0

                    quote_data.part_dimensions = PartDimensions(
                        length=length,
                        width=diameter if is_cylindrical else width,
                        thickness=diameter if is_cylindrical else thickness,
                        diameter=diameter,
                        is_cylindrical=is_cylindrical,
                    )

                    if verbose and is_cylindrical:
                        print(f"  [PUNCH] DEBUG: quote_data.part_dimensions AFTER update = diameter:{quote_data.part_dimensions.diameter}, is_cyl:{quote_data.part_dimensions.is_cylindrical}")
                        print(f"  [PUNCH] Cylindrical part dimensions set to: L={length:.3f}\", W={diameter:.3f}\", T={diameter:.3f}\" (W and T match diameter)")

                    # Also update punch_features dict so punch plan uses correct values
                    features["overall_length_in"] = length

                    # For cylindrical parts, preserve the detected diameter in max_od_or_width_in
                    # For rectangular parts, use the width from dimension override
                    if is_cylindrical:
                        # Keep max_od_or_width_in unchanged (it's the detected diameter)
                        pass  # Don't overwrite features["max_od_or_width_in"]
                    else:
                        features["max_od_or_width_in"] = width
                        if width != thickness:
                            # Rectangular punch - set body dimensions
                            features["body_width_in"] = width
                            features["body_thickness_in"] = thickness

                    # Regenerate punch plan with new dimensions
                    from dataclasses import asdict
                    from cad_quoter.planning.process_planner import create_punch_plan

                    updated_features_dict = features  # Already a dict
                    punch_plan = create_punch_plan(updated_features_dict)
                    punch_data["punch_plan"] = punch_plan
                    punch_data["punch_features"] = features

                    if verbose:
                        print(f"  [PUNCH] Plan regenerated with dimension overrides")

                # Apply diameter overrides for cylindrical punch parts
                # Support both new diameter_overrides tuple and legacy diameter_override
                diameter_1_override = None
                diameter_2_override = None

                if diameter_overrides and is_cylindrical:
                    diameter_1_override, diameter_2_override = diameter_overrides
                elif diameter_override and is_cylindrical:
                    # Backward compatibility: treat old diameter_override as diameter_1
                    diameter_1_override = diameter_override

                if diameter_1_override and is_cylindrical:
                    original_diameter = quote_data.part_dimensions.diameter
                    if verbose:
                        print(f"  [PUNCH] Applying diameter override(s):")
                        print(f"    Original detected diameter: {original_diameter:.3f}\"")
                        print(f"    Diameter 1 override: {diameter_1_override:.3f}\"")
                        if diameter_2_override:
                            print(f"    Diameter 2 override: {diameter_2_override:.3f}\" (tapered)")
                        print(f"    → Stock lookup will use diameter: {max(diameter_1_override, diameter_2_override) if diameter_2_override else diameter_1_override:.3f}\"")

                    quote_data.part_dimensions.diameter = diameter_1_override
                    if diameter_2_override:
                        quote_data.part_dimensions.diameter_2 = diameter_2_override

                    # Update punch features with new diameter (use larger diameter for stock selection)
                    max_diameter = max(diameter_1_override, diameter_2_override) if diameter_2_override else diameter_1_override
                    features["max_od_or_width_in"] = max_diameter

                    # IMPORTANT: For cylindrical parts, also update width and thickness to match the diameter
                    # This ensures scrap calculations use the correct dimensions
                    # (width and thickness should equal the diameter for round bar stock)
                    quote_data.part_dimensions.width = max_diameter
                    quote_data.part_dimensions.thickness = max_diameter

                    if verbose:
                        print(f"    → Updated part dimensions: L={quote_data.part_dimensions.length:.3f}\", W={max_diameter:.3f}\", T={max_diameter:.3f}\"")

                    # Regenerate punch plan with new diameter
                    from dataclasses import asdict
                    from cad_quoter.planning.process_planner import create_punch_plan

                    updated_features_dict = features  # Already a dict
                    punch_plan = create_punch_plan(updated_features_dict)
                    punch_data["punch_plan"] = punch_plan
                    punch_data["punch_features"] = features

                    if verbose:
                        print(f"  [PUNCH] Plan regenerated with diameter override(s)")

                # Set material from punch features
                punch_material = features.get("material_callout") or DEFAULT_MATERIAL
                density = get_material_density(punch_material)
                quote_data.material_info = MaterialInfo(
                    material_name=punch_material,
                    material_family="tool_steel",
                    density=density,
                    detected_from_cad=features.get("material_callout") is not None,
                    is_default=features.get("material_callout") is None,
                )

                # Set machine hours from punch estimates
                # Round each component to 2 decimal places for consistent display
                time_estimates = punch_data.get("time_estimates", {})
                mh = time_estimates.get("machine_hours", {})
                punch_milling_min = round(mh.get("total_milling_minutes", 0.0), 2)
                punch_grinding_min = round(mh.get("total_grinding_minutes", 0.0), 2)
                punch_tap_min = round(mh.get("total_tap_minutes", 0.0), 2)
                punch_drill_min = round(mh.get("total_drill_minutes", 0.0), 2)
                punch_edm_min = round(mh.get("total_edm_minutes", 0.0), 2)
                punch_other_min = round(mh.get("total_other_minutes", 0.0), 2)
                punch_cmm_min = round(mh.get("total_cmm_minutes", 0.0), 2)
                punch_inspection_min = round(mh.get("total_inspection_minutes", 0.0), 2)
                punch_etch_min = round(mh.get("total_etch_minutes", 0.0), 2)
                punch_edge_break_min = round(mh.get("total_edge_break_minutes", 0.0), 2)
                punch_polish_min = round(mh.get("total_polish_minutes", 0.0), 2)
                punch_total_min = round(mh.get("total_minutes", 0.0), 2)
                punch_machine_hours = round(punch_total_min / 60.0, 2)
                # Compute machine cost directly from total minutes for accuracy
                # (avoids rounding errors from hours conversion)
                punch_machine_cost = round(punch_total_min * (machine_rate / 60.0), 2)

                quote_data.machine_hours = MachineHoursBreakdown(
                    total_milling_minutes=punch_milling_min,
                    total_grinding_minutes=punch_grinding_min,
                    total_tap_minutes=punch_tap_min,
                    total_drill_minutes=punch_drill_min,
                    total_edm_minutes=punch_edm_min,
                    total_other_minutes=punch_other_min,
                    total_cmm_minutes=punch_cmm_min,
                    total_inspection_minutes=punch_inspection_min,
                    total_etch_minutes=punch_etch_min,
                    total_edge_break_minutes=punch_edge_break_min,
                    total_polish_minutes=punch_polish_min,
                    total_minutes=punch_total_min,
                    total_hours=punch_machine_hours,
                    machine_cost=punch_machine_cost,
                )

                # Set labor hours from punch estimates
                lh = time_estimates.get("labor_hours", {})
                punch_labor_hours = lh.get("total_hours", 0.0)

                # Extract individual category minutes
                setup_min = lh.get("total_setup_minutes", 0.0)
                programming_min = lh.get("cam_programming_minutes", 0.0)
                machining_min = lh.get("handling_minutes", 0.0)
                inspection_min = lh.get("inspection_minutes", 0.0)
                finishing_min = lh.get("deburring_minutes", 0.0)
                labor_total = lh.get("total_minutes", 0.0)

                # Calculate visible categories sum and misc overhead
                visible_sum = setup_min + programming_min + machining_min + inspection_min + finishing_min
                misc_overhead_min = labor_total - visible_sum

                # Round each category to 2 decimals for display consistency
                setup_min = round(setup_min, 2)
                programming_min = round(programming_min, 2)
                machining_min = round(machining_min, 2)
                inspection_min = round(inspection_min, 2)
                finishing_min = round(finishing_min, 2)
                labor_total = round(labor_total, 2)
                misc_overhead_min = round(misc_overhead_min, 2)
                punch_labor_hours = round(punch_labor_hours, 2)

                # Compute labor cost directly from total minutes for accuracy
                # (avoids rounding errors from hours conversion)
                punch_labor_cost = round(labor_total * (labor_rate / 60.0), 2)

                quote_data.labor_hours = LaborHoursBreakdown(
                    setup_minutes=setup_min,
                    programming_minutes=programming_min,
                    machining_steps_minutes=machining_min,
                    inspection_minutes=inspection_min,
                    finishing_minutes=finishing_min,
                    misc_overhead_minutes=misc_overhead_min,
                    total_minutes=labor_total,
                    total_hours=punch_labor_hours,
                    labor_cost=punch_labor_cost,
                )

                # Store punch-specific data in quote_data
                quote_data.raw_plan = {
                    "is_punch": True,
                    "punch_features": features,
                    "punch_plan": punch_data.get("punch_plan", {}),
                    "planner": "punch_planner",
                }

                if verbose:
                    print(f"[PUNCH] Extraction complete")
                    print(f"  Machine hours: {quote_data.machine_hours.total_hours:.2f}")
                    print(f"  Labor hours: {quote_data.labor_hours.total_hours:.2f}")

    # ========================================================================
    # STEP 2: Extract part dimensions and material
    # ========================================================================
    if verbose:
        print("[2/5] Extracting dimensions and material...")

    # Skip standard extraction if punch data was successfully extracted
    if is_punch and punch_data and "error" not in punch_data:
        if verbose:
            print("  [PUNCH] Using punch-extracted dimensions and material")
        # Jump to cost calculation section
        pass  # Continue with rest of function using punch data
    else:
        # Standard plate/die extraction path
        pass

    # Apply dimension override if provided (useful when OCR fails)
    if dimension_override and not (is_punch and punch_data and "error" not in punch_data):
        from cad_quoter.planning.process_planner import plan_job

        length, width, thickness = dimension_override
        if verbose:
            print(f"  Using dimension override: {length} x {width} x {thickness}")

        # Update extracted dimensions
        if 'extracted_dims' not in plan:
            plan['extracted_dims'] = {}
        plan['extracted_dims']['L'] = length
        plan['extracted_dims']['W'] = width
        plan['extracted_dims']['T'] = thickness

        # IMPORTANT: Regenerate plan with new dimensions
        # The original plan was generated with OCR dimensions, but squaring logic
        # depends on L/W/T, so we need to regenerate with the override dimensions
        params = {
            'plate_LxW': (length, width),
            'T': thickness,
            'profile_tol': plan.get('profile_tol'),
            'flatness_spec': plan.get('flatness_spec'),
            'parallelism_spec': plan.get('parallelism_spec'),
            'windows_need_sharp': plan.get('windows_need_sharp', False),
            'window_corner_radius_req': plan.get('window_corner_radius_req'),
            'hole_sets': plan.get('hole_sets', []),
        }

        # Get the planner type from original plan
        planner_type = plan.get('planner', 'die_plate')
        regenerated_plan = plan_job(planner_type, params)

        # Preserve extracted_dims and other metadata from original plan
        regenerated_plan['extracted_dims'] = plan['extracted_dims']
        if 'extracted_material' in plan:
            regenerated_plan['extracted_material'] = plan['extracted_material']

        # Replace plan with regenerated version
        plan = regenerated_plan

        if verbose:
            print(f"  Plan regenerated with dimension overrides")

    # Material detection and dimension extraction (skip for punch parts)
    if not (is_punch and punch_data and "error" not in punch_data):
        if material_override:
            material = material_override
            detected_from_cad = False
        else:
            # Detect material from CAD file
            material = detect_material_in_cad(cad_file_path)
            detected_from_cad = True
            if material == "GENERIC":
                material = DEFAULT_MATERIAL
                detected_from_cad = False

        # Get part info from plan
        part_info = extract_part_info_from_plan(plan, material)

        # Populate part dimensions
        quote_data.part_dimensions = PartDimensions(
            length=part_info.length,
            width=part_info.width,
            thickness=part_info.thickness,
            volume=part_info.volume,
            area=part_info.area
        )

        # Populate material info
        density = get_material_density(material)
        material_lower = material.lower()
        if any(kw in material_lower for kw in ["aluminum", "aluminium", "6061", "mic6"]):
            material_family = "aluminum"
        elif any(kw in material_lower for kw in ["stainless", "304", "316", "17-4"]):
            material_family = "stainless"
        elif any(kw in material_lower for kw in ["steel", "p20", "a36", "1018"]):
            material_family = "steel"
        else:
            material_family = "aluminum"  # Default

        quote_data.material_info = MaterialInfo(
            material_name=material,
            material_family=material_family,
            density=density,
            detected_from_cad=detected_from_cad,
            is_default=(material == DEFAULT_MATERIAL and not material_override)
        )

        if verbose:
            print(f"  Dimensions: {part_info.length:.2f} x {part_info.width:.2f} x {part_info.thickness:.2f} in")
            print(f"  Material: {material} ({'detected' if detected_from_cad else 'default'})")
    else:
        # For punch parts, use the already-set dimensions and material
        material = quote_data.material_info.material_name
        part_info = type('PartInfo', (), {
            'length': quote_data.part_dimensions.length,
            'width': quote_data.part_dimensions.width,
            'thickness': quote_data.part_dimensions.thickness,
            'volume': quote_data.part_dimensions.volume,
            'area': quote_data.part_dimensions.area,
        })()

    # ========================================================================
    # STEP 3: Calculate direct costs (McMaster stock, scrap, pricing)
    # ========================================================================
    if verbose:
        print("[3/5] Calculating direct costs...")

    # Use default catalog if not specified
    if catalog_csv_path is None:
        catalog_csv_path = str(default_catalog_csv())

    # Calculate desired stock dimensions (with machining allowances)
    # These are the actual dimensions needed for the starting stock
    desired_L = part_info.length + 0.50   # +0.50" face milling allowance
    desired_W = part_info.width + 0.50    # +0.50" face milling allowance
    desired_T = part_info.thickness + 0.25  # +0.25" thickness allowance

    # Calculate scrap info
    # Pass desired dimensions explicitly to ensure catalog lookup uses correct values
    # For cylindrical parts, pass diameter and is_cylindrical flag
    is_cylindrical = quote_data.part_dimensions.is_cylindrical if quote_data.part_dimensions else False
    part_diameter = quote_data.part_dimensions.diameter if quote_data.part_dimensions else None
    part_diameter_2 = quote_data.part_dimensions.diameter_2 if quote_data.part_dimensions else None

    # For tapered parts, use the larger diameter for stock lookup
    if is_cylindrical and part_diameter_2 and part_diameter_2 > part_diameter:
        stock_diameter = part_diameter_2
    else:
        stock_diameter = part_diameter

    if verbose and is_cylindrical:
        print(f"  DEBUG [Scrap calc inputs]:")
        print(f"    part_diameter={part_diameter}, part_diameter_2={part_diameter_2}, stock_diameter={stock_diameter}")
        print(f"    part_length={part_info.length:.3f}, desired_length={desired_L:.3f}")

    scrap_calc = calculate_total_scrap(
        cad_file_path=cad_file_path,
        material=material,
        desired_length=desired_L,
        desired_width=desired_W,
        desired_thickness=desired_T,
        part_length=part_info.length,
        part_width=part_info.width,
        part_thickness=part_info.thickness,
        catalog_csv_path=catalog_csv_path,
        is_cylindrical=is_cylindrical,
        part_diameter=stock_diameter,
        verbose=verbose
    )

    # Calculate scrap value
    scrap_value_calc = calculate_scrap_value(
        scrap_weight_lbs=scrap_calc.total_scrap_weight,
        material=material,
        fallback_scrap_price_per_lb=0.50,  # $0.50/lb default scrap rate
        verbose=verbose
    )

    # Apply scrap value override if provided
    if scrap_value_override is not None:
        scrap_value_calc['scrap_value'] = scrap_value_override
        if verbose:
            print(f"  Using manual scrap value override: ${scrap_value_override:.2f}")

    # Get McMaster pricing
    # Note: McMaster stock dimensions are already calculated in scrap_calc from calculate_total_scrap()
    # We just need to get the part number for pricing lookup
    catalog_rows = load_mcmaster_catalog_rows(catalog_csv_path)

    # Check if part is cylindrical (guide posts, spring pins, etc.)
    is_cylindrical = quote_data.part_dimensions.is_cylindrical if quote_data.part_dimensions else False
    part_diameter = quote_data.part_dimensions.diameter if quote_data.part_dimensions else 0.0
    part_diameter_2 = quote_data.part_dimensions.diameter_2 if quote_data.part_dimensions else 0.0
    part_length = quote_data.part_dimensions.length if quote_data.part_dimensions else 0.0

    # Initialize desired_diameter (will be calculated for cylindrical parts)
    desired_diameter_with_allowance = 0.0

    if is_cylindrical and part_diameter > 0 and part_length > 0:
        # For tapered parts, use the larger diameter for stock selection
        stock_diameter = max(part_diameter, part_diameter_2) if part_diameter_2 > 0 else part_diameter

        # Add machining allowances for cylindrical parts (similar to plate stock)
        # - Diameter needs extra material for turning/grinding (like thickness allowance)
        # - Length needs extra material for facing and holding (like length allowance)
        DIAMETER_ALLOWANCE = 0.25  # +0.25" for turning/grinding (matches thickness allowance)
        LENGTH_ALLOWANCE = 0.50    # +0.50" for facing/holding (matches length allowance)

        desired_diameter_with_allowance = stock_diameter + DIAMETER_ALLOWANCE
        desired_cylindrical_length = part_length + LENGTH_ALLOWANCE

        # Use cylindrical lookup for guide posts, spring pins, etc.
        if verbose:
            if part_diameter_2 > 0:
                print(f"  [CYLINDRICAL] Tapered part detected (diam1={part_diameter:.3f}\", diam2={part_diameter_2:.3f}\")")
                print(f"  [CYLINDRICAL] Using larger diameter for stock lookup: {stock_diameter:.3f}\" → {desired_diameter_with_allowance:.3f}\" (with {DIAMETER_ALLOWANCE}\" allowance)")
                print(f"  [CYLINDRICAL] Length for stock lookup: {part_length:.3f}\" → {desired_cylindrical_length:.3f}\" (with {LENGTH_ALLOWANCE}\" allowance)")
            else:
                print(f"  [CYLINDRICAL] Using cylindrical stock lookup")
                print(f"  [CYLINDRICAL]   Diameter: {stock_diameter:.3f}\" → {desired_diameter_with_allowance:.3f}\" (with {DIAMETER_ALLOWANCE}\" allowance)")
                print(f"  [CYLINDRICAL]   Length: {part_length:.3f}\" → {desired_cylindrical_length:.3f}\" (with {LENGTH_ALLOWANCE}\" allowance)")

        mcmaster_result = pick_mcmaster_cylindrical_sku(
            need_diam_in=desired_diameter_with_allowance,
            need_length_in=desired_cylindrical_length,
            material_key=material,
            catalog_rows=catalog_rows,
            verbose=verbose
        )
    else:
        # Use standard plate lookup
        mcmaster_result = pick_mcmaster_plate_sku(
            need_L_in=desired_L,
            need_W_in=desired_W,
            need_T_in=desired_T,
            material_key=material,
            catalog_rows=catalog_rows
        )

    mcmaster_part_num = None
    mcmaster_price = None
    price_is_estimated = False

    # Use price override if provided, otherwise lookup from McMaster
    if mcmaster_price_override is not None:
        mcmaster_price = mcmaster_price_override
        if verbose:
            print(f"  Using manual McMaster price override: ${mcmaster_price:.2f}")
    elif mcmaster_result:
        mcmaster_part_num = mcmaster_result.get('mcmaster_part')
        if mcmaster_part_num:
            # Check if catalog volume is more than 2x required stock (high scrap scenario)
            catalog_vol = scrap_calc.mcmaster_volume
            required_vol = scrap_calc.desired_volume
            use_volume_pricing = required_vol > 0 and catalog_vol > 2 * required_vol

            if use_volume_pricing:
                # Use volume-based pricing when catalog is much larger than required stock
                if verbose:
                    print(f"  Catalog volume ({catalog_vol:.2f} in³) > 2x required stock ({required_vol:.2f} in³)")
                    print(f"  Using volume-based price estimation for better accuracy...")

                from cad_quoter.pricing.DirectCostHelper import estimate_price_from_reference_part

                # Map material to McMaster catalog key for price estimation
                mcmaster_material = material_mapper.get_mcmaster_key(material) or material
                mcmaster_price = estimate_price_from_reference_part(
                    target_length=scrap_calc.desired_length,
                    target_width=scrap_calc.desired_width,
                    target_thickness=scrap_calc.desired_thickness,
                    material=mcmaster_material,
                    catalog_csv_path=catalog_csv_path,
                    verbose=verbose
                )

                if mcmaster_price:
                    price_is_estimated = True
                    if verbose:
                        print(f"  ✓ Estimated price from volume: ${mcmaster_price:.2f}")
            else:
                # Normal flow: try direct pricing first
                mcmaster_price = get_mcmaster_price(mcmaster_part_num, quantity=quantity)

                # If direct pricing failed, try volume-based estimation
                if mcmaster_price is None:
                    if verbose:
                        print(f"  No direct pricing available for {mcmaster_part_num}")
                        print(f"  Attempting volume-based price estimation...")

                    from cad_quoter.pricing.DirectCostHelper import estimate_price_from_reference_part

                    # Map material to McMaster catalog key for price estimation
                    mcmaster_material = material_mapper.get_mcmaster_key(material) or material
                    mcmaster_price = estimate_price_from_reference_part(
                        target_length=scrap_calc.mcmaster_length,
                        target_width=scrap_calc.mcmaster_width,
                        target_thickness=scrap_calc.mcmaster_thickness,
                        material=mcmaster_material,
                        catalog_csv_path=catalog_csv_path,
                        verbose=verbose
                    )

                    if mcmaster_price:
                        price_is_estimated = True
                        if verbose:
                            print(f"  ✓ Estimated price from volume: ${mcmaster_price:.2f}")

    # If we still don't have a price and no part was found, try volume-based estimation
    if mcmaster_price is None and not mcmaster_part_num:
        catalog_vol = scrap_calc.mcmaster_volume
        required_vol = scrap_calc.desired_volume
        use_required_stock = required_vol > 0 and catalog_vol > 2 * required_vol

        if verbose:
            print(f"  No McMaster part found in catalog")
            if use_required_stock:
                print(f"  Catalog volume ({catalog_vol:.2f} in³) > 2x required stock ({required_vol:.2f} in³)")
                print(f"  Using volume-based price estimation on required stock...")
            else:
                print(f"  Attempting volume-based price estimation...")

        from cad_quoter.pricing.DirectCostHelper import estimate_price_from_reference_part

        # Map material to McMaster catalog key for price estimation
        mcmaster_material = material_mapper.get_mcmaster_key(material) or material

        # Use required stock dimensions if catalog is much larger, otherwise use catalog dimensions
        if use_required_stock:
            target_length = scrap_calc.desired_length
            target_width = scrap_calc.desired_width
            target_thickness = scrap_calc.desired_thickness
        else:
            target_length = scrap_calc.mcmaster_length
            target_width = scrap_calc.mcmaster_width
            target_thickness = scrap_calc.mcmaster_thickness

        mcmaster_price = estimate_price_from_reference_part(
            target_length=target_length,
            target_width=target_width,
            target_thickness=target_thickness,
            material=mcmaster_material,
            catalog_csv_path=catalog_csv_path,
            verbose=verbose
        )

        if mcmaster_price:
            price_is_estimated = True
            if verbose:
                print(f"  ✓ Estimated price from volume: ${mcmaster_price:.2f}")

    # If volume-based estimation also failed, try weight-based estimation using largest catalog part
    if mcmaster_price is None:
        if verbose:
            print(f"  Volume-based estimation failed")
            print(f"  Attempting weight-based price estimation from largest catalog part...")

        from cad_quoter.pricing.mcmaster_helpers import estimate_price_from_catalog_reference, load_mcmaster_catalog_rows

        # Load catalog rows
        catalog_rows = load_mcmaster_catalog_rows(path=catalog_csv_path)
        if verbose:
            print(f"  Loaded {len(catalog_rows)} catalog rows from {catalog_csv_path}")

        # Get material density
        density_lb_in3 = material_mapper.get_density_lb_in3(material)
        if density_lb_in3 is None or density_lb_in3 <= 0:
            density_lb_in3 = 0.10  # Default to aluminum density

        # Map material to McMaster catalog key
        mcmaster_material = material_mapper.get_mcmaster_key(material) or material

        # Estimate price based on weight
        estimated_price, source = estimate_price_from_catalog_reference(
            material_key=mcmaster_material,
            weight_lb=scrap_calc.mcmaster_weight,
            density_lb_in3=density_lb_in3,
            catalog_rows=catalog_rows,
            verbose=verbose
        )

        if estimated_price and estimated_price > 0:
            mcmaster_price = estimated_price
            price_is_estimated = True
            if verbose:
                print(f"  ✓ Estimated price from weight: ${mcmaster_price:.2f}")
                print(f"     ({source})")

    # Final fallback: if all estimation methods failed, use default price per lb × weight
    if mcmaster_price is None:
        if verbose:
            print(f"  All price estimation methods failed")
            print(f"  Using fallback: default price per lb × stock weight...")

        from cad_quoter.pricing.materials import resolve_material_unit_price

        # Get fallback price per lb for this material
        fallback_price_per_lb, price_source = resolve_material_unit_price(material, unit="lb")

        # Calculate price based on stock weight
        stock_weight = scrap_calc.mcmaster_weight
        if stock_weight and stock_weight > 0:
            # Apply a factor to account for processing costs and typical markup
            WEIGHT_PRICE_FACTOR = 1.5  # 50% markup over raw material cost
            mcmaster_price = round(stock_weight * fallback_price_per_lb * WEIGHT_PRICE_FACTOR, 2)
            price_is_estimated = True
            if verbose:
                print(f"  ✓ Fallback price: ${mcmaster_price:.2f}")
                print(f"     (${fallback_price_per_lb:.2f}/lb × {stock_weight:.2f} lb × {WEIGHT_PRICE_FACTOR} factor)")
                print(f"     (source: {price_source})")

    # Populate stock info
    # Use McMaster dimensions from scrap_calc (which already did the catalog lookup)
    # For cylindrical parts, also include diameter
    if is_cylindrical:
        # Use the desired diameter with allowance (calculated earlier)
        desired_diameter = desired_diameter_with_allowance
        mcmaster_diameter = mcmaster_result.get('stock_diam_in', 0.0) if mcmaster_result else 0.0
    else:
        desired_diameter = 0.0
        mcmaster_diameter = 0.0

    quote_data.stock_info = StockInfo(
        desired_length=desired_L,
        desired_width=desired_W,
        desired_thickness=desired_T,
        desired_diameter=desired_diameter,
        desired_volume=desired_L * desired_W * desired_T,
        mcmaster_length=scrap_calc.mcmaster_length,
        mcmaster_width=scrap_calc.mcmaster_width,
        mcmaster_thickness=scrap_calc.mcmaster_thickness,
        mcmaster_diameter=mcmaster_diameter,
        mcmaster_volume=scrap_calc.mcmaster_length * scrap_calc.mcmaster_width * scrap_calc.mcmaster_thickness,
        mcmaster_part_number=mcmaster_part_num,
        mcmaster_price=mcmaster_price,
        price_is_estimated=price_is_estimated,
        mcmaster_weight=scrap_calc.mcmaster_weight,
        final_part_weight=scrap_calc.final_part_weight
    )

    # Stock size sanity guardrails
    # Check that stock dimensions are reasonable relative to part dimensions
    stock_thickness_ratio = scrap_calc.mcmaster_thickness / part_info.thickness if part_info.thickness > 0 else 0
    part_volume = part_info.length * part_info.width * part_info.thickness
    stock_volume = scrap_calc.mcmaster_length * scrap_calc.mcmaster_width * scrap_calc.mcmaster_thickness
    stock_volume_ratio = stock_volume / part_volume if part_volume > 0 else 0

    # Warning thresholds
    MAX_THICKNESS_RATIO = 3.0  # Stock thickness should be at most 3× part thickness
    MAX_VOLUME_RATIO = 5.0     # Stock volume should be at most 5× part volume

    stock_size_warnings = []
    if stock_thickness_ratio > MAX_THICKNESS_RATIO:
        stock_size_warnings.append(
            f"Stock thickness ({scrap_calc.mcmaster_thickness:.3f}\") is {stock_thickness_ratio:.1f}× "
            f"the part thickness ({part_info.thickness:.3f}\"). "
            f"Consider using plate stock rules for thin parts."
        )
    if stock_volume_ratio > MAX_VOLUME_RATIO:
        stock_size_warnings.append(
            f"Stock volume ({stock_volume:.2f} in³) is {stock_volume_ratio:.1f}× "
            f"the part volume ({part_volume:.2f} in³). "
            f"Stock may be oversized for this part geometry."
        )

    if stock_size_warnings:
        import logging
        for warning in stock_size_warnings:
            logging.warning(f"Stock size check: {warning}")
        if verbose:
            for warning in stock_size_warnings:
                print(f"  WARNING: {warning}")

    # Populate scrap info
    scrap_pct = scrap_calc.scrap_percentage
    quote_data.scrap_info = ScrapInfo(
        stock_prep_scrap=scrap_calc.stock_prep_scrap,
        face_milling_scrap=scrap_calc.face_milling_scrap,
        hole_drilling_scrap=scrap_calc.hole_drilling_scrap,
        total_scrap_volume=scrap_calc.total_scrap_volume,
        total_scrap_weight=scrap_calc.total_scrap_weight,
        scrap_percentage=scrap_pct,
        utilization_percentage=scrap_calc.utilization_percentage,
        scrap_price_per_lb=scrap_value_calc.get('scrap_price_per_lb'),
        scrap_value=scrap_value_calc.get('scrap_value', 0.0),
        scrap_price_source=scrap_value_calc.get('price_source', ''),
        high_scrap_warning=scrap_pct > HIGH_SCRAP_THRESHOLD
    )

    # Calculate direct cost breakdown
    # NOTE: Keep total_material_cost in sync with printed line items (avoid 1¢ drift).
    # Round each component to 2 decimal places before summing to ensure the total
    # matches the sum of the displayed line items exactly.
    if mcmaster_price:
        tax = round(mcmaster_price * 0.07, 2)
        # Use weight-based shipping estimator instead of percentage
        # IMPORTANT: Calculate shipping for total weight (all pieces), not per-item weight
        stock_weight = scrap_calc.mcmaster_weight if scrap_calc.mcmaster_weight else 0.0
        total_weight = stock_weight * quantity
        shipping = round(estimate_mcmaster_shipping(total_weight), 2)
        scrap_credit = round(scrap_value_calc.get('scrap_value', 0.0), 2)
        net_cost = max(0, round(mcmaster_price + tax + shipping - scrap_credit, 2))
    else:
        tax = shipping = scrap_credit = net_cost = 0.0

    quote_data.direct_cost_breakdown = DirectCostBreakdown(
        stock_cost=mcmaster_price or 0.0,
        tax=tax,
        shipping=shipping,
        scrap_credit=scrap_credit,
        net_material_cost=net_cost
    )

    if verbose:
        print(f"  McMaster: {mcmaster_part_num or 'N/A'}")
        print(f"  Stock price: ${mcmaster_price:.2f}" if mcmaster_price else "  Stock price: N/A")
        print(f"  Net material cost: ${net_cost:.2f}")

    # ========================================================================
    # STEP 4: Calculate machine hours
    # ========================================================================
    if verbose:
        print("[4/5] Calculating machine hours...")

    # Always process hole operations - both punch and non-punch parts need hole times
    # Extract hole operations from CAD file
    hole_table = extract_hole_operations_from_cad(cad_file_path)

    # For punch parts, process holes and add times to punch base times
    if is_punch and punch_data and "error" not in punch_data:
        if verbose:
            print(f"  [PUNCH] Base punch times: {quote_data.machine_hours.total_hours:.2f} hr")

        # Get base punch times
        punch_base_milling = quote_data.machine_hours.total_milling_minutes
        punch_base_grinding = quote_data.machine_hours.total_grinding_minutes
        punch_base_drill = quote_data.machine_hours.total_drill_minutes
        punch_base_tap = quote_data.machine_hours.total_tap_minutes
        punch_base_edm = quote_data.machine_hours.total_edm_minutes
        punch_base_other = quote_data.machine_hours.total_other_minutes
        punch_base_cmm = quote_data.machine_hours.total_cmm_minutes
        punch_base_inspection = quote_data.machine_hours.total_inspection_minutes

        # Process hole table for punch parts
        hole_entries = len(hole_table) if hole_table else 0
        holes_total = sum(int(hole.get('QTY', 1)) for hole in hole_table) if hole_table else 0

        # Initialize hole operation accumulators
        hole_drill_min = 0.0
        hole_tap_min = 0.0
        hole_cbore_min = 0.0
        hole_cdrill_min = 0.0
        hole_jig_grind_min = 0.0
        hole_edm_min = 0.0

        drill_ops = []
        tap_ops = []
        cbore_ops = []
        cdrill_ops = []
        jig_grind_ops = []
        edm_ops = []

        if hole_table:
            times = estimate_hole_table_times(hole_table, material, part_info.thickness)

            # Convert to HoleOperation objects
            drill_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='drill',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time'],
                    sfm=g.get('sfm'),
                    ipr=g.get('ipr')
                )
                for g in times.get('drill_groups', [])
            ]

            tap_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='tap',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time'],
                    sfm=g.get('sfm'),
                    tpi=g.get('tpi')
                )
                for g in times.get('tap_groups', [])
            ]

            cbore_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='cbore',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time'],
                    sfm=g.get('sfm'),
                    side=g.get('side')
                )
                for g in times.get('cbore_groups', [])
            ]

            cdrill_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='cdrill',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time']
                )
                for g in times.get('cdrill_groups', [])
            ]

            jig_grind_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='jig_grind',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time']
                )
                for g in times.get('jig_grind_groups', [])
            ]

            edm_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='edm',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time']
                )
                for g in times.get('edm_groups', [])
            ]

            # Get hole operation times
            hole_drill_min = times.get('total_drill_minutes', 0.0)
            hole_tap_min = times.get('total_tap_minutes', 0.0)
            hole_cbore_min = times.get('total_cbore_minutes', 0.0)
            hole_cdrill_min = times.get('total_cdrill_minutes', 0.0)
            hole_jig_grind_min = times.get('total_jig_grind_minutes', 0.0)
            hole_edm_min = times.get('total_edm_minutes', 0.0)

        # Merge hole times with punch base times
        total_drill_min = round(punch_base_drill + hole_drill_min, 2)
        total_tap_min = round(punch_base_tap + hole_tap_min, 2)
        total_cbore_min = round(hole_cbore_min, 2)
        total_cdrill_min = round(hole_cdrill_min, 2)
        total_jig_grind_min = round(hole_jig_grind_min, 2)
        total_milling_min = round(punch_base_milling, 2)
        total_grinding_min = round(punch_base_grinding, 2)
        total_edm_min = round(punch_base_edm + hole_edm_min, 2)
        total_other_min = round(punch_base_other, 2)
        total_cmm_min = round(punch_base_cmm, 2)
        total_inspection_min = round(punch_base_inspection, 2)

        # Extract special operation times from base punch hours
        total_etch_min = round(quote_data.machine_hours.total_etch_minutes, 2)
        total_edge_break_min = round(quote_data.machine_hours.total_edge_break_minutes, 2)
        total_polish_min = round(quote_data.machine_hours.total_polish_minutes, 2)

        # Calculate updated grand total (include special operations)
        grand_total_minutes = round(
            total_drill_min + total_tap_min + total_cbore_min +
            total_cdrill_min + total_jig_grind_min +
            total_milling_min + total_grinding_min + total_edm_min + total_other_min +
            total_cmm_min + total_inspection_min + total_etch_min + total_edge_break_min + total_polish_min, 2
        )
        grand_total_hours = round(grand_total_minutes / 60.0, 2)
        machine_cost = round(grand_total_minutes * (machine_rate / 60.0), 2)

        # Update machine hours with merged times
        quote_data.machine_hours = MachineHoursBreakdown(
            drill_operations=drill_ops,
            tap_operations=tap_ops,
            cbore_operations=cbore_ops,
            cdrill_operations=cdrill_ops,
            jig_grind_operations=jig_grind_ops,
            edm_operations=edm_ops,
            milling_operations=[],  # Punch uses turning, not explicit milling ops
            grinding_operations=[],  # Punch grinding handled differently
            total_drill_minutes=total_drill_min,
            total_tap_minutes=total_tap_min,
            total_cbore_minutes=total_cbore_min,
            total_cdrill_minutes=total_cdrill_min,
            total_jig_grind_minutes=total_jig_grind_min,
            total_milling_minutes=total_milling_min,
            total_grinding_minutes=total_grinding_min,
            total_edm_minutes=total_edm_min,
            total_other_minutes=total_other_min,
            total_cmm_minutes=total_cmm_min,
            total_inspection_minutes=total_inspection_min,
            total_etch_minutes=total_etch_min,
            total_edge_break_minutes=total_edge_break_min,
            total_polish_minutes=total_polish_min,
            cmm_holes_checked=holes_total,
            holes_total=holes_total,
            hole_entries=hole_entries,
            total_minutes=grand_total_minutes,
            total_hours=grand_total_hours,
            machine_cost=machine_cost
        )

        if verbose:
            print(f"  [PUNCH] Updated with hole operations:")
            print(f"    - Drilling: {total_drill_min:.1f} min (base: {punch_base_drill:.1f}, holes: {hole_drill_min:.1f})")
            print(f"    - Tapping: {total_tap_min:.1f} min")
            print(f"    - EDM: {total_edm_min:.1f} min (base: {punch_base_edm:.1f}, holes: {hole_edm_min:.1f})")
            print(f"  [PUNCH] Total machine hours: {grand_total_hours:.2f} hr")
            print(f"  Machine cost: ${machine_cost:.2f}")

    else:
        # Non-punch parts: Calculate hole counts
        # hole_entries = count of unique hole groups (A, B, C, etc.)
        # holes_total = sum of QTY field from each hole entry (total individual holes)
        # Note: hole_table may contain multiple operations per hole group (e.g., drill + tap + cbore)
        # so we count unique hole letters rather than operation count
        if hole_table:
            unique_holes = set(hole.get('HOLE', '') for hole in hole_table)
            hole_entries = len(unique_holes)
            # For holes_total, sum QTY only once per unique hole group
            # Group by hole letter and take first QTY value
            qty_by_hole = {}
            for hole in hole_table:
                hole_letter = hole.get('HOLE', '')
                if hole_letter not in qty_by_hole:
                    qty_by_hole[hole_letter] = int(hole.get('QTY', 1))
            holes_total = sum(qty_by_hole.values())
        else:
            hole_entries = 0
            holes_total = 0

        # Initialize time accumulators
        total_drill_min = 0.0
        total_tap_min = 0.0
        total_cbore_min = 0.0
        total_cdrill_min = 0.0
        total_jig_grind_min = 0.0
        total_milling_min = 0.0
        total_grinding_min = 0.0
        total_edm_min = 0.0
        total_other_min = 0.0

        drill_ops = []
        tap_ops = []
        cbore_ops = []
        cdrill_ops = []
        jig_grind_ops = []
        edm_ops = []

        if hole_table:
            times = estimate_hole_table_times(hole_table, material, part_info.thickness)

            # Convert to HoleOperation objects
            drill_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='drill',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time'],
                    sfm=g.get('sfm'),
                    ipr=g.get('ipr')
                )
                for g in times.get('drill_groups', [])
            ]

            tap_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='tap',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time'],
                    sfm=g.get('sfm'),
                    tpi=g.get('tpi')
                )
                for g in times.get('tap_groups', [])
            ]

            cbore_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='cbore',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time'],
                    sfm=g.get('sfm'),
                    side=g.get('side')
                )
                for g in times.get('cbore_groups', [])
            ]

            cdrill_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='cdrill',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time']
                )
                for g in times.get('cdrill_groups', [])
            ]

            jig_grind_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],
                    qty=g['qty'],
                    operation_type='jig_grind',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time']
                )
                for g in times.get('jig_grind_groups', [])
            ]

            edm_ops = [
                HoleOperation(
                    hole_id=g['hole_id'],
                    diameter=g['diameter'],
                    depth=g['depth'],  # This is the thickness for EDM
                    qty=g['qty'],
                    operation_type='edm',
                    time_per_hole=g['time_per_hole'],
                    total_time=g['total_time']
                )
                for g in times.get('edm_groups', [])
            ]

            # Accumulate hole operation times
            total_drill_min = times.get('total_drill_minutes', 0.0)
            total_tap_min = times.get('total_tap_minutes', 0.0)
            total_cbore_min = times.get('total_cbore_minutes', 0.0)
            total_cdrill_min = times.get('total_cdrill_minutes', 0.0)
            total_jig_grind_min = times.get('total_jig_grind_minutes', 0.0)
            # EDM time from "FOR WIRE EDM" holes (starter holes for wire EDM operations)
            hole_table_edm_min = times.get('total_edm_minutes', 0.0)
            # Slot milling time (obround features)
            slot_milling_min = times.get('total_slot_minutes', 0.0)
        else:
            hole_table_edm_min = 0.0
            slot_milling_min = 0.0

        # Calculate times for plan operations (squaring, face milling, EDM, etc.)
        # Pass both final part thickness and McMaster stock thickness for proper removal calculations
        plan_machine_times = estimate_machine_hours_from_plan(
            plan,
            material=material,
            plate_LxW=(part_info.length, part_info.width),
            thickness=part_info.thickness,
            stock_thickness=scrap_calc.mcmaster_thickness if scrap_calc else part_info.thickness
        )

        # Add plan operation times to totals
        # Use sum of detailed operations for consistency with displayed breakdown
        milling_ops_raw = plan_machine_times.get('milling_operations', [])
        grinding_ops_raw = plan_machine_times.get('grinding_operations', [])
        pocket_ops_raw = plan_machine_times.get('pocket_operations', [])
        slot_ops_raw = plan_machine_times.get('slot_operations', [])
        waterjet_ops_raw = plan_machine_times.get('waterjet_operations', [])  # NEW: Waterjet ops

        # Calculate totals from detailed operations (what's displayed in report)
        total_milling_ops_min = sum(op.get('time_minutes', 0.0) for op in milling_ops_raw)
        total_grinding_ops_min = sum(op.get('time_minutes', 0.0) for op in grinding_ops_raw)
        total_pocket_ops_min = sum(op.get('pocket_time_min', 0.0) for op in pocket_ops_raw)
        total_slot_ops_min = sum(op.get('slot_mill_time_min', 0.0) for op in slot_ops_raw)
        total_waterjet_ops_min = sum(op.get('time_min', 0.0) for op in waterjet_ops_raw)  # NEW

        # Get breakdown totals (may include non-detailed operations)
        breakdown_milling_min = plan_machine_times['breakdown_minutes'].get('milling', 0.0)
        breakdown_grinding_min = plan_machine_times['breakdown_minutes'].get('grinding', 0.0)
        breakdown_pocket_min = plan_machine_times['breakdown_minutes'].get('pockets', 0.0)
        breakdown_slot_min = plan_machine_times['breakdown_minutes'].get('slots', 0.0)
        breakdown_waterjet_min = plan_machine_times['breakdown_minutes'].get('waterjet', 0.0)  # NEW
        # EDM from plan operations + EDM from hole table "FOR WIRE EDM" entries
        plan_edm_min = plan_machine_times['breakdown_minutes'].get('edm', 0.0)
        total_edm_min = plan_edm_min + hole_table_edm_min
        if verbose:
            print(f"[DEBUG EDM] plan_edm_min={plan_edm_min:.2f}, hole_table_edm_min={hole_table_edm_min:.2f}, total_edm_min={total_edm_min:.2f}")

        # Get other_ops_detail from plan (NEW)
        other_ops_detail_raw = plan_machine_times.get('other_ops_detail', [])
        total_other_min = plan_machine_times.get('other_ops_minutes', 0.0)

        # Any milling/grinding/pocket/slot time not in detailed ops goes to "other" for transparency
        milling_overhead_min = breakdown_milling_min - total_milling_ops_min
        grinding_overhead_min = breakdown_grinding_min - total_grinding_ops_min
        pocket_overhead_min = breakdown_pocket_min - total_pocket_ops_min
        slot_overhead_min = breakdown_slot_min - total_slot_ops_min

        # Add overflow to other_ops_detail if non-zero
        overflow_total = milling_overhead_min + grinding_overhead_min + pocket_overhead_min + slot_overhead_min
        if overflow_total > 0.01:
            other_ops_detail_raw.append({
                "type": "overflow_from_milling",
                "label": "Other mill/grind/pocket/slot ops",
                "minutes": round(overflow_total, 1),
                "source": "overflow"
            })
        total_other_min += overflow_total

        # Use detailed ops totals for display (ensures Total Milling Time matches ops sum)
        # Add slot milling time from hole table to milling totals (legacy slot handling)
        total_milling_min = total_milling_ops_min + slot_milling_min
        total_grinding_min = total_grinding_ops_min
        total_pocket_min = total_pocket_ops_min
        total_slot_min = total_slot_ops_min
        total_waterjet_min = total_waterjet_ops_min  # NEW

        # Sanity check: totals should match (total_milling_min includes slot time + plan ops)
        expected_milling = total_milling_ops_min + slot_milling_min
        assert abs(total_milling_min - expected_milling) < 0.01, \
            f"Milling time mismatch: {total_milling_min:.2f} vs expected {expected_milling:.2f}"

        # Convert detailed operations to dataclass objects
        # Filter out extra fields that aren't part of the dataclass definitions
        milling_field_names = {f.name for f in fields(MillingOperation)}
        milling_ops = [
            MillingOperation(**{k: v for k, v in op.items() if k in milling_field_names})
            for op in milling_ops_raw
        ]

        grinding_field_names = {f.name for f in fields(GrindingOperation)}
        grinding_ops = [
            GrindingOperation(**{k: v for k, v in op.items() if k in grinding_field_names})
            for op in grinding_ops_raw
        ]

        pocket_field_names = {f.name for f in fields(PocketOperation)}
        pocket_ops = [
            PocketOperation(**{k: v for k, v in op.items() if k in pocket_field_names})
            for op in pocket_ops_raw
        ]

        slot_field_names = {f.name for f in fields(SlotOperation)}
        slot_ops = [
            SlotOperation(**{k: v for k, v in op.items() if k in slot_field_names})
            for op in slot_ops_raw
        ]

        # Waterjet operations are already in the correct format (no dataclass filtering needed)
        waterjet_ops = waterjet_ops_raw  # NEW

        # Calculate CMM inspection time (split between labor setup and machine checking)
        # Use override if provided, otherwise use inspection_level from plan, or default to "full_inspection"
        from cad_quoter.planning.process_planner import cmm_inspection_minutes
        if cmm_inspection_level_override:
            inspection_level = cmm_inspection_level_override
        else:
            inspection_level = plan.get('inspection_level', 'full_inspection')
        cmm_breakdown = cmm_inspection_minutes(holes_total, inspection_level=inspection_level)
        cmm_setup_labor_min = cmm_breakdown['setup_labor_min']
        cmm_checking_machine_min = cmm_breakdown['checking_machine_min']
        cmm_holes_checked = cmm_breakdown['holes_checked']

        # Calculate grand totals (machine time only includes CMM checking, not setup)
        # Round each component to 2 decimal places before summing to ensure the total
        # matches the sum of the displayed line items exactly.
        total_drill_min = round(total_drill_min, 2)
        total_tap_min = round(total_tap_min, 2)
        total_cbore_min = round(total_cbore_min, 2)
        total_cdrill_min = round(total_cdrill_min, 2)
        total_jig_grind_min = round(total_jig_grind_min, 2)
        total_milling_min = round(total_milling_min, 2)
        total_grinding_min = round(total_grinding_min, 2)
        total_pocket_min = round(total_pocket_min, 2)
        total_slot_min = round(total_slot_min, 2)
        total_edm_min = round(total_edm_min, 2)
        total_other_min = round(total_other_min, 2)
        total_waterjet_min = round(total_waterjet_min, 2)  # NEW
        cmm_checking_machine_min = round(cmm_checking_machine_min, 2)

        # Extract special operation times (edge break, etch, polish)
        total_edge_break_min = round(plan_machine_times.get('edge_break_minutes', 0.0), 2)
        total_etch_min = round(plan_machine_times.get('etch_minutes', 0.0), 2)
        total_polish_min = round(plan_machine_times.get('polish_minutes', 0.0), 2)

        grand_total_minutes = round(
            total_drill_min + total_tap_min + total_cbore_min +
            total_cdrill_min + total_jig_grind_min +
            total_milling_min + total_grinding_min + total_pocket_min + total_slot_min +
            total_edm_min + total_other_min + total_waterjet_min +  # NEW: Added waterjet
            total_edge_break_min + total_etch_min + total_polish_min +
            cmm_checking_machine_min, 2
        )
        grand_total_hours = round(grand_total_minutes / 60.0, 2)

        # Compute machine cost directly from total minutes for accuracy
        # (avoids rounding errors from hours conversion)
        machine_cost = round(grand_total_minutes * (machine_rate / 60.0), 2)

        quote_data.machine_hours = MachineHoursBreakdown(
            drill_operations=drill_ops,
            tap_operations=tap_ops,
            cbore_operations=cbore_ops,
            cdrill_operations=cdrill_ops,
            jig_grind_operations=jig_grind_ops,
            edm_operations=edm_ops,
            milling_operations=milling_ops,
            grinding_operations=grinding_ops,
            pocket_operations=pocket_ops,
            slot_operations=slot_ops,
            waterjet_operations=waterjet_ops,  # NEW
            total_drill_minutes=total_drill_min,
            total_tap_minutes=total_tap_min,
            total_cbore_minutes=total_cbore_min,
            total_cdrill_minutes=total_cdrill_min,
            total_jig_grind_minutes=total_jig_grind_min,
            total_edge_break_minutes=total_edge_break_min,
            total_etch_minutes=total_etch_min,
            total_polish_minutes=total_polish_min,
            total_milling_minutes=total_milling_min,
            total_grinding_minutes=total_grinding_min,
            total_pocket_minutes=total_pocket_min,
            total_slot_minutes=total_slot_min,
            total_edm_minutes=total_edm_min,
            total_other_minutes=total_other_min,
            total_waterjet_minutes=total_waterjet_min,  # NEW
            other_ops_detail=other_ops_detail_raw,  # NEW
            total_cmm_minutes=cmm_checking_machine_min,
            cmm_holes_checked=cmm_holes_checked,
            holes_total=holes_total,
            hole_entries=hole_entries,
            total_minutes=grand_total_minutes,
            total_hours=grand_total_hours,
            machine_cost=machine_cost
        )

        # Sanity check: warn if milling/overhead time is disproportionately high for small jobs
        primary_ops_min = total_drill_min + total_grinding_min + total_jig_grind_min + total_tap_min
        overhead_ops_min = total_milling_min + total_cbore_min + total_edm_min + total_other_min

        if primary_ops_min < 5.0 and overhead_ops_min > primary_ops_min and overhead_ops_min > 1.0:
            import logging
            logging.info(
                f"Machine time check: primary ops={primary_ops_min:.2f} min, "
                f"overhead ops (milling/cbore/edm/other)={overhead_ops_min:.2f} min. "
                f"High overhead ratio for small job - verify milling time is correct."
            )

        if verbose:
            print(f"  Machine hours: {grand_total_hours:.2f} hr")
            print(f"    - Drilling: {total_drill_min:.1f} min")
            print(f"    - Tapping: {total_tap_min:.1f} min")
            print(f"    - Milling (inc. squaring): {total_milling_min:.1f} min")
            print(f"    - Grinding (inc. wet grind): {total_grinding_min:.1f} min")
            print(f"    - EDM: {total_edm_min:.1f} min")
            print(f"    - CMM Inspection (checking only): {cmm_checking_machine_min:.1f} min")
            print(f"  Machine cost: ${quote_data.machine_hours.machine_cost:.2f}")

            # Also print the sanity check warning in verbose mode
            if primary_ops_min < 5.0 and overhead_ops_min > primary_ops_min and overhead_ops_min > 1.0:
                print(f"  WARNING: High overhead ratio - primary ops={primary_ops_min:.1f} min, "
                      f"overhead ops={overhead_ops_min:.1f} min")

        # ========================================================================
        # STEP 5: Calculate labor hours
        # ========================================================================
        if verbose:
            print("[5/5] Calculating labor hours...")

        ops = plan.get('ops', [])
        # Note: holes_total is already calculated earlier in STEP 4

        # Estimate labor inputs (simplified - could be more sophisticated)
        # Pass machine_time_minutes for simple-part setup guardrail
        # Pass plan data for finishing labor detail calculation
        labor_inputs = LaborInputs(
            ops_total=len(ops),
            holes_total=holes_total,
            tool_changes=len(ops) * 2,  # Rough estimate
            fixturing_complexity=1,
            cmm_setup_min=cmm_setup_labor_min,  # Add CMM setup to inspection labor
            cmm_holes_checked=cmm_holes_checked,  # Holes checked by CMM (avoid double counting)
            inspection_level=inspection_level,  # Inspection intensity knob
            net_weight_lb=quote_data.stock_info.final_part_weight,  # Part weight for handling bump
            machine_time_minutes=grand_total_minutes,  # For simple-part setup guardrail
            # Plan data for finishing labor detail
            plan=plan,
            ops=ops,
            part_length=quote_data.part_dimensions.length,
            part_width=quote_data.part_dimensions.width,
            part_thickness=quote_data.part_dimensions.thickness,
            material=material
        )

        labor_result = compute_labor_minutes(labor_inputs)
        minutes = labor_result['minutes']

        # Extract individual category minutes with rounding for consistent display
        setup_min = round(minutes.get('Setup', 0.0), 2)
        programming_min = round(minutes.get('Programming', 0.0), 2)
        machining_min = round(minutes.get('Machining_Steps', 0.0), 2)
        inspection_min = round(minutes.get('Inspection', 0.0), 2)
        finishing_min = round(minutes.get('Finishing', 0.0), 2)
        labor_total = round(minutes.get('Labor_Total', 0.0), 2)

        # Extract finishing detail breakdown
        finishing_breakdown = labor_result.get('finishing_breakdown', {})
        finishing_detail = finishing_breakdown.get('detail', [])

        # Calculate visible categories sum
        visible_labor_sum = (
            setup_min + programming_min + machining_min +
            inspection_min + finishing_min
        )

        # Calculate any misc overhead (should be 0 if compute_labor_minutes is consistent)
        misc_overhead_min = round(labor_total - visible_labor_sum, 2)

        # Sanity check: total should equal sum of categories + overhead
        assert abs(labor_total - (visible_labor_sum + misc_overhead_min)) < 0.01, \
            f"Labor time mismatch: total={labor_total:.2f}, sum={visible_labor_sum:.2f}, overhead={misc_overhead_min:.2f}"

        labor_total_hours = round(labor_total / 60.0, 2)
        # Compute labor cost directly from total minutes for accuracy
        # (avoids rounding errors from hours conversion)
        labor_cost = round(labor_total * (labor_rate / 60.0), 2)

        quote_data.labor_hours = LaborHoursBreakdown(
            setup_minutes=setup_min,
            programming_minutes=programming_min,
            machining_steps_minutes=machining_min,
            inspection_minutes=inspection_min,
            finishing_minutes=finishing_min,
            misc_overhead_minutes=misc_overhead_min,
            total_minutes=labor_total,
            total_hours=labor_total_hours,
            labor_cost=labor_cost,
            ops_total=len(ops),
            holes_total=holes_total,
            tool_changes=len(ops) * 2,
            finishing_detail=finishing_detail
        )

        if verbose:
            print(f"  Labor hours: {quote_data.labor_hours.total_hours:.2f} hr")
            print(f"  Labor cost: ${quote_data.labor_hours.labor_cost:.2f}")

    # ========================================================================
    # STEP 6: Calculate cost summary with quantity-aware amortization
    # ========================================================================

    # Calculate costs directly from rounded minutes for accuracy
    # Use the same formula as labor_hours.labor_cost to ensure consistency
    setup_labor = round(quote_data.labor_hours.setup_minutes * (labor_rate / 60.0), 2)
    programming_labor = round(quote_data.labor_hours.programming_minutes * (labor_rate / 60.0), 2)
    inspection_labor = round(quote_data.labor_hours.inspection_minutes * (labor_rate / 60.0), 2)

    # Job-level costs (setup, programming, first-article inspection) - amortized across quantity
    job_level_labor = setup_labor + programming_labor + inspection_labor
    amortized_job_level_cost = round(job_level_labor / quantity, 2)

    # Calculate variable costs per unit (material, machining, finishing)
    material_cost_per_unit = quote_data.direct_cost_breakdown.net_material_cost
    machine_cost_per_unit = quote_data.machine_hours.machine_cost

    # Variable labor costs per unit (machining, finishing only - inspection is job-level)
    # Round each component before summing for display consistency
    machining_labor = round(quote_data.labor_hours.machining_steps_minutes * (labor_rate / 60.0), 2)
    finishing_labor = round(quote_data.labor_hours.finishing_minutes * (labor_rate / 60.0), 2)
    misc_overhead_labor = round(quote_data.labor_hours.misc_overhead_minutes * (labor_rate / 60.0), 2)
    variable_labor_per_unit = machining_labor + finishing_labor + misc_overhead_labor

    # Per-unit costs
    # Round each component to 2 decimal places before summing to ensure the total
    # matches the sum of the displayed line items exactly.
    per_unit_direct_cost = round(material_cost_per_unit, 2)
    per_unit_machine_cost = round(machine_cost_per_unit, 2)
    per_unit_labor_cost = round(amortized_job_level_cost + variable_labor_per_unit, 2)
    per_unit_total_cost = round(per_unit_direct_cost + per_unit_machine_cost + per_unit_labor_cost, 2)

    # Total costs for all units
    # Use displayed (rounded) components to ensure total = sum of displayed parts
    total_direct_cost = round(per_unit_direct_cost * quantity, 2)
    total_machine_cost = round(per_unit_machine_cost * quantity, 2)
    total_labor_cost = round(job_level_labor + (variable_labor_per_unit * quantity), 2)
    total_total_cost = total_direct_cost + total_machine_cost + total_labor_cost

    # Margin and pricing
    per_unit_margin_amount = round(per_unit_total_cost * margin_rate, 2)
    per_unit_final_price = round(per_unit_total_cost + per_unit_margin_amount, 2)

    total_margin_amount = round(total_total_cost * margin_rate, 2)
    total_final_price = round(total_total_cost + total_margin_amount, 2)

    quote_data.cost_summary = CostSummary(
        # Per-unit costs
        direct_cost=per_unit_direct_cost,
        machine_cost=per_unit_machine_cost,
        labor_cost=per_unit_labor_cost,
        total_cost=per_unit_total_cost,
        # Total costs
        total_direct_cost=total_direct_cost,
        total_machine_cost=total_machine_cost,
        total_labor_cost=total_labor_cost,
        total_total_cost=total_total_cost,
        # Margin and pricing
        margin_rate=margin_rate,
        margin_amount=per_unit_margin_amount,
        final_price=per_unit_final_price,
        total_final_price=total_final_price
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*70}")
        if quantity > 1:
            print(f"Quantity: {quantity} parts")
            print(f"Per-unit cost: ${per_unit_total_cost:.2f}")
            print(f"Per-unit margin ({margin_rate:.0%}): ${per_unit_margin_amount:.2f}")
            print(f"Per-unit price: ${per_unit_final_price:.2f}")
            print(f"---")
            print(f"Total cost (all units): ${total_total_cost:.2f}")
            print(f"Total margin: ${total_margin_amount:.2f}")
            print(f"Total price: ${total_final_price:.2f}")
        else:
            print(f"Total cost: ${per_unit_total_cost:.2f}")
            print(f"Margin ({margin_rate:.0%}): ${per_unit_margin_amount:.2f}")
            print(f"Final price: ${per_unit_final_price:.2f}")
        print(f"{'='*70}\n")

    return quote_data


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def save_quote_data(
    quote_data: QuoteData,
    output_path: str | Path,
    pretty: bool = True
) -> None:
    """
    Save QuoteData to JSON file.

    Args:
        quote_data: QuoteData instance
        output_path: Path to save JSON file
        pretty: Use pretty formatting (indent=2)
    """
    indent = 2 if pretty else None
    quote_data.to_json(output_path, indent=indent)
    print(f"Saved quote data to: {output_path}")


def load_quote_data(input_path: str | Path) -> QuoteData:
    """
    Load QuoteData from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        QuoteData instance
    """
    return QuoteData.from_json(input_path)
