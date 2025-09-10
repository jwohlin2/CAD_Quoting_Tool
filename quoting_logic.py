# quoting_logic.py

# =============================================================================
# SHOP PARAMETERS - The main variables to configure for your specific shop
# =============================================================================

# --- Shop Labor Rates ---
SHOP_RATES = {
    'blended_rate_per_hour': 125.00,
    'setup_hours': 0.3,
    'inspection_pack_hours': 0.4,
    # ## NEW ## - Added costs for special processes
    'sinker_edm_per_hour': 150.00,
    'surface_grinding_per_hour': 90.00,
}

# --- Material Database ---
MATERIAL_DATA = {
    'aluminum_6061': {
        'density_g_cm3': 2.70,
        'cost_per_gram': 0.05,
        'min_charge': 20.00,
        'machining_multiplier': 1.0
    },
    'stainless_steel_304': {
        'density_g_cm3': 8.00,
        'cost_per_gram': 0.15,
        'min_charge': 40.00,
        'machining_multiplier': 1.8
    },
    'carbide_k20': {
        'density_g_cm3': 14.5,
        'cost_per_gram': 0.85,
        'min_charge': 50.00,
        'machining_multiplier': 3.5
    }
}

# --- Complexity & Process Factors ---
# These factors add time or cost based on part features.
PROCESS_FACTORS = {
    # ## NEW ## - Time added for tight tolerances
    'tight_tolerance_hours': 1.5, # Extra hours for high-precision work
    # ## NEW ## - Time added for EDM
    'sinker_edm_hours': 4.0, # Fixed time for setting up and running sinker EDM
    # ## NEW ## - Time added for edge honing
    'edge_hone_hours': 0.5, # Fixed time for deburring and edge finishing
}

# --- Quantity Discount Schedule ---
# ## NEW ## - A simple tiered discount on the final price.
QUANTITY_DISCOUNTS = {
    10: 0.05,  # 5% discount for 10 or more parts
    50: 0.10,  # 10% discount for 50 or more parts
    100: 0.15, # 15% discount for 100 or more parts
}

# --- General Calculation Parameters ---
BLANK_ALLOWANCE_MM = 0.6
MARGIN_PERCENTAGE = 40

def generate_quote(part_params):
    """
    Calculates a price quote based on a wide range of manufacturing parameters.
    """
    
    # 1. GET AND VALIDATE PARAMETERS
    # ---------------------------------
    material_name = part_params.get('material', 'aluminum_6061')
    quantity = part_params.get('quantity', 1)
    # ## NEW ## - Get new process parameters
    tight_tolerances = part_params.get('tight_tolerances', False)
    internal_sharp_corners = part_params.get('internal_sharp_corners', False)
    surface_grinding = part_params.get('surface_grinding', False)
    edge_honing = part_params.get('edge_honing', False)

    material_info = MATERIAL_DATA[material_name]

    # 2. MATERIAL COST CALCULATION
    # ---------------------------------
    blank_length_mm = part_params['length_mm'] + BLANK_ALLOWANCE_MM
    blank_width_mm = part_params['width_mm'] + BLANK_ALLOWANCE_MM
    blank_height_mm = part_params['height_mm'] + BLANK_ALLOWANCE_MM
    blank_vol_cm3 = (blank_length_mm * blank_width_mm * blank_height_mm) / 1000
    
    material_mass_g = blank_vol_cm3 * material_info['density_g_cm3']
    calculated_material_cost = material_mass_g * material_info['cost_per_gram']
    material_cost_per_part = max(calculated_material_cost, material_info['min_charge'])
    total_material_cost = material_cost_per_part * quantity

    # 3. LABOR COST CALCULATION
    # ---------------------------------
    net_vol_cm3 = (part_params['length_mm'] * part_params['width_mm'] * part_params['height_mm']) / 1000
    
    # --- Machining Time ---
    base_machining_hours = (net_vol_cm3 / 10) + 0.5
    adjusted_machining_hours = base_machining_hours * material_info['machining_multiplier']
    total_machining_hours = adjusted_machining_hours * quantity

    # --- Additional Process Time ---
    additional_hours = 0
    if tight_tolerances:
        additional_hours += PROCESS_FACTORS['tight_tolerance_hours']
    if internal_sharp_corners:
        additional_hours += PROCESS_FACTORS['sinker_edm_hours']
    if edge_honing:
        additional_hours += PROCESS_FACTORS['edge_hone_hours']
    if surface_grinding:
        # Grinding time is estimated based on surface area (if available)
        surface_area_cm2 = part_params.get('surface_area_mm2', 0) / 100
        additional_hours += (surface_area_cm2 / 50) # Example: 1 hour per 50 cm^2 of grinding

    # --- Total Labor ---
    setup_and_qc_hours = SHOP_RATES['setup_hours'] + SHOP_RATES['inspection_pack_hours']
    total_labor_hours = total_machining_hours + setup_and_qc_hours + additional_hours
    labor_cost = total_labor_hours * SHOP_RATES['blended_rate_per_hour']

    # 4. FINAL QUOTE ASSEMBLY
    # ---------------------------------
    consumables_cost = 35.00 # Reset to a simpler fixed cost for now
    
    cost_of_goods = total_material_cost + labor_cost + consumables_cost
    margin_amount = cost_of_goods * (MARGIN_PERCENTAGE / 100)
    final_price_before_discount = cost_of_goods + margin_amount

    # --- Quantity Discount ---
    discount_percentage = 0
    for qty_threshold, discount in sorted(QUANTITY_DISCOUNTS.items(), reverse=True):
        if quantity >= qty_threshold:
            discount_percentage = discount
            break
    
    discount_amount = final_price_before_discount * discount_percentage
    final_quote_price = final_price_before_discount - discount_amount
    price_per_part = final_quote_price / quantity

    # --- Notes ---
    notes = f"Quote for {quantity} part(s). Material: {material_name}."
    notes += "\nStandard Tolerances: general ±0.01 mm; thickness ±0.005 mm."
    notes += "\nExcluded Processes: Threads, Laser Marking, Coating."

    quote_breakdown = {
        "final_quote_price": round(final_quote_price, 2),
        "price_per_part": round(price_per_part, 2),
        "quantity": quantity,
        "details": {
            "total_material_cost": round(total_material_cost, 2),
            "labor_cost": round(labor_cost, 2),
            "consumables_cost": round(consumables_cost, 2),
            "margin_amount": round(margin_amount, 2),
            "discount_amount": round(discount_amount, 2),
            "total_labor_hours": round(total_labor_hours, 2),
        },
        "notes": notes
    }
    
    return quote_breakdown
