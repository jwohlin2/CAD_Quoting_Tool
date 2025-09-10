# quoting_logic.py

# --- Hard-coded Shop Parameters (Replace with your actual numbers) ---
SHOP_RATES = {
    'blended_rate_per_hour': 85.00, # More reasonable blended rate
}

MATERIAL_DATA = {
    'aluminum_6061': {
        'density_g_cm3': 2.7,
        'cost_per_gram': 0.05, # $0.05/gram
        'min_charge': 20.00,   # Minimum material cost
    },
    'carbide_k20': {
        'density_g_cm3': 14.5,
        'cost_per_gram': 0.85, # $0.85/gram
        'min_charge': 50.00,   # Minimum material cost
    }
}

# --- Calculation Parameters ---
BLANK_ALLOWANCE_MM = 0.6 # Add ~0.3mm per face
MARGIN_PERCENTAGE = 25  # 25% margin

def generate_quote(part_params):
    """
    Calculates a price quote based on part dimensions and shop rates.
    """
    
    # --- Sanity Check ---
    max_dim_mm = 1000 # 1 meter
    if any(d > max_dim_mm for d in [part_params['length_mm'], part_params['width_mm'], part_params['height_mm']]):
        raise ValueError(f"Part dimensions exceed the maximum allowed size of {max_dim_mm}mm.")

    # 1. MATERIAL COST CALCULATION
    # ---------------------------------
    material_name = 'aluminum_6061' # Default to a more common material
    material_info = MATERIAL_DATA[material_name]

    net_vol_cm3 = (part_params['length_mm'] * part_params['width_mm'] * part_params['height_mm']) / 1000

    blank_length_mm = part_params['length_mm'] + BLANK_ALLOWANCE_MM
    blank_width_mm = part_params['width_mm'] + BLANK_ALLOWANCE_MM
    blank_height_mm = part_params['height_mm'] + BLANK_ALLOWANCE_MM
    blank_vol_cm3 = (blank_length_mm * blank_width_mm * blank_height_mm) / 1000
    
    material_mass_g = blank_vol_cm3 * material_info['density_g_cm3']
    calculated_material_cost = material_mass_g * material_info['cost_per_gram']
    
    material_cost = max(calculated_material_cost, material_info['min_charge'])

    # 2. LABOR COST CALCULATION
    # ---------------------------------
    prog_setup_hours = 0.5 # Slightly increased for more complex parts
    inspection_pack_hours = 0.25

    # Adjusted machining hours formula
    machining_hours = (net_vol_cm3 / 50) + 0.5 # Base 0.5hr + time per cm³
    
    total_labor_hours = prog_setup_hours + inspection_pack_hours + machining_hours
    labor_cost = total_labor_hours * SHOP_RATES['blended_rate_per_hour']

    # 3. FINAL QUOTE ASSEMBLY
    # ---------------------------------
    # Consumables as a percentage of labor
    consumables_cost = labor_cost * 0.05 # 5% of labor for tooling, etc.
    
    cost_of_goods = material_cost + labor_cost + consumables_cost
    margin_amount = cost_of_goods * (MARGIN_PERCENTAGE / 100)
    
    final_quote_price = cost_of_goods + margin_amount

    # Create a dictionary with the results
    quote_breakdown = {
        "final_quote_price": round(final_quote_price, 2),
        "cost_of_goods": round(cost_of_goods, 2),
        "margin_amount": round(margin_amount, 2),
        "details": {
            "material_cost": round(material_cost, 2),
            "labor_cost": round(labor_cost, 2),
            "consumables_cost": round(consumables_cost, 2),
            "total_labor_hours": round(total_labor_hours, 2),
        },
        "notes": f"Quote for 1-off part. Material: {material_name}. Lead time: 5-7 days."
    }
    
    return quote_breakdown

# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    example_part = {
        'length_mm': 15.145,
        'width_mm': 7.622,
        'height_mm': 5.217
    }

    test_quote = generate_quote(example_part)

    import json
    print(json.dumps(test_quote, indent=4))