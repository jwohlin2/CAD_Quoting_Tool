"""Test the labor hours estimation."""
from pathlib import Path
from cad_quoter.planning import plan_from_cad_file
from cad_quoter.planning.process_planner import LaborInputs, compute_labor_minutes

cad_file = Path("Cad Files/301_redacted.dxf")
if not cad_file.exists():
    print(f"ERROR: Test file not found: {cad_file}")
    exit(1)

print("=" * 80)
print("LABOR HOURS ESTIMATION TEST")
print("=" * 80)

# Get the process plan
print(f"\n1. Loading CAD file: {cad_file.name}")
plan = plan_from_cad_file(cad_file, verbose=False)

# Extract operation counts from plan
ops = plan.get('ops', [])
print(f"   Found {len(ops)} operations in plan")
print(f"   Operations: {[op.get('op') for op in ops[:5]]}...")

# Count different operation types
ops_total = len(ops)
holes_total = plan.get('extracted_hole_operations', 0)

print(f"\n2. Counting operation types...")
print(f"   Total operations: {ops_total}")
print(f"   Total holes: {holes_total}")

# Count specific operations
tool_changes = 0
fixturing_complexity = 1  # Default: light fixturing
edm_window_count = 0
edm_skim_passes = 0
thread_mill = 0
jig_grind_bore_qty = 0
grind_face_pairs = 0
deep_holes = 0
counterbore_qty = 0
counterdrill_qty = 0
ream_press_dowel = 0
ream_slip_dowel = 0
tap_rigid = 0
tap_npt = 0
outsource_touches = 0
part_flips = 0

# Analyze operations
for op in ops:
    op_type = op.get('op', '').lower()

    # Tool changes (estimate 2 per unique operation type)
    tool_changes += 2

    # EDM operations
    if 'wedm' in op_type or 'wire_edm' in op_type:
        edm_window_count += 1
        edm_skim_passes += op.get('skims', 0)

    # Thread operations
    if 'thread_mill' in op_type:
        thread_mill += 1

    # Tapping
    if 'tap' in op_type:
        if 'rigid' in op_type:
            tap_rigid += 1
        elif 'npt' in op_type or 'pipe' in op_type:
            tap_npt += 1
        else:
            tap_rigid += 1  # Default to rigid tap

    # Grinding
    if 'jig_grind' in op_type and 'bore' in op_type:
        jig_grind_bore_qty += 1
    if 'grind' in op_type and 'face' in op_type:
        grind_face_pairs += 1

    # Counterbore/Counterdrill
    if 'counterbore' in op_type or 'cbore' in op_type:
        counterbore_qty += 1
    if 'counterdrill' in op_type or 'cdrill' in op_type:
        counterdrill_qty += 1

    # Dowel operations
    if 'ream_press' in op_type or ('ream' in op_type and 'press' in op_type):
        ream_press_dowel += 1
    if 'ream_slip' in op_type or ('ream' in op_type and 'slip' in op_type):
        ream_slip_dowel += 1

    # Outsourced operations
    if 'heat_treat' in op_type or 'coat' in op_type or 'hone' in op_type:
        outsource_touches += 1

    # Part flips
    if 'back' in str(op.get('side', '')).lower():
        part_flips = max(part_flips, 1)

print(f"   Tool changes: {tool_changes}")
print(f"   Jig grind bores: {jig_grind_bore_qty}")
print(f"   Rigid taps: {tap_rigid}")

# Check fixturing notes
fixturing_notes = plan.get('fixturing', [])
if len(fixturing_notes) > 2:
    fixturing_complexity = 2  # Moderate
if any('mag' in str(note).lower() or 'parallel' in str(note).lower() for note in fixturing_notes):
    fixturing_complexity = max(fixturing_complexity, 2)

print(f"   Fixturing complexity: {fixturing_complexity}")

# Create LaborInputs
print(f"\n3. Creating LaborInputs object...")
labor_inputs = LaborInputs(
    ops_total=ops_total,
    holes_total=holes_total,
    tool_changes=tool_changes,
    fixturing_complexity=fixturing_complexity,
    edm_window_count=edm_window_count,
    edm_skim_passes=edm_skim_passes,
    thread_mill=thread_mill,
    jig_grind_bore_qty=jig_grind_bore_qty,
    grind_face_pairs=grind_face_pairs,
    deep_holes=deep_holes,
    counterbore_qty=counterbore_qty,
    counterdrill_qty=counterdrill_qty,
    ream_press_dowel=ream_press_dowel,
    ream_slip_dowel=ream_slip_dowel,
    tap_rigid=tap_rigid,
    tap_npt=tap_npt,
    outsource_touches=outsource_touches,
    inspection_frequency=0.1,  # 10% sampling
    part_flips=part_flips,
)

# Compute labor minutes
print(f"\n4. Computing labor minutes...")
labor_result = compute_labor_minutes(labor_inputs)
minutes = labor_result['minutes']

# Display the report
print("\n" + "=" * 80)
print("LABOR HOURS ESTIMATION")
print("=" * 80)
print()
print("LABOR BREAKDOWN BY CATEGORY")
print("-" * 80)
print(f"  Setup / Prep:                    {minutes['Setup']:>10.2f} minutes")
print(f"  Programming / Prove-out:         {minutes['Programming']:>10.2f} minutes")
print(f"  Machining Steps:                 {minutes['Machining_Steps']:>10.2f} minutes")
print(f"  Inspection:                      {minutes['Inspection']:>10.2f} minutes")
print(f"  Finishing / Deburr:              {minutes['Finishing']:>10.2f} minutes")
print("-" * 80)
print(f"  TOTAL LABOR TIME:                {minutes['Labor_Total']:>10.2f} minutes")
print(f"                                   {minutes['Labor_Total']/60:>10.2f} hours")
print()

print("LABOR INPUT DETAILS")
print("-" * 80)
print(f"  Total Operations:                {labor_inputs.ops_total:>10}")
print(f"  Total Holes:                     {labor_inputs.holes_total:>10}")
print(f"  Tool Changes:                    {labor_inputs.tool_changes:>10}")
print(f"  Fixturing Complexity:            {labor_inputs.fixturing_complexity:>10} (0=none, 1=light, 2=moderate, 3=complex)")

if edm_window_count > 0:
    print(f"  EDM Windows:                     {labor_inputs.edm_window_count:>10}")
    print(f"  EDM Skim Passes:                 {labor_inputs.edm_skim_passes:>10}")

if thread_mill > 0:
    print(f"  Thread Mill Operations:          {labor_inputs.thread_mill:>10}")

if jig_grind_bore_qty > 0:
    print(f"  Jig Grind Bores:                 {labor_inputs.jig_grind_bore_qty:>10}")

if grind_face_pairs > 0:
    print(f"  Grind Face Pairs:                {labor_inputs.grind_face_pairs:>10}")

if tap_rigid > 0:
    print(f"  Rigid Tap Operations:            {labor_inputs.tap_rigid:>10}")

if tap_npt > 0:
    print(f"  NPT Tap Operations:              {labor_inputs.tap_npt:>10}")

if counterbore_qty > 0:
    print(f"  Counterbore Operations:          {labor_inputs.counterbore_qty:>10}")

if counterdrill_qty > 0:
    print(f"  Counterdrill Operations:         {labor_inputs.counterdrill_qty:>10}")

if ream_press_dowel > 0:
    print(f"  Press Fit Dowel Reaming:         {labor_inputs.ream_press_dowel:>10}")

if ream_slip_dowel > 0:
    print(f"  Slip Fit Dowel Reaming:          {labor_inputs.ream_slip_dowel:>10}")

if outsource_touches > 0:
    print(f"  Outsource Touches:               {labor_inputs.outsource_touches:>10}")

if part_flips > 0:
    print(f"  Part Flips:                      {labor_inputs.part_flips:>10}")

print(f"  Inspection Frequency:            {labor_inputs.inspection_frequency:>10.1%}")

print()
print("=" * 80)

print("\n[SUCCESS] Labor hours estimation test completed!")
print("\nThis output will now appear in AppV7's Output tab when you click 'Generate Quote'")
