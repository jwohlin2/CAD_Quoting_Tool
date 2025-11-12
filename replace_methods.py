"""
Script to replace the three report methods in AppV7.py with QuoteData-based versions.
"""

from pathlib import Path

def replace_methods():
    # Read the old file
    appv7_path = Path("AppV7.py")
    with open(appv7_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Read the new methods
    with open("appv7_new_methods.py", 'r', encoding='utf-8') as f:
        new_methods = f.read()

    # Method boundaries (1-indexed):
    # _generate_direct_costs_report: line 504 starts, line 814 ends (blank line before next method)
    # _generate_machine_hours_report: line 815 starts
    # _generate_labor_hours_report: line 955 starts
    # _format_cost_summary_line: line 1163 starts (next method after labor)

    print("Reading AppV7.py...")
    print(f"Total lines: {len(lines)}")

    # Find the exact line numbers
    direct_start = None
    next_after_labor = None

    for i, line in enumerate(lines):
        if "def _generate_direct_costs_report" in line:
            direct_start = i
            print(f"Found _generate_direct_costs_report at line {i+1}")
        elif direct_start is not None and i > direct_start + 100 and "def _format_cost_summary" in line:
            next_after_labor = i
            print(f"Found _format_cost_summary_line at line {i+1}")
            break

    if direct_start is None:
        print("ERROR: Could not find _generate_direct_costs_report")
        return
    if next_after_labor is None:
        print("ERROR: Could not find end of labor report method")
        return

    # Create backup
    backup_path = Path("AppV7.py.backup")
    print(f"\nCreating backup at {backup_path}...")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print("\nReplacing methods...")

    # Keep everything before _generate_direct_costs_report
    new_file = lines[:direct_start]

    # Add the new methods
    new_file.append(new_methods)
    new_file.append("\n")

    # Add everything after _generate_labor_hours_report
    new_file.extend(lines[next_after_labor:])

    # Write the new file
    with open(appv7_path, 'w', encoding='utf-8') as f:
        f.writelines(new_file)

    print(f"[OK] Successfully replaced methods in AppV7.py")
    print(f"[OK] Backup saved to {backup_path}")
    print(f"[OK] Old file: {len(lines)} lines")
    print(f"[OK] New file: {len(new_file)} lines")

if __name__ == "__main__":
    replace_methods()
