from cad_quoter.planning import extract_hole_table_from_cad
import json

ht = extract_hole_table_from_cad('Cad Files/301_redacted.dxf')

with open('holes_list.txt', 'w', encoding='utf-8') as f:
    for e in ht:
        f.write(f"\nHole {e['HOLE']}:\n")
        f.write(f"  REF_DIAM: {e['REF_DIAM']}\n")
        f.write(f"  QTY: {e['QTY']}\n")
        f.write(f"  DESCRIPTION: {e['DESCRIPTION']}\n")

print("Wrote to holes_list.txt")
