"""Test DirectCostHelper McMaster functions."""

from cad_quoter.pricing.DirectCostHelper import get_mcmaster_part_and_price

print("=" * 60)
print("Testing DirectCostHelper McMaster Integration")
print("=" * 60)

# Test 1: Exact match
print("\nTest 1: Exact match (aluminum 5083, 6x6x0.25)")
result = get_mcmaster_part_and_price(
    length=6.0,
    width=6.0,
    thickness=0.25,
    material="aluminum 5083"
)
print(f"  Part Number: {result['part_number']}")
print(f"  Price: {result['price']}")
print(f"  Success: {result['success']}")

# Test 2: Round up
print("\nTest 2: Round up (aluminum 5083, 5.5x5.5x0.25 -> should find 6x6)")
result = get_mcmaster_part_and_price(
    length=5.5,
    width=5.5,
    thickness=0.25,
    material="aluminum 5083"
)
print(f"  Part Number: {result['part_number']}")
print(f"  Price: {result['price']}")
print(f"  Success: {result['success']}")

# Test 3: 303 Stainless Steel
print("\nTest 3: 303 Stainless Steel (6x6x0.5)")
result = get_mcmaster_part_and_price(
    length=6.0,
    width=6.0,
    thickness=0.5,
    material="303 Stainless Steel"
)
print(f"  Part Number: {result['part_number']}")
print(f"  Price: {result['price']}")
print(f"  Success: {result['success']}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
