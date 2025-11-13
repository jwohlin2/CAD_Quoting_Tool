# McMaster Pricing Helper – Volume-Based Fallback

This module calls the McMaster-Carr API to fetch price tiers for a given part number and now includes an **optional volume-based price estimate** for when there is **no price-per-piece** tier available (no qty-1 price).

The goal:  
If McMaster only gives you pricing like “10 pcs minimum” or “25 pcs minimum” and you still want to estimate a price for **your** part, we:

1. Find the **highest unit price tier** (smallest minimum quantity).
2. Convert that into **price per cubic inch** of material.
3. Multiply by **your part’s cubic inches** to get a rough per-part price.

---

## What This Script Does

1. Logs into McMaster-Carr using the existing API logic.
2. Fetches all **price tiers** for a given part number.
3. Prints the pricing table as usual.
4. If there is **no qty-1 tier**, it offers a **volume-based fallback**:
   - Asks you for the size of the McMaster stock piece you’re buying.
   - Asks you for the size of your machined part.
   - Computes:
     - Dollars per cubic inch ($/in³) from the stock.
     - An estimated price for your part based on its volume.

---

## New Helper Functions

These helpers sit on top of the existing `get_price_tiers` logic.

### 1. `get_qty_one_tier(tiers)`

**Purpose:**  
Find the tier that applies at quantity 1 (if it exists).

**Logic:**

- Look through the list of tier dictionaries.
- Return the first tier where `MinimumQuantity <= 1`.
- If none match, return `None`.

This answers:  
> “Do we have a real price-per-piece from McMaster?”  

If **yes**, we can just use that as the price per part.  
If **no**, we need the volume-based fallback.

---

### 2. `compute_price_per_cubic_inch(tiers, stock_volume_cuin)`

**Purpose:**  
Turn McMaster’s pricing into a **price per cubic inch** of material.

**Inputs:**

- `tiers`: The list of price tiers from `get_price_tiers`.
- `stock_volume_cuin`: The cubic inches of the McMaster piece you’re buying.

**Logic:**

1. If there’s a qty-1 tier (`MinimumQuantity <= 1`), use **that** tier’s `Amount` as `base_price`.
2. If there is **no** qty-1 tier:
   - Use the **“largest price” tier** (the one with the smallest minimum quantity – that’s usually the most expensive per piece).
3. Compute:

   ```text
   price_per_cubic_inch = base_price / stock_volume_cuin
   ```

4. Return `price_per_cubic_inch` (or `None` if we can’t compute it).

This answers:  
> “How many dollars is one cubic inch of this McMaster material?”

---

### 3. `estimate_price_for_part_from_volume(tiers, stock_volume_cuin, part_volume_cuin)`

**Purpose:**  
Use the $/in³ to estimate a **price for your part**.

**Inputs:**

- `tiers`: Price tiers from McMaster.
- `stock_volume_cuin`: Volume of the McMaster stock piece.
- `part_volume_cuin`: Volume of your finished part.

**Logic:**

1. Call `compute_price_per_cubic_inch` to get $/in³.
2. Multiply by the part’s volume:

   ```text
   estimated_part_price = price_per_cubic_inch × part_volume_cuin
   ```

3. Return `estimated_part_price` (or `None` if something is missing).

This answers:  
> “If McMaster charges X dollars per cubic inch, what’s a rough price for my Y-cubic-inch part?”

---

## CLI Add-On: Volume Prompts

To use this logic in the **interactive script**, we add:

### `prompt_volume(label)`

Prompts for length, width, and thickness in inches and returns:

```text
volume_cuin = length × width × thickness
```

If any value is blank or invalid, it returns `0.0` and we just skip the approximation.

---

## How It’s Wired Into `main()`

After you:

- Log in
- Fetch tiers with `get_price_tiers`
- Call `print_tiers(tiers)`

We do:

1. Check for a qty-1 tier:

   ```python
   one_tier = get_qty_one_tier(tiers)
   ```

2. If `one_tier` **exists**:
   - You already have a price-per-piece, nothing else needed.

3. If `one_tier` is **None**:
   - Print a message:  
     `No qty=1 unit price available. You can approximate using $/cubic inch.`
   - Prompt for:
     - **Stock volume** (the McMaster block you’re buying)
     - **Part volume** (your machined part)
   - Compute and print:
     - Approximate $/in³.
     - Approximate price for your part.

---

## Simple Example (Conceptual)

Let’s say:

- McMaster only shows:  
  “10 pcs: $200” → $20 per piece  
- The McMaster block you buy is:  
  12" × 14" × 2" → `12 × 14 × 2 = 336 in³`
- Your part’s volume is:  
  8" × 6" × 1" → `48 in³`

Steps:

1. Base price = $20 (from the smallest-quantity tier).
2. $/in³ = `20 / 336 ≈ 0.0595 $/in³`.
3. Estimated price for your part:

   ```text
   0.0595 × 48 ≈ $2.86
   ```

So the script would report roughly:

- Approx price per cubic inch: `$0.0595/in³`
- Approx price for your part: `~$2.86`

(You’ll probably add margin and real quoting logic somewhere else in the pipeline.)

---

## Integration Into a Larger Quoting App

If you’re using this inside a bigger quoting system:

- Use the volumes you already compute from CAD:
  - `stock_volume_cuin` from your chosen McMaster blank.
  - `part_volume_cuin` from your part geometry.
- Call:

  ```python
  est_price = estimate_price_for_part_from_volume(
      tiers,
      stock_volume_cuin,
      part_volume_cuin,
  )
  ```

- If `est_price` is not `None`, you can:
  - Treat it as the **material price per part**, or
  - Use it as a **sanity check** against other material-cost logic.

---

## Summary

**Problem:** McMaster doesn’t always give a direct price per single piece.

**Solution:**

1. Look for a qty-1 tier. If it exists, use it.
2. If not, use the smallest-quantity tier as the “largest price.”
3. Turn that into **$/cubic inch** using the stock volume.
4. Multiply by your part’s volume to get an estimated per-part material price.

This keeps the original API flow intact while giving you a simple, understandable fallback that’s easy to plug into your quoting logic.
