# ``merge_effective`` owns all merge behavior.  These tables allow other helpers to
# share the same rules without duplicating the merge logic spread around the code
# base.
LOCKED_EFFECTIVE_FIELDS: frozenset[str] = frozenset(
    {
        "totals",
        "process_plan_pricing",
        "pricing_result",
        "process_hours",  # computed below from multipliers + adders
    }
)

SUGGESTIBLE_EFFECTIVE_FIELDS: frozenset[str] = frozenset(
    {
        "notes",
        "operation_sequence",
        "drilling_strategy",
        *SUGGESTION_SCALAR_KEYS,
    }
)

NUMERIC_EFFECTIVE_FIELDS: dict[str, tuple[float | None, float | None]] = {
    "fixture_build_hr": (0.0, 2.0),
    "soft_jaw_hr": (0.0, 1.0),
    "soft_jaw_material_cost": (0.0, 60.0),
    "handling_adder_hr": (0.0, 0.2),
    "cmm_minutes": (0.0, 60.0),
    "in_process_inspection_hr": (0.0, 0.5),
    "inspection_total_hr": (0.0, 12.0),
    "fai_prep_hr": (0.0, 1.0),
    "packaging_hours": (0.0, 0.5),
    "packaging_flat_cost": (0.0, 25.0),
    "shipping_cost": (0.0, None),
}

BOOL_EFFECTIVE_FIELDS: frozenset[str] = frozenset({"fai_required"})

TEXT_EFFECTIVE_FIELDS: dict[str, int] = {"shipping_hint": 80}

LIST_EFFECTIVE_FIELDS: frozenset[str] = frozenset({"operation_sequence"})

DICT_EFFECTIVE_FIELDS: frozenset[str] = frozenset({"drilling_strategy"})


    suggestions = {k: v for k, v in dict(suggestions or {}).items() if k not in LOCKED_EFFECTIVE_FIELDS}
    overrides = {k: v for k, v in dict(overrides or {}).items() if k not in LOCKED_EFFECTIVE_FIELDS}
    if "notes" in SUGGESTIBLE_EFFECTIVE_FIELDS:
        notes_val: list[str] = []
        if isinstance(suggestions.get("notes"), list):
            notes_val.extend(
                [str(n).strip() for n in suggestions["notes"] if isinstance(n, str) and n.strip()]
            )
        if isinstance(overrides.get("notes"), list):
            notes_val.extend(
                [str(n).strip() for n in overrides["notes"] if isinstance(n, str) and n.strip()]
            )
        if notes_val:
            eff["notes"] = notes_val

    for key, (lo, hi) in NUMERIC_EFFECTIVE_FIELDS.items():
        _merge_numeric_field(key, lo, hi, key)

    for key in BOOL_EFFECTIVE_FIELDS:
        _merge_bool_field(key)

    for key, max_len in TEXT_EFFECTIVE_FIELDS.items():
        _merge_text_field(key, max_len=max_len)

    for key in LIST_EFFECTIVE_FIELDS:
        _merge_list_field(key)

    for key in DICT_EFFECTIVE_FIELDS:
        _merge_dict_field(key)
    for key in NUMERIC_EFFECTIVE_FIELDS:
    for key in BOOL_EFFECTIVE_FIELDS:
    for key in TEXT_EFFECTIVE_FIELDS:
    if "operation_sequence" in LIST_EFFECTIVE_FIELDS and effective.get("operation_sequence"):
    if "drilling_strategy" in DICT_EFFECTIVE_FIELDS and isinstance(effective.get("drilling_strategy"), dict):
