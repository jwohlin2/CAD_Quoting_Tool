import pandas as pd
import pytest

import appV5


def _render_text_and_payload(result: dict) -> tuple[str, dict]:
    rendered = appV5.render_quote(result, currency="$")
    breakdown = result.get("breakdown")
    assert isinstance(breakdown, dict), "expected dict breakdown"
    payload = breakdown.get("render_payload")
    assert isinstance(payload, dict), "expected structured render payload"
    return rendered, payload


def test_project_management_and_toolmaker_rows_rendered() -> None:
    df = pd.DataFrame(
        [
            {"Item": "Qty", "Example Values / Options": 1, "Data Type / Input Method": "number"},
            {
                "Item": "Material Name",
                "Example Values / Options": "6061-T6 Aluminum",
                "Data Type / Input Method": "text",
            },
            {
                "Item": "Net Volume (cm^3)",
                "Example Values / Options": 50,
                "Data Type / Input Method": "number",
            },
            {
                "Item": "Thickness (in)",
                "Example Values / Options": 1.0,
                "Data Type / Input Method": "number",
            },
            {
                "Item": "Project Management Hours",
                "Example Values / Options": 2.0,
                "Data Type / Input Method": "number",
            },
            {
                "Item": "Tool & Die Maker Hours",
                "Example Values / Options": 1.5,
                "Data Type / Input Method": "number",
            },
        ]
    )

    result = appV5.compute_quote_from_df(df, llm_enabled=False)
    breakdown = result["breakdown"]
    process_costs = breakdown["process_costs"]
    project_meta = breakdown["process_meta"].get("project_management")
    toolmaker_meta = breakdown["process_meta"]["toolmaker_support"]

    assert project_meta is None or project_meta.get("hr", 0.0) == 0.0, breakdown
    assert toolmaker_meta["hr"] > 0, breakdown
    assert "project_management" not in process_costs or process_costs["project_management"] == 0.0, breakdown
    assert process_costs["toolmaker_support"] > 0, breakdown

    rendered, payload = _render_text_and_payload(result)
    assert "Project Management" not in rendered
    processes = payload.get("processes", [])
    toolmaker_entry = next(
        (
            entry
            for entry in processes
            if entry.get("label") == "Toolmaker Support"
        ),
        None,
    )
    assert toolmaker_entry is not None, processes
    assert toolmaker_entry.get("amount") == pytest.approx(
        process_costs["toolmaker_support"], abs=1e-2
    )
    assert toolmaker_entry.get("hours") == pytest.approx(
        toolmaker_meta["hr"], abs=1e-2
    )
