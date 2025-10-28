from __future__ import annotations

import logging

import pytest

from cad_quoter.domain_models.values import safe_float
from cad_quoter.render.guards import render_drilling_guard
from cad_quoter.utils import jdump


@pytest.fixture()
def guard_logger():
    logger = logging.getLogger("test.render.guard")
    logger.propagate = True
    return logger


def _extract_debug(caplog):
    for record in caplog.records:
        if record.levelno == logging.DEBUG and "drill guard snapshot" in record.getMessage():
            return record.getMessage()
    raise AssertionError("expected drilling guard debug log")


def test_render_drilling_guard_logs_when_values_match(caplog, guard_logger):
    breakdown = {
        "process_plan": {"drilling": {"total_minutes_billed": 10.0}},
        "bucket_minutes_detail": {"drilling": 10.0},
        "material": {"total_cost": 12.5},
        "total_direct_costs": 42.0,
    }
    nre_detail = {"programming": {"prog_hr": 3.25}}

    with caplog.at_level(logging.DEBUG, logger=guard_logger.name):
        result = render_drilling_guard(
            logger=guard_logger,
            jdump=jdump,
            safe_float=safe_float,
            breakdown=breakdown,
            process_plan_summary=breakdown["process_plan"],
            bucket_minutes_detail=breakdown["bucket_minutes_detail"],
            nre_detail=nre_detail,
            ladder_subtotal=99.0,
        )

    assert result is None
    message = _extract_debug(caplog)
    assert '"drill_min_card": 10.0' in message
    assert '"drill_min_row": 10.0' in message
    assert '"programming_hr": 3.25' in message
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


def test_render_drilling_guard_logs_when_values_mismatch(caplog, guard_logger):
    breakdown = {
        "process_plan": {
            "drilling": {
                "total_minutes_billed": 4.0,
                "total_minutes_with_toolchange": 6.0,
            }
        },
        "bucket_minutes_detail": {"drilling": 8.0},
        "material": {"total_cost": 21.75},
        "total_direct_costs": 88.0,
    }

    with caplog.at_level(logging.DEBUG, logger=guard_logger.name):
        result = render_drilling_guard(
            logger=guard_logger,
            jdump=jdump,
            safe_float=safe_float,
            breakdown=breakdown,
            process_plan_summary=breakdown["process_plan"],
            bucket_minutes_detail=None,
            nre_detail=None,
            ladder_subtotal=15.5,
        )

    assert result is None
    message = _extract_debug(caplog)
    assert '"drill_min_card": 4.0' in message
    assert '"drill_min_row": 8.0' in message
    assert '"ladder_subtotal": 15.5' in message
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
