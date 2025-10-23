from __future__ import annotations

from cad_quoter.geometry.dxf_text import _clean_mtext, _iter_segments


def test_clean_mtext_replaces_tokens_and_alignment() -> None:
    raw = r"\A1;%%C 25 \P + %%P 0.1"
    assert _clean_mtext(raw) == "Ø 25 + ± 0.1"


def test_iter_segments_cleans_mtext_tokens() -> None:
    class DummyMText:
        def __init__(self, text: str) -> None:
            self.text = text

        def dxftype(self) -> str:
            return "MTEXT"

    dummy = DummyMText(r"\A0;MAIN \P%%D 45")

    segments = list(_iter_segments(dummy))

    assert segments == [("MAIN", None), ("° 45", None)]
