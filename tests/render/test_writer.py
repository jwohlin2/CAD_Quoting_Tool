from cad_quoter.render.writer import QuoteWriter
from cad_quoter.utils.render_utils import QuoteDocRecorder


def test_quote_writer_records_sections() -> None:
    divider = "-" * 5
    recorder = QuoteDocRecorder(divider)
    writer = QuoteWriter(divider=divider, page_width=20, currency="$", recorder=recorder)

    writer.line("Quote Title")
    writer.line("Section One")
    writer.line(divider)
    writer.line("First row")

    doc = recorder.build_doc()

    assert doc.title == "Quote Title"
    assert len(doc.sections) == 1
    assert doc.sections[0].title == "Section One"
    assert [row.text for row in doc.sections[0].rows] == ["First row"]


def test_quote_writer_placeholder_updates_line() -> None:
    divider = "-" * 6
    recorder = QuoteDocRecorder(divider)
    writer = QuoteWriter(divider=divider, page_width=30, currency="$", recorder=recorder)

    writer.line("Title")
    writer.line("Section")
    writer.line(divider)
    placeholder = writer.placeholder("Pending")
    placeholder.replace("Resolved")

    assert writer.lines[-1] == "Resolved"
    doc = recorder.build_doc()
    assert doc.sections[0].rows[-1].text == "Resolved"


def test_quote_writer_total_row_inserts_separator() -> None:
    divider = "-" * 10
    writer = QuoteWriter(divider=divider, page_width=30, currency="$")

    writer.line("Title")
    writer.row("Subtotal:", 12.34)
    writer.row("Total Cost:", 45.67)

    assert writer.lines[-2].strip(" ") == "-" * len(writer.lines[-1].split()[-1])


def test_quote_writer_list_interface_delegates() -> None:
    divider = "-" * 4
    recorder = QuoteDocRecorder(divider)
    writer = QuoteWriter(divider=divider, page_width=25, currency="$", recorder=recorder)

    writer.line("Header")
    writer.lines.append("Section")
    writer.lines.append(divider)
    writer.lines.append("Body")

    doc = recorder.build_doc()
    assert doc.sections[0].rows[-1].text == "Body"
