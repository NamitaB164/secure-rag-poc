from secure_rag.ingestion.parsers.pdf_elements import PDFElement


def test_pdf_element():
    element = PDFElement(
        type="text",
        page=1,
        content="Hello world",
    )

    assert element.type == "text"
    assert element.page == 1
    assert element.content == "Hello world"
    assert element.metadata == {}


def test_pdf_element_metadata():
    element = PDFElement(
        type="table",
        page=2,
        content="Name | Age\nJohn | 24",
        metadata={
            "rows": 2,
            "columns": 2,
        },
    )

    assert element.type == "table"
    assert element.page == 2
    assert element.metadata["rows"] == 2
    assert element.metadata["columns"] == 2
