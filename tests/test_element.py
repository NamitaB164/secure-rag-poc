from secure_rag.ingestion.elements import ContentElement


def test_content_element_stores_text():
    element = ContentElement(
        type="text",
        content="Hello world",
    )

    assert element.type == "text"
    assert element.content == "Hello world"


def test_content_element_stores_metadata():
    element = ContentElement(
        type="text",
        content="Hello world",
        metadata={
            "source": "test.txt",
            "page": 1,
        },
    )

    assert element.metadata["source"] == "test.txt"
    assert element.metadata["page"] == 1
