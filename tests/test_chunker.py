from secure_rag.ingestion.chunking.dynamic import (
    _split_sentences,
    chunk_elements,
)
from secure_rag.ingestion.elements import ContentElement


def test_short_text_becomes_one_chunk():
    elements = [
        ContentElement(
            type="text",
            content="Employees must use strong passwords.",
            metadata={"source": "policy.txt"},
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert len(chunks) == 1
    assert chunks[0].content == "Employees must use strong passwords."


def test_chunk_preserves_document_security_metadata():
    elements = [
        ContentElement(
            type="text",
            content="Internal company policy.",
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert chunks[0].document_id == "doc-001"
    assert chunks[0].clearance == 2
    assert chunks[0].trust == 1


def test_chunk_indices_are_sequential():
    elements = [
        ContentElement(
            type="text",
            content=(
                "Paragraph one. " * 100
                + "\n\n"
                + "Paragraph two. " * 100
            ),
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
        max_chars=300,
    )

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunk_preserves_element_metadata():
    elements = [
        ContentElement(
            type="text",
            content="Authentication policy.",
            metadata={
                "format": "markdown",
                "heading": "Authentication",
                "heading_level": 2,
            },
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert chunks[0].metadata["format"] == "markdown"
    assert chunks[0].metadata["heading"] == "Authentication"
    assert chunks[0].metadata["heading_level"] == 2


def test_heading_stays_with_following_content():
    elements = [
        ContentElement(
            type="text",
            content="Authentication",
            metadata={
                "heading": "Authentication",
                "heading_level": 1,
            },
        ),
        ContentElement(
            type="text",
            content="Employees must use MFA before accessing production.",
            metadata={
                "heading": "Authentication",
                "heading_level": 1,
            },
        ),
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert len(chunks) == 1
    assert "Authentication" in chunks[0].content
    assert "Employees must use MFA" in chunks[0].content


def test_small_table_stays_intact():
    elements = [
        ContentElement(
            type="table",
            content=(
                "Name | Role | Clearance\n"
                "Alice | Engineer | 2\n"
                "Bob | Manager | 3"
            ),
            metadata={
                "page": 1,
                "rows": 3,
                "columns": 3,
            },
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert len(chunks) == 1
    assert "Alice" in chunks[0].content
    assert "Bob" in chunks[0].content


def test_large_table_preserves_header():
    rows = "\n".join(
        f"User{i} | Engineer | {i % 4}"
        for i in range(50)
    )

    elements = [
        ContentElement(
            type="table",
            content=f"Name | Role | Clearance\n{rows}",
            metadata={
                "page": 1,
                "rows": 51,
                "columns": 3,
            },
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
        max_chars=300,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert "Name | Role | Clearance" in chunk.content


def test_image_is_not_split_as_text():
    elements = [
        ContentElement(
            type="image",
            content="",
            metadata={
                "page": 1,
                "asset_path": "assets/report/page-1-image-0.png",
                "width": 100,
                "height": 80,
            },
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert len(chunks) == 1
    assert chunks[0].content == ""
    assert chunks[0].metadata["asset_path"] == (
        "assets/report/page-1-image-0.png"
    )


def test_empty_elements_produce_no_chunks():
    chunks = chunk_elements(
        [],
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert chunks == []


def test_long_text_has_sentence_overlap():
    elements = [
        ContentElement(
            type="text",
            content=(
                "Sentence one. "
                "Sentence two. "
                "Sentence three. "
                "Sentence four."
            ),
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
        max_chars=30,
        overlap_sentences=1,
    )

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        previous_sentences = _split_sentences(previous.content)
        assert previous_sentences[-1] in current.content


def test_short_text_has_no_overlap():
    elements = [
        ContentElement(
            type="text",
            content="Employees must use MFA.",
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
        overlap_sentences=1,
    )

    assert len(chunks) == 1
def test_different_headings_create_separate_sections():
    elements = [
        ContentElement(
            type="text",
            content="Authentication",
            metadata={
                "heading": "Authentication",
                "heading_level": 1,
            },
        ),
        ContentElement(
            type="text",
            content="Employees must use MFA.",
            metadata={
                "heading": "Authentication",
                "heading_level": 1,
            },
        ),
        ContentElement(
            type="text",
            content="Authorization",
            metadata={
                "heading": "Authorization",
                "heading_level": 1,
            },
        ),
        ContentElement(
            type="text",
            content="Access is controlled by user clearance.",
            metadata={
                "heading": "Authorization",
                "heading_level": 1,
            },
        ),
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert len(chunks) == 2

    assert "Authentication" in chunks[0].content
    assert "Employees must use MFA." in chunks[0].content

    assert "Authorization" in chunks[1].content
    assert "Access is controlled by user clearance." in chunks[1].content


def test_heading_does_not_merge_with_different_heading():
    elements = [
        ContentElement(
            type="text",
            content="Authentication",
            metadata={
                "heading": "Authentication",
                "heading_level": 1,
            },
        ),
        ContentElement(
            type="text",
            content="MFA is required.",
            metadata={
                "heading": "Authentication",
                "heading_level": 1,
            },
        ),
        ContentElement(
            type="text",
            content="Networking",
            metadata={
                "heading": "Networking",
                "heading_level": 1,
            },
        ),
        ContentElement(
            type="text",
            content="All traffic must use TLS.",
            metadata={
                "heading": "Networking",
                "heading_level": 1,
            },
        ),
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert len(chunks) == 2

    assert "Networking" not in chunks[0].content
    assert "Authentication" not in chunks[1].content


def test_multiple_paragraphs_under_heading_remain_in_same_section():
    elements = [
        ContentElement(
            type="text",
            content="Authentication",
            metadata={
                "heading": "Authentication",
                "heading_level": 1,
            },
        ),
        ContentElement(
            type="text",
            content="Employees must use MFA.",
            metadata={
                "heading": "Authentication",
                "heading_level": 1,
            },
        ),
        ContentElement(
            type="text",
            content="Privileged users require hardware keys.",
            metadata={
                "heading": "Authentication",
                "heading_level": 1,
            },
        ),
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert len(chunks) == 1
    assert "Employees must use MFA." in chunks[0].content
    assert "Privileged users require hardware keys." in chunks[0].content


def test_section_metadata_is_preserved_after_chunking():
    elements = [
        ContentElement(
            type="text",
            content="Authentication",
            metadata={
                "heading": "Authentication",
                "heading_level": 2,
                "page": 4,
                "format": "markdown",
            },
        ),
        ContentElement(
            type="text",
            content="Employees must use MFA.",
            metadata={
                "heading": "Authentication",
                "heading_level": 2,
                "page": 4,
                "format": "markdown",
            },
        ),
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["heading"] == "Authentication"
    assert chunks[0].metadata["heading_level"] == 2
    assert chunks[0].metadata["page"] == 4
    assert chunks[0].metadata["format"] == "markdown"


def test_paragraphs_are_split_before_sentence_fallback():
    elements = [
        ContentElement(
            type="text",
            content=(
                "First paragraph contains important authentication policy. "
                "It explains the required controls.\n\n"
                "Second paragraph describes authorization requirements. "
                "It explains clearance levels."
            ),
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
        max_chars=200,
    )

    assert len(chunks) >= 2

    assert any(
        "First paragraph" in chunk.content
        for chunk in chunks
    )

    assert any(
        "Second paragraph" in chunk.content
        for chunk in chunks
    )

# Chunker edge cases



def test_sentence_longer_than_max_chars_falls_back_to_words():
    elements = [
        ContentElement(
            type="text",
            content=(
                "This is an extremely long sentence containing many words "
                "that must be split because the sentence itself is longer "
                "than the configured maximum chunk size."
            ),
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
        max_chars=50,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.content) <= 50


def test_text_exactly_at_max_chars_stays_in_one_chunk():
    content = "A" * 100

    elements = [
        ContentElement(
            type="text",
            content=content,
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
        max_chars=100,
    )

    assert len(chunks) == 1
    assert chunks[0].content == content


def test_table_row_larger_than_max_chars_does_not_break_header():
    large_row = (
        "User | Engineer | "
        + "A" * 200
    )

    elements = [
        ContentElement(
            type="table",
            content=(
                "Name | Role | Clearance\n"
                f"{large_row}\n"
                "Bob | Manager | 3"
            ),
            metadata={
                "page": 1,
                "rows": 3,
                "columns": 3,
            },
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
        max_chars=100,
    )

    assert len(chunks) >= 1

    for chunk in chunks:
        assert "Name | Role | Clearance" in chunk.content


def test_multiple_tables_remain_separate():
    elements = [
        ContentElement(
            type="table",
            content=(
                "Name | Role\n"
                "Alice | Engineer"
            ),
            metadata={
                "page": 1,
                "rows": 2,
                "columns": 2,
            },
        ),
        ContentElement(
            type="table",
            content=(
                "System | Status\n"
                "Database | Active"
            ),
            metadata={
                "page": 2,
                "rows": 2,
                "columns": 2,
            },
        ),
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert len(chunks) == 2

    assert "Alice" in chunks[0].content
    assert "Database" in chunks[1].content


def test_image_between_text_sections_remains_independent():
    elements = [
        ContentElement(
            type="text",
            content="Authentication policy.",
            metadata={"heading": "Authentication"},
        ),
        ContentElement(
            type="image",
            content="",
            metadata={
                "page": 1,
                "asset_path": "assets/auth.png",
            },
        ),
        ContentElement(
            type="text",
            content="Authorization policy.",
            metadata={"heading": "Authorization"},
        ),
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert len(chunks) == 3

    assert chunks[0].content == "Authentication policy."
    assert chunks[1].content == ""
    assert chunks[1].metadata["asset_path"] == "assets/auth.png"
    assert chunks[2].content == "Authorization policy."


def test_heading_table_and_text_remain_structurally_separate():
    elements = [
        ContentElement(
            type="text",
            content="Authentication",
            metadata={
                "heading": "Authentication",
                "heading_level": 1,
            },
        ),
        ContentElement(
            type="table",
            content=(
                "Method | Required\n"
                "MFA | Yes"
            ),
            metadata={
                "heading": "Authentication",
                "heading_level": 1,
                "page": 1,
                "rows": 2,
                "columns": 2,
            },
        ),
        ContentElement(
            type="text",
            content="All employees must use MFA.",
            metadata={
                "heading": "Authentication",
                "heading_level": 1,
            },
        ),
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert len(chunks) == 3

    assert chunks[0].content == "Authentication"
    assert "MFA | Yes" in chunks[1].content
    assert chunks[2].content == "All employees must use MFA."


def test_heading_image_and_text_remain_structurally_separate():
    elements = [
        ContentElement(
            type="text",
            content="Architecture",
            metadata={
                "heading": "Architecture",
                "heading_level": 1,
            },
        ),
        ContentElement(
            type="image",
            content="",
            metadata={
                "heading": "Architecture",
                "heading_level": 1,
                "page": 2,
                "asset_path": "assets/architecture.png",
            },
        ),
        ContentElement(
            type="text",
            content="The system uses a secure RAG pipeline.",
            metadata={
                "heading": "Architecture",
                "heading_level": 1,
            },
        ),
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert len(chunks) == 3

    assert chunks[0].content == "Architecture"
    assert chunks[1].content == ""
    assert chunks[1].metadata["asset_path"] == (
        "assets/architecture.png"
    )
    assert "secure RAG pipeline" in chunks[2].content


def test_metadata_is_isolated_between_chunks():
    elements = [
        ContentElement(
            type="text",
            content=(
                "First sentence is intentionally long enough "
                "to require multiple chunks. "
                "Second sentence is also intentionally long "
                "enough to require another chunk."
            ),
            metadata={
                "format": "markdown",
                "heading": "Security",
            },
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
        max_chars=60,
    )

    assert len(chunks) > 1

    chunks[0].metadata["modified"] = True

    assert "modified" not in chunks[1].metadata


def test_page_metadata_is_preserved_per_element():
    elements = [
        ContentElement(
            type="text",
            content="Content from page one.",
            metadata={"page": 1},
        ),
        ContentElement(
            type="text",
            content="Content from page two.",
            metadata={"page": 2},
        ),
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert len(chunks) == 2
    assert chunks[0].metadata["page"] == 1
    assert chunks[1].metadata["page"] == 2


def test_empty_table_produces_no_chunks():
    elements = [
        ContentElement(
            type="table",
            content="",
            metadata={
                "page": 1,
                "rows": 0,
                "columns": 0,
            },
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert chunks == []


def test_whitespace_table_produces_no_chunks():
    elements = [
        ContentElement(
            type="table",
            content="   \n\n   ",
            metadata={
                "page": 1,
                "rows": 0,
                "columns": 0,
            },
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert chunks == []


def test_whitespace_text_produces_no_chunks():
    elements = [
        ContentElement(
            type="text",
            content="   \n\n   ",
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
    )

    assert chunks == []


def test_chunk_ids_are_unique_and_sequential():
    elements = [
        ContentElement(
            type="text",
            content=(
                "Authentication is required. "
                "Authorization is required. "
                "Auditing is required. "
                "Monitoring is required."
            ),
        )
    ]

    chunks = chunk_elements(
        elements,
        document_id="doc-001",
        clearance=2,
        trust=1,
        max_chars=50,
    )

    ids = [chunk.id for chunk in chunks]

    assert len(ids) == len(set(ids))

    assert ids == [
        f"doc-001-chunk-{index}"
        for index in range(len(chunks))
    ]
