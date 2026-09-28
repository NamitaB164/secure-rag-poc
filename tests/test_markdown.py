from pathlib import Path

from secure_rag.ingestion.parsers.markdown import parse_markdown


def test_parse_markdown_extracts_text(tmp_path: Path):
    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "Employees must use strong passwords.",
        encoding="utf-8",
    )

    elements = parse_markdown(file_path)

    assert len(elements) == 1
    assert elements[0].type == "text"
    assert elements[0].content == (
        "Employees must use strong passwords."
    )


def test_parse_markdown_detects_heading(tmp_path: Path):
    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "# Authentication\n\n"
        "Employees must use MFA.",
        encoding="utf-8",
    )

    elements = parse_markdown(file_path)

    assert len(elements) == 2

    assert elements[0].content == "Authentication"
    assert elements[0].metadata["heading"] == "Authentication"
    assert elements[0].metadata["heading_level"] == 1

    assert elements[1].content == "Employees must use MFA."
    assert elements[1].metadata["heading"] == "Authentication"
    assert elements[1].metadata["heading_level"] == 1


def test_parse_markdown_supports_heading_levels(tmp_path: Path):
    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "# Authentication\n\n"
        "Authentication overview.\n\n"
        "## Password Policy\n\n"
        "Passwords must be strong.",
        encoding="utf-8",
    )

    elements = parse_markdown(file_path)

    headings = [
        element
        for element in elements
        if "heading_level" in element.metadata
        and element.content == element.metadata["heading"]
    ]

    assert headings[0].metadata["heading_level"] == 1
    assert headings[1].metadata["heading_level"] == 2


def test_parse_markdown_preserves_heading_context(tmp_path: Path):
    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "# Authentication\n\n"
        "Employees must use MFA.\n\n"
        "MFA is required for production.",
        encoding="utf-8",
    )

    elements = parse_markdown(file_path)

    text_elements = [
        element
        for element in elements
        if element.content != "Authentication"
    ]

    assert len(text_elements) == 2

    for element in text_elements:
        assert element.metadata["heading"] == "Authentication"
        assert element.metadata["heading_level"] == 1


def test_parse_markdown_preserves_paragraph_boundaries(tmp_path: Path):
    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "First paragraph.\n\n"
        "Second paragraph.",
        encoding="utf-8",
    )

    elements = parse_markdown(file_path)

    assert len(elements) == 2
    assert elements[0].content == "First paragraph."
    assert elements[1].content == "Second paragraph."


def test_parse_markdown_extracts_table(tmp_path: Path):
    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "# Users\n\n"
        "| Name | Role |\n"
        "| --- | --- |\n"
        "| Alice | Engineer |\n"
        "| Bob | Manager |",
        encoding="utf-8",
    )

    elements = parse_markdown(file_path)

    tables = [
        element
        for element in elements
        if element.type == "table"
    ]

    assert len(tables) == 1
    assert "Alice" in tables[0].content
    assert "Bob" in tables[0].content


def test_parse_markdown_table_preserves_metadata(tmp_path: Path):
    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "| Name | Role |\n"
        "| --- | --- |\n"
        "| Alice | Engineer |",
        encoding="utf-8",
    )

    elements = parse_markdown(file_path)

    table = next(
        element
        for element in elements
        if element.type == "table"
    )

    assert table.metadata["format"] == "md"
    assert table.metadata["rows"] == 2
    assert table.metadata["columns"] == 2


def test_parse_markdown_preserves_heading_context_for_table(
    tmp_path: Path,
):
    file_path = tmp_path / "policy.md"
    file_path.write_text(
        "# Users\n\n"
        "| Name | Role |\n"
        "| --- | --- |\n"
        "| Alice | Engineer |",
        encoding="utf-8",
    )

    elements = parse_markdown(file_path)

    table = next(
        element
        for element in elements
        if element.type == "table"
    )

    assert table.metadata["heading"] == "Users"
    assert table.metadata["heading_level"] == 1


def test_parse_markdown_missing_file_raises():
    path = Path("does-not-exist.md")

    try:
        parse_markdown(path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")