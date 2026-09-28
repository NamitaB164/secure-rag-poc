
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.pdfgen import canvas

from secure_rag.ingestion.elements import ContentElement
from secure_rag.ingestion.loader import load_document
from secure_rag.ingestion.parsers.markdown import parse_markdown
from secure_rag.ingestion.parsers.pdf import parse_pdf
from secure_rag.ingestion.parsers.text import parse_text


# ---------------------------------------------------------------------------
# TXT parsing
# ---------------------------------------------------------------------------


def test_parse_text_file(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello world", encoding="utf-8")

    content, metadata = parse_text(file_path)

    assert content == "Hello world"
    assert metadata["format"] == "txt"


def test_parse_multiline_text(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text(
        "First line\nSecond line\nThird line",
        encoding="utf-8",
    )

    content, _ = parse_text(file_path)

    assert content == "First line\nSecond line\nThird line"


def test_parse_unicode_text(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text(
        "Café\nകേരളം\n日本語",
        encoding="utf-8",
    )

    content, _ = parse_text(file_path)

    assert content == "Café\nകേരളം\n日本語"


def test_parse_text_metadata(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Some content", encoding="utf-8")

    _, metadata = parse_text(file_path)

    assert metadata == {"format": "txt"}


def test_parse_missing_file():
    file_path = Path("does_not_exist.txt")

    with pytest.raises(FileNotFoundError):
        parse_text(file_path)


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------


def test_parse_missing_markdown_file():
    file_path = Path("does_not_exist.md")

    with pytest.raises(FileNotFoundError):
        parse_markdown(file_path)


def test_parse_markdown_file(tmp_path: Path):
    file_path = tmp_path / "test.md"
    file_path.write_text("# Security Policy", encoding="utf-8")

    elements = parse_markdown(file_path)

    assert len(elements) == 1
    assert isinstance(elements[0], ContentElement)
    assert elements[0].content == "Security Policy"
    assert elements[0].metadata["format"] == "md"
    assert elements[0].metadata["heading"] == "Security Policy"
    assert elements[0].metadata["heading_level"] == 1


def test_parse_markdown_preserves_headings(tmp_path: Path):
    file_path = tmp_path / "test.md"
    file_path.write_text(
        "# Main Heading\n\n"
        "## Authentication\n\n"
        "Users must authenticate.",
        encoding="utf-8",
    )

    elements = parse_markdown(file_path)

    assert elements[0].content == "Main Heading"
    assert elements[0].metadata["heading_level"] == 1

    assert elements[1].content == "Authentication"
    assert elements[1].metadata["heading_level"] == 2

    assert elements[2].content == "Users must authenticate."
    assert elements[2].metadata["heading"] == "Authentication"
    assert elements[2].metadata["heading_level"] == 2


def test_parse_markdown_preserves_formatting(tmp_path: Path):
    file_path = tmp_path / "test.md"
    file_path.write_text(
        "**Important** text with `code` and [a link](https://example.com).",
        encoding="utf-8",
    )

    elements = parse_markdown(file_path)

    assert len(elements) == 1
    assert elements[0].content == (
        "**Important** text with `code` and [a link](https://example.com)."
    )


def test_parse_markdown_unicode(tmp_path: Path):
    file_path = tmp_path / "test.md"
    file_path.write_text(
        "# കേരളം\n\n日本語\n\nCafé",
        encoding="utf-8",
    )

    elements = parse_markdown(file_path)

    assert elements[0].content == "കേരളം"
    assert elements[1].content == "日本語"
    assert elements[2].content == "Café"


def test_parse_markdown_metadata(tmp_path: Path):
    file_path = tmp_path / "test.md"
    file_path.write_text("Some content", encoding="utf-8")

    elements = parse_markdown(file_path)

    assert elements[0].metadata["format"] == "md"


def test_load_markdown_file(tmp_path: Path):
    file_path = tmp_path / "test.md"
    file_path.write_text("# Hello Markdown", encoding="utf-8")

    elements = load_document(file_path)

    assert len(elements) == 1
    assert isinstance(elements[0], ContentElement)
    assert elements[0].content == "Hello Markdown"
    assert elements[0].metadata["format"] == "md"
    assert elements[0].metadata["heading"] == "Hello Markdown"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_load_unsupported_file(tmp_path: Path):
    file_path = tmp_path / "test.xyz"
    file_path.write_text("Unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(file_path)


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------


def create_pdf(path: Path, page_count: int = 1) -> None:
    writer = PdfWriter()

    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)

    with path.open("wb") as file:
        writer.write(file)


def create_table_pdf(path: Path) -> None:
    document = SimpleDocTemplate(str(path), pagesize=letter)

    data = [
        ["Name", "Age", "Department"],
        ["Alice", "24", "Engineering"],
        ["Bob", "27", "Security"],
    ]

    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    document.build([table])


def create_image_pdf(path: Path) -> None:
    image_path = path.parent / "test-image.png"

    image = Image.new("RGB", (100, 80))
    image.save(image_path)

    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawImage(
        str(image_path),
        100,
        500,
        width=100,
        height=80,
    )
    pdf.save()


def create_scanned_pdf(path: Path) -> None:
    image_path = path.parent / "scanned-page.png"

    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(
        "C:/Windows/Fonts/arial.ttf",
        48,
    )

    draw.text(
        (100, 100),
        "Secure RAG OCR Test",
        fill="black",
        font=font,
    )

    draw.text(
        (100, 200),
        "This text exists only inside an image.",
        fill="black",
        font=font,
    )

    image.save(image_path)

    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawImage(
        str(image_path),
        0,
        0,
        width=letter[0],
        height=letter[1],
    )
    pdf.save()


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------


def test_parse_pdf_returns_elements(tmp_path: Path):
    file_path = tmp_path / "test.pdf"
    create_pdf(file_path)

    elements = parse_pdf(file_path)

    assert isinstance(elements, list)
    assert all(isinstance(element, ContentElement) for element in elements)


def test_parse_pdf_preserves_page_number(tmp_path: Path):
    file_path = tmp_path / "test.pdf"
    create_pdf(file_path, page_count=3)

    elements = parse_pdf(file_path)

    pages = {
        element.metadata["page"]
        for element in elements
        if "page" in element.metadata
    }

    assert pages <= {1, 2, 3}


def test_parse_pdf_missing_file():
    file_path = Path("does_not_exist.pdf")

    with pytest.raises(FileNotFoundError):
        parse_pdf(file_path)


def test_parse_pdf_extracts_tables(tmp_path: Path):
    file_path = tmp_path / "table.pdf"
    create_table_pdf(file_path)

    elements = parse_pdf(file_path)

    tables = [
        element
        for element in elements
        if element.type == "table"
    ]

    assert len(tables) == 1


def test_parse_pdf_table_preserves_content(tmp_path: Path):
    file_path = tmp_path / "table.pdf"
    create_table_pdf(file_path)

    elements = parse_pdf(file_path)

    tables = [
        element
        for element in elements
        if element.type == "table"
    ]

    assert "Alice" in tables[0].content
    assert "Engineering" in tables[0].content
    assert "Security" in tables[0].content


def test_parse_pdf_table_preserves_page_number(tmp_path: Path):
    file_path = tmp_path / "table.pdf"
    create_table_pdf(file_path)

    elements = parse_pdf(file_path)

    tables = [
        element
        for element in elements
        if element.type == "table"
    ]

    assert tables[0].metadata["page"] == 1


def test_parse_pdf_table_has_dimensions(tmp_path: Path):
    file_path = tmp_path / "table.pdf"
    create_table_pdf(file_path)

    elements = parse_pdf(file_path)

    tables = [
        element
        for element in elements
        if element.type == "table"
    ]

    assert tables[0].metadata["rows"] == 3
    assert tables[0].metadata["columns"] == 3


def test_parse_pdf_does_not_duplicate_table_content(tmp_path: Path):
    file_path = tmp_path / "table.pdf"
    create_table_pdf(file_path)

    elements = parse_pdf(file_path)

    text_elements = [
        element
        for element in elements
        if element.type == "text"
    ]

    table_elements = [
        element
        for element in elements
        if element.type == "table"
    ]

    assert len(table_elements) == 1

    table_content = table_elements[0].content

    for element in text_elements:
        assert "Alice" not in element.content
        assert "Engineering" not in element.content
        assert "Security" not in element.content

    assert "Alice" in table_content


def test_parse_pdf_extracts_images(tmp_path: Path):
    file_path = tmp_path / "image.pdf"
    create_image_pdf(file_path)

    elements = parse_pdf(file_path)

    images = [
        element
        for element in elements
        if element.type == "image"
    ]

    assert len(images) == 1


def test_parse_pdf_image_preserves_page_number(tmp_path: Path):
    file_path = tmp_path / "image.pdf"
    create_image_pdf(file_path)

    elements = parse_pdf(file_path)

    images = [
        element
        for element in elements
        if element.type == "image"
    ]

    assert images[0].metadata["page"] == 1


def test_parse_pdf_image_has_metadata(tmp_path: Path):
    file_path = tmp_path / "image.pdf"
    create_image_pdf(file_path)

    elements = parse_pdf(file_path)

    images = [
        element
        for element in elements
        if element.type == "image"
    ]

    assert images[0].metadata["width"] > 0
    assert images[0].metadata["height"] > 0
    assert images[0].metadata["extension"] == "png"


def test_parse_pdf_extracts_image_asset(tmp_path: Path):
    file_path = tmp_path / "image.pdf"
    asset_dir = tmp_path / "assets"

    create_image_pdf(file_path)

    elements = parse_pdf(
        file_path,
        asset_dir=asset_dir,
    )

    images = [
        element
        for element in elements
        if element.type == "image"
    ]

    asset_path = Path(images[0].metadata["asset_path"])

    assert asset_path.exists()
    assert asset_path.suffix == ".png"


def test_parse_pdf_image_asset_has_expected_name(tmp_path: Path):
    file_path = tmp_path / "image.pdf"
    asset_dir = tmp_path / "assets"

    create_image_pdf(file_path)

    elements = parse_pdf(
        file_path,
        asset_dir=asset_dir,
    )

    images = [
        element
        for element in elements
        if element.type == "image"
    ]

    asset_path = Path(images[0].metadata["asset_path"])

    assert asset_path.name == "page-1-image-0.png"


def test_parse_pdf_without_asset_dir_does_not_write_image(
    tmp_path: Path,
):
    file_path = tmp_path / "image.pdf"
    create_image_pdf(file_path)

    elements = parse_pdf(file_path)

    images = [
        element
        for element in elements
        if element.type == "image"
    ]

    assert "asset_path" not in images[0].metadata


# ---------------------------------------------------------------------------
# PDF OCR
# ---------------------------------------------------------------------------


def test_parse_pdf_ocr_extracts_scanned_text(tmp_path: Path):
    file_path = tmp_path / "scanned.pdf"
    create_scanned_pdf(file_path)

    elements = parse_pdf(file_path)

    text_elements = [
        element
        for element in elements
        if element.type == "text"
    ]

    content = " ".join(
        element.content
        for element in text_elements
    )

    assert "Secure RAG OCR Test" in content


def test_parse_pdf_ocr_preserves_page_number(tmp_path: Path):
    file_path = tmp_path / "scanned.pdf"
    create_scanned_pdf(file_path)

    elements = parse_pdf(file_path)

    text_elements = [
        element
        for element in elements
        if (
            element.type == "text"
            and "Secure RAG OCR Test" in element.content
        )
    ]

    assert text_elements[0].metadata["page"] == 1
