from pathlib import Path

from secure_rag.ingestion.elements import ContentElement
from secure_rag.ingestion.parsers.markdown import parse_markdown
from secure_rag.ingestion.parsers.pdf import parse_pdf
from secure_rag.ingestion.parsers.text import parse_text


def load_document(
    path: Path,
    asset_dir: Path | None = None,
) -> list[ContentElement]:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        content, metadata = parse_text(path)

        return [
            ContentElement(
                type="text",
                content=content,
                metadata=metadata,
            )
        ]

    if suffix == ".md":
        return parse_markdown(path)

    if suffix == ".pdf":
        return parse_pdf(path, asset_dir=asset_dir)

    raise ValueError(f"Unsupported file type: {suffix}")