from pathlib import Path

from secure_rag.ingestion.parsers.text import parse_text


def load_document(path: Path) -> tuple[str, dict[str, object]]:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return parse_text(path)

    raise ValueError(f"Unsupported file type: {suffix}")


from pathlib import Path

from secure_rag.ingestion.parsers.markdown import parse_markdown


def load_document(path: Path) -> tuple[str, dict[str, object]]:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return parse_text(path)

    if suffix == ".md":
        return parse_markdown(path)

    raise ValueError(f"Unsupported file type: {suffix}")
