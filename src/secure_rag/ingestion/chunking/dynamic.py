from secure_rag.ingestion.elements import ContentElement
from secure_rag.models import Chunk


def chunk_elements(
    elements: list[ContentElement],
    document_id: str,
    clearance: int,
    trust: int,
    max_chars: int = 1000,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    grouped_elements = _group_heading_sections(elements)

    for element in grouped_elements:
        if not element.content and element.type != "image":
            continue

        if element.type == "image":
            chunks.append(
                Chunk(
                    id=f"{document_id}-chunk-{len(chunks)}",
                    document_id=document_id,
                    content=element.content,
                    chunk_index=len(chunks),
                    clearance=clearance,
                    trust=trust,
                    metadata=dict(element.metadata),
                )
            )
            continue

        if element.type == "table":
            chunks.extend(
        _chunk_table(
            element=element,
            document_id=document_id,
            clearance=clearance,
            trust=trust,
            max_chars=max_chars,
            start_index=len(chunks),
        )
    )
            continue

        chunks.extend(
            _chunk_text(
                element=element,
                document_id=document_id,
                clearance=clearance,
                trust=trust,
                max_chars=max_chars,
                start_index=len(chunks),
                overlap_sentences=overlap_sentences,
            )
        )

    return chunks

def _chunk_text(
    element: ContentElement,
    document_id: str,
    clearance: int,
    trust: int,
    max_chars: int,
    start_index: int,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    text = element.content.strip()

    if not text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    # Headings define a section, so the heading and its following
    # content should remain together when they fit.
    if element.metadata.get("heading"):
        return _split_section(
            paragraphs=paragraphs,
            element=element,
            document_id=document_id,
            clearance=clearance,
            trust=trust,
            max_chars=max_chars,
            start_index=start_index,
            overlap_sentences=overlap_sentences,
        )

    chunks: list[Chunk] = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(
                _build_chunk(
                    document_id=document_id,
                    content=paragraph,
                    chunk_index=start_index + len(chunks),
                    clearance=clearance,
                    trust=trust,
                    metadata=element.metadata,
                )
            )
            continue

        chunks.extend(
            _split_long_text(
                paragraph=paragraph,
                document_id=document_id,
                clearance=clearance,
                trust=trust,
                max_chars=max_chars,
                start_index=start_index + len(chunks),
                metadata=element.metadata,
                overlap_sentences=overlap_sentences,
            )
        )

    return chunks
def _split_section(
    paragraphs: list[str],
    element: ContentElement,
    document_id: str,
    clearance: int,
    trust: int,
    max_chars: int,
    start_index: int,
    overlap_sentences: int,
) -> list[Chunk]:
    text = "\n\n".join(paragraphs)

    if len(text) <= max_chars:
        return [
            _build_chunk(
                document_id=document_id,
                content=text,
                chunk_index=start_index,
                clearance=clearance,
                trust=trust,
                metadata=element.metadata,
            )
        ]

    return _split_long_text(
        paragraph=text,
        document_id=document_id,
        clearance=clearance,
        trust=trust,
        max_chars=max_chars,
        start_index=start_index,
        metadata=element.metadata,
        overlap_sentences=overlap_sentences,
    )

def _split_long_text(
    paragraph: str,
    document_id: str,
    clearance: int,
    trust: int,
    max_chars: int,
    start_index: int,
    metadata: dict[str, object],
    overlap_sentences: int = 0,
) -> list[Chunk]:
    return _split_paragraph(
        paragraph=paragraph,
        document_id=document_id,
        clearance=clearance,
        trust=trust,
        max_chars=max_chars,
        start_index=start_index,
        metadata=metadata,
        overlap_sentences=overlap_sentences,
    )

def _split_paragraph(
    paragraph: str,
    document_id: str,
    clearance: int,
    trust: int,
    max_chars: int,
    start_index: int,
    metadata: dict[str, object],
    overlap_sentences: int = 0,
) -> list[Chunk]:
    sentences = _split_sentences(paragraph)

    chunks: list[Chunk] = []
    current: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > max_chars:
            if current:
                chunks.append(
                    _build_chunk(
                        document_id=document_id,
                        content=" ".join(current),
                        chunk_index=start_index + len(chunks),
                        clearance=clearance,
                        trust=trust,
                        metadata=metadata,
                    )
                )

                current = []
                current_length = 0

            chunks.extend(
                _split_by_words(
                    text=sentence,
                    document_id=document_id,
                    clearance=clearance,
                    trust=trust,
                    max_chars=max_chars,
                    start_index=start_index + len(chunks),
                    metadata=metadata,
                )
            )
            continue

        additional_length = (
            len(sentence)
            if not current
            else len(sentence) + 1
        )

        if current and current_length + additional_length > max_chars:
            chunks.append(
                _build_chunk(
                    document_id=document_id,
                    content=" ".join(current),
                    chunk_index=start_index + len(chunks),
                    clearance=clearance,
                    trust=trust,
                    metadata=metadata,
                )
            )

            # Carry the final N sentences into the next chunk.
            if overlap_sentences > 0:
                overlap = current[-overlap_sentences:]
                overlap_length = len(" ".join(overlap))

                if overlap_length <= max_chars:
                    current = overlap
                    current_length = overlap_length
                else:
                    current = []
                    current_length = 0
            else:
                current = []
                current_length = 0

        current.append(sentence)

        current_length = len(" ".join(current))

    if current:
        chunks.append(
            _build_chunk(
                document_id=document_id,
                content=" ".join(current),
                chunk_index=start_index + len(chunks),
                clearance=clearance,
                trust=trust,
                metadata=metadata,
            )
        )

    return chunks

def _split_by_words(
    text: str,
    document_id: str,
    clearance: int,
    trust: int,
    max_chars: int,
    start_index: int,
    metadata: dict[str, object],
) -> list[Chunk]:
    words = text.split()
    chunks: list[Chunk] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        additional_length = (
            len(word)
            if not current
            else len(word) + 1
        )

        if current and current_length + additional_length > max_chars:
            chunks.append(
                _build_chunk(
                    document_id=document_id,
                    content=" ".join(current),
                    chunk_index=start_index + len(chunks),
                    clearance=clearance,
                    trust=trust,
                    metadata=metadata,
                )
            )
            current = []
            current_length = 0

        current.append(word)
        current_length += (
            len(word)
            if len(current) == 1
            else len(word) + 1
        )

    if current:
        chunks.append(
            _build_chunk(
                document_id=document_id,
                content=" ".join(current),
                chunk_index=start_index + len(chunks),
                clearance=clearance,
                trust=trust,
                metadata=metadata,
            )
        )

    return chunks


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []

    for word in text.split():
        current.append(word)

        if word.endswith((".", "!", "?")):
            sentences.append(" ".join(current))
            current = []

    if current:
        sentences.append(" ".join(current))

    return sentences


def _chunk_table(
    element: ContentElement,
    document_id: str,
    clearance: int,
    trust: int,
    max_chars: int,
    start_index: int,
) -> list[Chunk]:
    rows = [
        row.strip()
        for row in element.content.splitlines()
        if row.strip()
    ]

    if not rows:
        return []

    header = rows[0]

    if len(element.content) <= max_chars:
        return [
            _build_chunk(
                document_id=document_id,
                content=element.content.strip(),
                chunk_index=start_index,
                clearance=clearance,
                trust=trust,
                metadata=element.metadata,
            )
        ]

    chunks: list[Chunk] = []
    current_rows = [header]
    current_length = len(header)

    for row in rows[1:]:
        additional_length = len(row) + 1

        if (
            len(current_rows) > 1
            and current_length + additional_length > max_chars
        ):
            chunks.append(
                _build_chunk(
                    document_id=document_id,
                    content="\n".join(current_rows),
                    chunk_index=start_index + len(chunks),
                    clearance=clearance,
                    trust=trust,
                    metadata=element.metadata,
                )
            )

            current_rows = [header]
            current_length = len(header)

        current_rows.append(row)
        current_length += additional_length

    if len(current_rows) > 1:
        chunks.append(
            _build_chunk(
                document_id=document_id,
                content="\n".join(current_rows),
                chunk_index=start_index + len(chunks),
                clearance=clearance,
                trust=trust,
                metadata=element.metadata,
            )
        )

    return chunks


def _build_chunk(
    document_id: str,
    content: str,
    chunk_index: int,
    clearance: int,
    trust: int,
    metadata: dict[str, object],
) -> Chunk:
    return Chunk(
        id=f"{document_id}-chunk-{chunk_index}",
        document_id=document_id,
        content=content,
        chunk_index=chunk_index,
        clearance=clearance,
        trust=trust,
        metadata=dict(metadata),
    )
def _group_heading_sections(
    elements: list[ContentElement],
) -> list[ContentElement]:
    grouped: list[ContentElement] = []
    current: ContentElement | None = None

    for element in elements:
        heading = element.metadata.get("heading")

        if (
            current is not None
            and heading
            and current.metadata.get("heading") == heading
            and element.type == current.type == "text"
        ):
            current.content = f"{current.content}\n\n{element.content}"
            continue

        if current is not None:
            grouped.append(current)

        current = ContentElement(
            type=element.type,
            content=element.content,
            metadata=dict(element.metadata),
        )

    if current is not None:
        grouped.append(current)

    return grouped