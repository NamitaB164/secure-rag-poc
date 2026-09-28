from pathlib import Path

from secure_rag.ingestion.elements import ContentElement


def parse_markdown(path: Path) -> list[ContentElement]:
    if not path.exists():
        raise FileNotFoundError(path)

    content = path.read_text(encoding="utf-8-sig")

    lines = content.splitlines()
    elements: list[ContentElement] = []

    current_heading: str | None = None
    current_heading_level: int | None = None

    index = 0

    while index < len(lines):
        line = lines[index].strip()

        if not line:
            index += 1
            continue

        # Heading
        heading = _parse_heading(line)

        if heading is not None:
            heading_text, heading_level = heading

            current_heading = heading_text
            current_heading_level = heading_level

            elements.append(
                ContentElement(
                    type="text",
                    content=heading_text,
                    metadata={
                        "format": "md",
                        "heading": heading_text,
                        "heading_level": heading_level,
                    },
                )
            )

            index += 1
            continue

        # Markdown table
        if _is_table_start(lines, index):
            table_lines = [lines[index].strip()]

            index += 1

            while index < len(lines):
                candidate = lines[index].strip()

                if not candidate or "|" not in candidate:
                    break

                table_lines.append(candidate)
                index += 1

            rows = _parse_table_rows(table_lines)

            if rows:
                metadata: dict[str, object] = {
                    "format": "md",
                    "rows": len(rows),
                    "columns": len(rows[0]),
                }

                if current_heading is not None:
                    metadata["heading"] = current_heading
                    metadata["heading_level"] = current_heading_level

                elements.append(
                    ContentElement(
                        type="table",
                        content="\n".join(
                            " | ".join(row)
                            for row in rows
                        ),
                        metadata=metadata,
                    )
                )

            continue

        # Paragraph
        paragraph_lines = [line]
        index += 1

        while index < len(lines):
            candidate = lines[index].strip()

            if not candidate:
                break

            if _parse_heading(candidate) is not None:
                break

            if _is_table_start(lines, index):
                break

            paragraph_lines.append(candidate)
            index += 1

        metadata = {
            "format": "md",
        }

        if current_heading is not None:
            metadata["heading"] = current_heading
            metadata["heading_level"] = current_heading_level

        elements.append(
            ContentElement(
                type="text",
                content=" ".join(paragraph_lines),
                metadata=metadata,
            )
        )

    return elements


def _parse_heading(line: str) -> tuple[str, int] | None:
    if not line.startswith("#"):
        return None

    level = 0

    while level < len(line) and line[level] == "#":
        level += 1

    if level == 0 or level > 6:
        return None

    if level >= len(line) or line[level] != " ":
        return None

    heading = line[level:].strip()

    if not heading:
        return None

    return heading, level


def _is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False

    header = lines[index].strip()
    separator = lines[index + 1].strip()

    if "|" not in header:
        return False

    return _is_table_separator(separator)


def _is_table_separator(line: str) -> bool:
    cells = [
        cell.strip()
        for cell in line.strip("|").split("|")
    ]

    if not cells:
        return False

    return all(
        len(cell) >= 3
        and cell.replace("-", "").replace(":", "").strip() == ""
        for cell in cells
    )


def _parse_table_rows(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []

    for line in lines:
        if _is_table_separator(line):
            continue

        cells = [
            cell.strip()
            for cell in line.strip("|").split("|")
        ]

        rows.append(cells)

    return rows