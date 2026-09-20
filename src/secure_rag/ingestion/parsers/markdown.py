from pathlib import Path


def parse_markdown(path: Path) -> tuple[str, dict[str, object]]:
    content = path.read_text(encoding="utf-8-sig")

    return content, {"format": "md"}
