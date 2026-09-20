from dataclasses import dataclass, field


@dataclass
class PDFElement:
    type: str
    page: int
    content: str
    metadata: dict[str, object] = field(default_factory=dict)
