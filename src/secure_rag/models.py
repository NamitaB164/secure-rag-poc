from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    source: str
    content: str
    clearance: int
    trust: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    document_id: str
    content: str
    chunk_index: int
    clearance: int
    trust: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class User:
    id: str
    clearance: int
