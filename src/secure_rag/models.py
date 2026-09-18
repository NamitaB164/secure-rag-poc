from dataclasses import dataclass


@dataclass
class Document:
    id: str
    source: str
    content: str
    clearance: int
    trust: int


@dataclass
class Chunk:
    id: str
    document_id: str
    content: str
    chunk_index: int
    clearance: int
    trust: int