from pathlib import Path

from secure_rag.embeddings import EmbeddingModel
from secure_rag.ingestion.elements import ContentElement
from secure_rag.ingestion.service import DocumentIngester


class FakeEmbeddingModel(EmbeddingModel):
    def __init__(self) -> None:
        self.received_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.received_texts = texts
        return [[float(index)] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.received_chunks = []
        self.received_embeddings = []

    def add_chunks(self, chunks, embeddings) -> None:
        self.received_chunks = chunks
        self.received_embeddings = embeddings


def test_ingest_loads_chunks_and_indexes_text(
    monkeypatch,
    tmp_path: Path,
) -> None:
    elements = [
        ContentElement(
            type="text",
            content="This is document content.",
            metadata={"format": "txt"},
        )
    ]

    monkeypatch.setattr(
        "secure_rag.ingestion.service.load_document",
        lambda path, asset_dir=None: elements,
    )

    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore()

    ingester = DocumentIngester(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    chunks = ingester.ingest(
        path=tmp_path / "document.txt",
        document_id="doc-1",
        clearance=2,
        trust=2,
    )

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].clearance == 2
    assert chunks[0].trust == 2

    assert embedding_model.received_texts == [
        "This is document content."
    ]

    assert vector_store.received_chunks == chunks
    assert vector_store.received_embeddings == [[0.0]]


def test_ingest_does_not_embed_empty_chunks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    elements = [
        ContentElement(
            type="text",
            content="Actual text.",
        ),
        ContentElement(
            type="image",
            content="",
            metadata={"asset_path": "page-1-image-1.png"},
        ),
    ]

    monkeypatch.setattr(
        "secure_rag.ingestion.service.load_document",
        lambda path, asset_dir=None: elements,
    )

    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore()

    ingester = DocumentIngester(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    chunks = ingester.ingest(
        path=tmp_path / "document.pdf",
        document_id="doc-1",
        clearance=1,
        trust=2,
    )

    assert len(chunks) == 2

    assert embedding_model.received_texts == [
        "Actual text."
    ]

    assert len(vector_store.received_chunks) == 1
    assert vector_store.received_chunks[0].content == "Actual text."


def test_ingest_returns_chunks_without_indexing_when_all_are_empty(
    monkeypatch,
    tmp_path: Path,
) -> None:
    elements = [
        ContentElement(
            type="image",
            content="",
            metadata={"asset_path": "page-1-image-1.png"},
        )
    ]

    monkeypatch.setattr(
        "secure_rag.ingestion.service.load_document",
        lambda path, asset_dir=None: elements,
    )

    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore()

    ingester = DocumentIngester(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    chunks = ingester.ingest(
        path=tmp_path / "document.pdf",
        document_id="doc-1",
        clearance=1,
        trust=2,
    )

    assert len(chunks) == 1
    assert embedding_model.received_texts == []
    assert vector_store.received_chunks == []


def test_ingest_passes_asset_dir_to_loader(
    monkeypatch,
    tmp_path: Path,
) -> None:
    received_asset_dir = None

    def fake_load_document(
        path: Path,
        asset_dir: Path | None = None,
    ) -> list[ContentElement]:
        nonlocal received_asset_dir
        received_asset_dir = asset_dir

        return [
            ContentElement(
                type="text",
                content="PDF text.",
            )
        ]

    monkeypatch.setattr(
        "secure_rag.ingestion.service.load_document",
        fake_load_document,
    )

    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore()

    ingester = DocumentIngester(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    asset_dir = tmp_path / "assets"

    ingester.ingest(
        path=tmp_path / "document.pdf",
        document_id="doc-1",
        clearance=1,
        trust=2,
        asset_dir=asset_dir,
    )

    assert received_asset_dir == asset_dir