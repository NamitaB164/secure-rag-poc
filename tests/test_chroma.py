from pathlib import Path

# pyrefly: ignore [missing-import]
import pytest

from secure_rag.models import Chunk
from secure_rag.vectorstore import ChromaVectorStore


def make_chunk(
    index: int,
    content: str,
    clearance: int = 2,
    trust: int = 1,
) -> Chunk:
    return Chunk(
        id=f"doc-001-chunk-{index}",
        document_id="doc-001",
        content=content,
        chunk_index=index,
        clearance=clearance,
        trust=trust,
        metadata={
            "source": "policy.txt",
        },
    )


def test_chroma_store_creates_collection(tmp_path: Path):
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )

    assert store.collection is not None
    assert store.collection.name == "secure_rag"


def test_add_chunks(tmp_path: Path):
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )

    chunks = [
        make_chunk(
            0,
            "Employees must use MFA.",
        ),
        make_chunk(
            1,
            "Passwords must be strong.",
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.add_chunks(chunks, embeddings)

    assert store.collection.count() == 2


def test_add_chunks_preserves_metadata(tmp_path: Path):
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )

    chunk = make_chunk(
        0,
        "Employees must use MFA.",
        clearance=3,
        trust=2,
    )

    store.add_chunks(
        [chunk],
        [[1.0, 0.0, 0.0]],
    )

    result = store.collection.get(
        ids=[chunk.id],
    )

    metadata = result["metadatas"][0]

    assert metadata["document_id"] == "doc-001"
    assert metadata["chunk_index"] == 0
    assert metadata["clearance"] == 3
    assert metadata["trust"] == 2
    assert metadata["source"] == "policy.txt"


def test_search_returns_similar_chunks(tmp_path: Path):
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )

    chunks = [
        make_chunk(
            0,
            "Employees must use MFA.",
        ),
        make_chunk(
            1,
            "Employees must use strong passwords.",
        ),
        make_chunk(
            2,
            "The office cafeteria closes at six.",
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ]

    store.add_chunks(chunks, embeddings)

    result = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        n_results=2,
    )

    assert len(result["ids"][0]) == 2
    assert result["ids"][0][0] == "doc-001-chunk-0"


def test_search_returns_documents(tmp_path: Path):
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )

    chunk = make_chunk(
        0,
        "Employees must use MFA.",
    )

    store.add_chunks(
        [chunk],
        [[1.0, 0.0, 0.0]],
    )

    result = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        n_results=1,
    )

    assert result["documents"][0][0] == "Employees must use MFA."


def test_chunk_embedding_count_must_match(tmp_path: Path):
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )

    chunks = [
        make_chunk(0, "First chunk."),
        make_chunk(1, "Second chunk."),
    ]

    with pytest.raises(
        ValueError,
        match="Number of chunks must match number of embeddings",
    ):
        store.add_chunks(
            chunks,
            [[1.0, 0.0, 0.0]],
        )


def test_empty_chunks_do_nothing(tmp_path: Path):
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )

    store.add_chunks([], [])

    assert store.collection.count() == 0


def test_invalid_n_results(tmp_path: Path):
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )

    with pytest.raises(
        ValueError,
        match="n_results must be greater than 0",
    ):
        store.search(
            query_embedding=[1.0, 0.0, 0.0],
            n_results=0,
        )