from secure_rag.embeddings import EmbeddingModel
from secure_rag.models import Chunk, User
from secure_rag.retrieval import SecureRetriever
from secure_rag.vectorstore import ChromaVectorStore


class FakeEmbeddingModel(EmbeddingModel):
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [1.0, 0.0, 0.0]

class FakeVectorStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        selected = self.chunks

        if where is not None:
            clearance_filter = where.get("clearance")

            if clearance_filter is not None:
                max_clearance = clearance_filter.get("$lte")

                if max_clearance is not None:
                    selected = [
                        chunk
                        for chunk in selected
                        if chunk.clearance <= max_clearance
                    ]

        selected = selected[:n_results]

        return {
            "ids": [[chunk.id for chunk in selected]],
            "documents": [[chunk.content for chunk in selected]],
            "metadatas": [
                [
                    {
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "clearance": chunk.clearance,
                        "trust": chunk.trust,
                        **chunk.metadata,
                    }
                    for chunk in selected
                ]
            ],
        }

def make_chunk(
    index: int,
    content: str,
    clearance: int,
    trust: int,
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


def test_retriever_returns_authorized_chunks():
    chunks = [
        make_chunk(
            0,
            "Public security policy.",
            clearance=1,
            trust=1,
        ),
        make_chunk(
            1,
            "Highly restricted policy.",
            clearance=4,
            trust=1,
        ),
    ]

    retriever = SecureRetriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(chunks),
    )

    user = User(
        id="user-001",
        clearance=2,
    )

    results = retriever.retrieve(
        query="security policy",
        user=user,
    )

    assert len(results) == 1
    assert results[0].content == "Public security policy."


def test_retriever_filters_low_trust_chunks():
    chunks = [
        make_chunk(
            0,
            "Trusted policy.",
            clearance=2,
            trust=2,
        ),
        make_chunk(
            1,
            "Untrusted policy.",
            clearance=2,
            trust=0,
        ),
    ]

    retriever = SecureRetriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(chunks),
        trust_threshold=1,
    )

    user = User(
        id="user-001",
        clearance=2,
    )

    results = retriever.retrieve(
        query="policy",
        user=user,
    )

    assert len(results) == 1
    assert results[0].content == "Trusted policy."


def test_retriever_applies_acl_before_trust():
    chunks = [
        make_chunk(
            0,
            "Authorized and trusted.",
            clearance=2,
            trust=2,
        ),
        make_chunk(
            1,
            "Unauthorized but trusted.",
            clearance=4,
            trust=2,
        ),
        make_chunk(
            2,
            "Authorized but untrusted.",
            clearance=2,
            trust=0,
        ),
    ]

    retriever = SecureRetriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(chunks),
        trust_threshold=1,
    )

    user = User(
        id="user-001",
        clearance=2,
    )

    results = retriever.retrieve(
        query="policy",
        user=user,
    )

    assert len(results) == 1
    assert results[0].content == "Authorized and trusted."
def test_retriever_applies_acl_before_top_k():
    chunks = [
        make_chunk(
            0,
            "Unauthorized but highly relevant.",
            clearance=4,
            trust=1,
        ),
        make_chunk(
            1,
            "Authorized and slightly less relevant.",
            clearance=1,
            trust=1,
        ),
    ]

    retriever = SecureRetriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(chunks),
    )

    user = User(
        id="user-001",
        clearance=1,
    )

    results = retriever.retrieve(
        query="security policy",
        user=user,
        n_results=1,
    )

    assert len(results) == 1
    assert results[0].content == "Authorized and slightly less relevant."


def test_retriever_preserves_chunk_metadata():
    chunk = make_chunk(
        0,
        "Security policy.",
        clearance=2,
        trust=1,
    )

    retriever = SecureRetriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore([chunk]),
    )

    user = User(
        id="user-001",
        clearance=2,
    )

    results = retriever.retrieve(
        query="security",
        user=user,
    )

    assert results[0].metadata["source"] == "policy.txt"


def test_retriever_respects_n_results():
    chunks = [
        make_chunk(i, f"Policy {i}.", 1, 1)
        for i in range(5)
    ]

    retriever = SecureRetriever(
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(chunks),
    )

    user = User(
        id="user-001",
        clearance=1,
    )

    results = retriever.retrieve(
        query="policy",
        user=user,
        n_results=2,
    )

    assert len(results) == 2