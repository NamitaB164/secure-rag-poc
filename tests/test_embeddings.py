# pyrefly: ignore [missing-import]
import pytest

from secure_rag.embeddings import SentenceTransformerEmbedding


@pytest.fixture(scope="module")
def embedding_model():
    return SentenceTransformerEmbedding()


def test_embed_documents_returns_embeddings(embedding_model):
    texts = [
        "Employees must use strong passwords.",
        "All production systems require MFA.",
    ]

    embeddings = embedding_model.embed_documents(texts)

    assert len(embeddings) == 2
    assert all(isinstance(embedding, list) for embedding in embeddings)
    assert all(len(embedding) > 0 for embedding in embeddings)


def test_embedding_dimensions_are_consistent(embedding_model):
    texts = [
        "Authentication policy.",
        "Authorization policy.",
        "Network security policy.",
    ]

    embeddings = embedding_model.embed_documents(texts)

    dimensions = {
        len(embedding)
        for embedding in embeddings
    }

    assert len(dimensions) == 1


def test_query_embedding_has_same_dimension_as_documents(
    embedding_model,
):
    documents = embedding_model.embed_documents(
        ["Employees must use MFA."]
    )

    query = embedding_model.embed_query(
        "What authentication controls are required?"
    )

    assert len(query) == len(documents[0])


def test_empty_documents_return_empty_list(embedding_model):
    embeddings = embedding_model.embed_documents([])

    assert embeddings == []


def test_empty_query_raises_error(embedding_model):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embedding_model.embed_query("")


def test_whitespace_query_raises_error(embedding_model):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embedding_model.embed_query("   ")


def test_same_text_produces_same_embedding(embedding_model):
    first = embedding_model.embed_query(
        "Employees must use MFA."
    )

    second = embedding_model.embed_query(
        "Employees must use MFA."
    )

    assert first == pytest.approx(second)