from secure_rag.models import Chunk, User
from secure_rag.retrieval.acl import filter_authorized_chunks


def test_user_can_access_lower_clearance_chunks():
    user = User(id="user-001", clearance=2)

    chunks = [
        Chunk(
            id="chunk-001",
            document_id="doc-001",
            content="Public information.",
            chunk_index=0,
            clearance=1,
            trust=1,
        ),
        Chunk(
            id="chunk-002",
            document_id="doc-002",
            content="Internal information.",
            chunk_index=0,
            clearance=2,
            trust=1,
        ),
        Chunk(
            id="chunk-003",
            document_id="doc-003",
            content="Confidential information.",
            chunk_index=0,
            clearance=3,
            trust=1,
        ),
    ]

    authorized = filter_authorized_chunks(user, chunks)

    assert [chunk.id for chunk in authorized] == [
        "chunk-001",
        "chunk-002",
    ]


def test_user_cannot_access_higher_clearance_chunks():
    user = User(id="user-001", clearance=1)

    chunks = [
        Chunk(
            id="chunk-001",
            document_id="doc-001",
            content="Internal information.",
            chunk_index=0,
            clearance=2,
            trust=1,
        ),
    ]

    authorized = filter_authorized_chunks(user, chunks)

    assert authorized == []


def test_user_can_access_same_clearance():
    user = User(id="user-001", clearance=3)

    chunk = Chunk(
        id="chunk-001",
        document_id="doc-001",
        content="Confidential information.",
        chunk_index=0,
        clearance=3,
        trust=1,
    )

    authorized = filter_authorized_chunks(user, [chunk])

    assert authorized == [chunk]