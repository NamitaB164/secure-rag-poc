from secure_rag.models import Chunk, Document, User

def test_document_creation():
    document = Document(
        id="doc-001",
        source="company_policy.pdf",
        content="Employees must use strong passwords.",
        clearance=2,
        trust=1,
    )

    assert document.id == "doc-001"
    assert document.source == "company_policy.pdf"
    assert document.content == "Employees must use strong passwords."
    assert document.clearance == 2
    assert document.trust == 1


def test_chunk_creation():
    chunk = Chunk(
        id="chunk-001",
        document_id="doc-001",
        content="Employees must use strong passwords.",
        chunk_index=0,
        clearance=2,
        trust=1,
    )

    assert chunk.id == "chunk-001"
    assert chunk.document_id == "doc-001"
    assert chunk.chunk_index == 0
    assert chunk.clearance == 2
    assert chunk.trust == 1


def test_document_supports_clearance_and_trust():
    document = Document(
        id="doc-001",
        source="policy.pdf",
        content="Internal policy.",
        clearance=2,
        trust=1,
    )

    assert document.clearance == 2
    assert document.trust == 1


def test_chunk_inherits_document_security_metadata():
    chunk = Chunk(
        id="chunk-001",
        document_id="doc-001",
        content="Internal policy.",
        chunk_index=0,
        clearance=2,
        trust=2,
    )

    assert chunk.document_id == "doc-001"
    assert chunk.clearance == 2
    assert chunk.trust == 2

def test_user_creation():
    user = User(
        id="user-001",
        clearance=2,
    )

    assert user.id == "user-001"
    assert user.clearance == 2