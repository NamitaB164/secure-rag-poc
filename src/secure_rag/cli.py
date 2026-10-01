from pathlib import Path

from secure_rag.classifiers.input import InputClassifier
from secure_rag.classifiers.output import OutputClassifier
from secure_rag.embeddings import SentenceTransformerEmbedding
from secure_rag.generation.generator import GroundedGenerator
from secure_rag.ingestion.service import DocumentIngester
from secure_rag.logging.audit import AuditLogger
from secure_rag.models import User
from secure_rag.pipeline import SecureRAGPipeline
from secure_rag.retrieval.retriever import SecureRetriever
from secure_rag.vectorstore import ChromaVectorStore


def build_components() -> tuple[SecureRAGPipeline, DocumentIngester]:
    embedding_model = SentenceTransformerEmbedding()

    vector_store = ChromaVectorStore(
        persist_directory=Path("data/chroma"),
    )

    retriever = SecureRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    input_classifier = InputClassifier()
    generator = GroundedGenerator()
    output_classifier = OutputClassifier()

    audit_logger = AuditLogger(
        Path("data/audit/audit.jsonl"),
    )

    pipeline = SecureRAGPipeline(
        retriever=retriever,
        input_classifier=input_classifier,
        generator=generator,
        output_classifier=output_classifier,
        audit_logger=audit_logger,
    )

    ingester = DocumentIngester(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    return pipeline, ingester


def prompt_clearance(prompt: str) -> int:
    while True:
        value = input(prompt).strip()

        try:
            clearance = int(value)
        except ValueError:
            print("Must be an integer.")
            continue

        if clearance < 1:
            print("Must be at least 1.")
            continue

        return clearance


def prompt_trust() -> int:
    while True:
        value = input("Document trust (1=trusted, 2=untrusted, 3=hostile): ").strip()

        try:
            trust = int(value)
        except ValueError:
            print("Trust must be 1, 2, or 3.")
            continue

        if trust not in {1, 2, 3}:
            print("Trust must be 1, 2, or 3.")
            continue

        return trust


def ingest_document(ingester: DocumentIngester) -> None:
    print()
    path = Path(input("Document path: ").strip())

    if not path.exists():
        print(f"File not found: {path}")
        return

    if not path.is_file():
        print(f"Path is not a file: {path}")
        return

    document_id = input("Document ID: ").strip()

    if not document_id:
        print("Document ID cannot be empty.")
        return

    clearance = prompt_clearance("Document clearance: ")
    trust = prompt_trust()

    asset_dir = None

    if path.suffix.lower() == ".pdf":
        asset_dir = Path("data/documents/assets")

    print()
    print("Ingesting...")

    try:
        chunks = ingester.ingest(
            path=path,
            document_id=document_id,
            clearance=clearance,
            trust=trust,
            asset_dir=asset_dir,
        )
    except (OSError, ValueError) as exc:
        print(f"Ingestion failed: {exc}")
        return

    indexed_chunks = [
        chunk
        for chunk in chunks
        if chunk.content.strip()
    ]

    print()
    print("Ingestion complete.")
    print(f"Total chunks: {len(chunks)}")
    print(f"Indexed chunks: {len(indexed_chunks)}")
    print()


def query_loop(
    pipeline: SecureRAGPipeline,
    user: User,
) -> None:
    while True:
        query = input("Query: ").strip()

        if query.lower() == "exit":
            return

        if not query:
            print("Query cannot be empty.")
            continue

        print()
        print("Running security checks...")

        result = pipeline.run(
            query=query,
            user=user,
        )

        print()
        print("Answer:")
        print(result.response)
        print()
        print(
            f"Security: "
            f"{result.security_action.decision.value.upper()}"
        )
        print(
            f"Retrieved chunks: "
            f"{len(result.retrieved_chunk_ids)}"
        )
        print()


def main() -> None:
    print("Secure RAG POC")
    print("==============")
    print()

    user_id = input("User ID: ").strip()

    if not user_id:
        print("User ID cannot be empty.")
        return

    clearance = prompt_clearance("Clearance: ")

    user = User(
        id=user_id,
        clearance=clearance,
    )

    pipeline, ingester = build_components()

    while True:
        print("1. Ingest document")
        print("2. Query")
        print("3. Exit")
        print()

        choice = input("Choice: ").strip()

        if choice == "1":
            ingest_document(ingester)

        elif choice == "2":
            print()
            print("Type 'exit' to return to the menu.")
            print()
            query_loop(pipeline, user)

        elif choice == "3":
            print("Goodbye.")
            break

        else:
            print("Invalid choice. Enter 1, 2, or 3.")

        print()


if __name__ == "__main__":
    main()