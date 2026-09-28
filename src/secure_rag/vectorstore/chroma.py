from pathlib import Path
# pyrefly: ignore [missing-import]
import chromadb

from secure_rag.models import Chunk


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "secure_rag",
    ) -> None:
        self.persist_directory = Path(persist_directory)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "clearance": chunk.clearance,
                    "trust": chunk.trust,
                    **chunk.metadata,
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> dict:
        if n_results <= 0:
            raise ValueError("n_results must be greater than 0.")

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )