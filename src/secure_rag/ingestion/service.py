from pathlib import Path

from secure_rag.embeddings import EmbeddingModel
from secure_rag.ingestion.chunking.dynamic import chunk_elements
from secure_rag.ingestion.loader import load_document
from secure_rag.models import Chunk
from secure_rag.vectorstore import ChromaVectorStore


class DocumentIngester:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def ingest(
        self,
        *,
        path: Path,
        document_id: str,
        clearance: int,
        trust: int,
        max_chars: int = 1000,
        overlap_sentences: int = 1,
        asset_dir: Path | None = None,
    ) -> list[Chunk]:
        elements = load_document(
            path,
            asset_dir=asset_dir,
        )

        chunks = chunk_elements(
            elements=elements,
            document_id=document_id,
            clearance=clearance,
            trust=trust,
            max_chars=max_chars,
            overlap_sentences=overlap_sentences,
        )

        indexable_chunks = [
            chunk
            for chunk in chunks
            if chunk.content.strip()
        ]

        if not indexable_chunks:
            return chunks

        embeddings = self.embedding_model.embed_documents(
            [chunk.content for chunk in indexable_chunks]
        )

        self.vector_store.add_chunks(
            chunks=indexable_chunks,
            embeddings=embeddings,
        )

        return chunks