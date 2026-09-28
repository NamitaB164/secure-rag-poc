from secure_rag.embeddings import SentenceTransformerEmbedding
from secure_rag.models import Chunk, User
from secure_rag.retrieval.acl import filter_authorized_chunks
from secure_rag.vectorstore import ChromaVectorStore


class SecureRetriever:
    def __init__(
        self,
        embedding_model: SentenceTransformerEmbedding,
        vector_store: ChromaVectorStore,
        trust_threshold: int = 0,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.trust_threshold = trust_threshold

    def retrieve(
        self,
        query: str,
        user: User,
        n_results: int = 5,
    ) -> list[Chunk]:
        query_embedding = self.embedding_model.embed(query)

        result = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
        )

        candidates = self._result_to_chunks(result)

        authorized = filter_authorized_chunks(
            user=user,
            chunks=candidates,
        )

        return [
            chunk
            for chunk in authorized
            if chunk.trust >= self.trust_threshold
        ]

    @staticmethod
    def _result_to_chunks(result: dict) -> list[Chunk]:
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        chunks: list[Chunk] = []

        for chunk_id, document, metadata in zip(
            ids,
            documents,
            metadatas,
        ):
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=metadata["document_id"],
                    content=document,
                    chunk_index=metadata["chunk_index"],
                    clearance=metadata["clearance"],
                    trust=metadata["trust"],
                    metadata={
                        key: value
                        for key, value in metadata.items()
                        if key
                        not in {
                            "document_id",
                            "chunk_index",
                            "clearance",
                            "trust",
                        }
                    },
                )
            )

        return chunks