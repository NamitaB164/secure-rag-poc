from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed document texts."""

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Embed a query."""