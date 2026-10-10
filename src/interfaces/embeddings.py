from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings

class BaseEmbeddingModel(Embeddings, ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding vector for a single query string.

        Args:
            text: The text to embed.

        Returns:
            list[float]: Embedding vector.
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for a list of document strings.

        Args:
            texts: List of document texts to embed.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        pass
