from abc import ABC, abstractmethod

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: list[dict]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of dictionaries containing document data:
                       [{'text': 'document text', 'metadata': {'key': 'value'}}]
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4, filter_metadata: dict = None) -> list[dict]:
        """
        Perform a similarity search in the vector store.

        Args:
            query: The search query text.
            k: The number of nearest neighbors to return.
            filter_metadata: Optional dictionary of metadata keys/values to filter the search.

        Returns:
            list[dict]: List of retrieved documents with keys:
                        [{'text': str, 'metadata': dict, 'score': float}]
        """
        pass
