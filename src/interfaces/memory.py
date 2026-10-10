from abc import ABC, abstractmethod

class BaseMemory(ABC):
    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to the chat history.

        Args:
            session_id: Unique chat session identifier.
            role: Speaker role ('user', 'assistant').
            content: The text content of the message.
        """
        pass

    @abstractmethod
    def get_history(self, session_id: str, limit: int = 10, max_tokens: int = None) -> list[dict]:
        """
        Retrieve recent chat history for the session with optional token budgeting.

        Args:
            session_id: Unique chat session identifier.
            limit: Number of recent messages to return.
            max_tokens: Maximum token ceiling for returned history (optional).

        Returns:
            list[dict]: List of message dicts: [{'role': str, 'content': str}]
        """
        pass

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """
        Clear history for a given session.

        Args:
            session_id: Unique chat session identifier.
        """
        pass
