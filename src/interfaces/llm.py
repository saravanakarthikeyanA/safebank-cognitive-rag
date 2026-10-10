from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, system_instruction: str = None) -> str:
        """
        Generate a text response from the LLM.

        Args:
            prompt: User prompt.
            system_instruction: Optional system instruction/system prompt.

        Returns:
            str: Generated text.
        """
        pass

    @abstractmethod
    def generate_structured_json(self, prompt: str, schema: dict, system_instruction: str = None) -> dict:
        """
        Generate a structured JSON output matching the target schema with system/human role isolation.

        Args:
            prompt: User prompt.
            schema: Dict representing the target JSON schema.
            system_instruction: Optional system instruction for strict role separation.

        Returns:
            dict: Parsed JSON response.
        """
    def stream_response(self, prompt: str, system_instruction: str = None):
        """
        Stream a text response from the LLM chunk by chunk.

        Args:
            prompt: User prompt.
            system_instruction: Optional system instruction.

        Yields:
            str: Chunks of generated text.
        """
        response = self.generate_response(prompt, system_instruction)
        yield response
