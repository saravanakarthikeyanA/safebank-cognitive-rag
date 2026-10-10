import os
from src.interfaces.embeddings import BaseEmbeddingModel
from src.config_loader import clean_api_key

class LangChainEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, provider: str, model_name: str):
        self.provider = provider.lower()
        self.model_name = model_name
        self.model = self._init_model()

    def _init_model(self):
        if self.provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=self.model_name)
        elif self.provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            api_key = clean_api_key(os.environ.get("GEMINI_API_KEY"))
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            clean_model_name = self.model_name
            if clean_model_name.startswith("models/"):
                clean_model_name = clean_model_name.replace("models/", "")
            return GoogleGenerativeAIEmbeddings(model=clean_model_name, google_api_key=api_key)
        elif self.provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            api_key = clean_api_key(os.environ.get("OPENAI_API_KEY"))
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return OpenAIEmbeddings(model=self.model_name, openai_api_key=api_key)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)

def get_embedding_model(config: dict) -> BaseEmbeddingModel:
    """Factory function to get embedding model from config settings."""
    provider = config.get("embedding", {}).get("provider", "huggingface").lower()
    model_name = config.get("embedding", {}).get("model", "all-MiniLM-L6-v2")
    return LangChainEmbeddingModel(provider, model_name)
