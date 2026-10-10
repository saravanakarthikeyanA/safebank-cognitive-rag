import os
import json
from src.interfaces.llm import BaseLLMClient
from src.config_loader import clean_api_key
from src.utils.token_budget import get_rate_limiter
from langchain_core.messages import SystemMessage, HumanMessage

class LangChainLLMClient(BaseLLMClient):
    def __init__(self, provider: str, model_name: str, temperature: float = 0.1, max_tokens: int = 300, config: dict = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.config = config or {}
        self.rate_limiter = get_rate_limiter(self.config)
        self.model = self._init_model()

    def _init_model(self):
        if self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = clean_api_key(os.environ.get("GEMINI_API_KEY"))
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            clean_model = self.model_name
            if clean_model.startswith("models/"):
                clean_model = clean_model.replace("models/", "")
            return ChatGoogleGenerativeAI(
                model=clean_model,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                google_api_key=api_key
            )

        elif self.provider == "groq":
            from langchain_groq import ChatGroq
            api_key = clean_api_key(os.environ.get("GROQ_API_KEY"))
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            return ChatGroq(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                groq_api_key=api_key
            )

        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = clean_api_key(os.environ.get("OPENAI_API_KEY"))
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=api_key
            )

        elif self.provider == "nvidia":
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            api_key = clean_api_key(os.environ.get("NVIDIA_API_KEY"))
            if not api_key:
                raise ValueError("NVIDIA_API_KEY environment variable not set")
            return ChatNVIDIA(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                nvidia_api_key=api_key
            )

        elif self.provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                num_predict=self.max_tokens
            )

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def generate_response(self, prompt: str, system_instruction: str = None) -> str:
        # Enforce rate limiting before issuing request
        self.rate_limiter.acquire()

        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=prompt))

        response = self.model.invoke(messages)
        return response.content

    def stream_response(self, prompt: str, system_instruction: str = None):
        # Enforce rate limiting before issuing request
        self.rate_limiter.acquire()

        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=prompt))

        for chunk in self.model.stream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
            elif isinstance(chunk, str) and chunk:
                yield chunk

    def generate_structured_json(self, prompt: str, schema: dict, system_instruction: str = None) -> dict:
        # Enforce rate limiting before issuing request
        self.rate_limiter.acquire()

        structured_model = self.model.with_structured_output(schema)
        if system_instruction:
            messages = [
                SystemMessage(content=system_instruction),
                HumanMessage(content=prompt)
            ]
            result = structured_model.invoke(messages)
        else:
            result = structured_model.invoke(prompt)

        if isinstance(result, dict):
            return result
        elif hasattr(result, "dict"):
            return result.dict()
        else:
            return json.loads(str(result))

def get_llm_client(config: dict) -> BaseLLMClient:
    """Factory function to get LLM client from config settings."""
    provider = config.get("llm", {}).get("provider", "gemini").lower()
    model_name = config.get("llm", {}).get("model", "gemini-1.5-flash")
    temp = config.get("llm", {}).get("temperature", 0.1)

    # Read max_tokens from llm or token_budget section
    tb_cfg = config.get("token_budget", {})
    max_tokens = config.get("llm", {}).get("max_tokens", tb_cfg.get("max_output_tokens", 300))

    return LangChainLLMClient(provider, model_name, temp, max_tokens=max_tokens, config=config)
