import os
import json
import time
import requests
from src.config_loader import clean_api_key
from src.utils.token_budget import get_rate_limiter

class JevOpenRouterClient:
    """
    Client for interacting with TypeSafe's JEV System 1 Decision Model via OpenRouter.
    Provides fast, deterministic, schema-constrained evaluations with retry resilience and token budgeting.
    """
    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "typesafe/jev-router",
        timeout_seconds: float = 5.0,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5,
        max_tokens: int = 400
    ):
        self.api_key = clean_api_key(api_key or os.environ.get("OPENROUTER_API_KEY", ""))
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.max_tokens = max_tokens
        self.session = requests.Session()
        self.rate_limiter = get_rate_limiter()

    def is_configured(self) -> bool:
        """Check if client has a valid API key configured."""
        return bool(self.api_key)

    @staticmethod
    def _clean_json_string(text: str) -> str:
        """Strip markdown fences and extract clean JSON string."""
        if not text:
            return ""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx+1]
        return text

    def evaluate_decision(self, system_instruction: str, user_payload: dict, schema: dict) -> dict:
        """
        Execute a fast, schema-constrained decision evaluation with retry resilience and minified tokens.

        Args:
            system_instruction: Decision parameters, guidelines, and context.
            user_payload: Input state (text, transaction details, or flags).
            schema: Expected JSON schema of the output.

        Returns:
            dict: Parsed and validated decision output.
        """
        if not self.is_configured():
            raise ValueError("OPENROUTER_API_KEY is missing or empty.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost:8501"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "SafeBank AI Fraud Intelligence"),
            "Content-Type": "application/json"
        }

        # Compact JSON formatting (removes redundant indentation tokens)
        payload_compact = json.dumps(user_payload, separators=(',', ':'))
        schema_compact = json.dumps(schema, separators=(',', ':'))

        prompt_content = (
            f"{system_instruction}\n\n"
            f"Input: {payload_compact}\n\n"
            f"Required Output JSON Schema: {schema_compact}\n\n"
            "Respond ONLY with a valid JSON object matching the schema. Do not include extra commentary."
        )

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are Jev, a high-speed System 1 decision engine. You must output strictly valid JSON matching the schema."},
                {"role": "user", "content": prompt_content}
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens
        }

        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.rate_limiter.acquire()
                start_t = time.time()
                response = self.session.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=self.timeout_seconds
                )
                duration_ms = int((time.time() - start_t) * 1000)

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError(f"OpenRouter response contains no choices: {data}")

                    content = choices[0].get("message", {}).get("content", "")
                    if not content or not content.strip():
                        raise ValueError(f"OpenRouter returned empty content: {data}")

                    cleaned_content = self._clean_json_string(content)
                    if not cleaned_content:
                        raise ValueError(f"No JSON object found in response: {content}")

                    parsed = json.loads(cleaned_content)
                    parsed["_latency_ms"] = duration_ms
                    parsed["_model"] = self.model
                    return parsed
                elif response.status_code in [429, 500, 502, 503, 504]:
                    # Transient error, apply exponential backoff
                    backoff = self.retry_backoff_factor * (2 ** (attempt - 1))
                    time.sleep(backoff)
                    last_exception = RuntimeError(f"OpenRouter HTTP {response.status_code}: {response.text}")
                else:
                    raise RuntimeError(f"OpenRouter API error (HTTP {response.status_code}): {response.text}")

            except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as e:
                last_exception = e
                backoff = self.retry_backoff_factor * (2 ** (attempt - 1))
                time.sleep(backoff)

        raise RuntimeError(f"JEV decision evaluation failed after {self.max_retries} attempts: {last_exception}")
