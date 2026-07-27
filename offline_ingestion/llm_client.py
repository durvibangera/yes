"""
LLM client abstraction — swappable between Ollama and Internal Qwen.
All generation code calls get_llm_client().generate() — never the ollama package directly.
"""
from abc import ABC, abstractmethod
import logging
import ollama as _ollama

logger = logging.getLogger("LLMClient")


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...


class OllamaClient(LLMClient):
    def __init__(self, model: str, host: str, temperature: float):
        self.model = model
        self.host = host
        self.temperature = temperature
        self._client = _ollama.Client(host=host)

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": temperature},
        )
        return response.response.strip()


class InternalQwenClient(LLMClient):
    """
    Stub for the company-internal Qwen deployment.
    Fill in once the endpoint URL, auth mechanism, and request/response format are confirmed.
    Do NOT guess at the API surface.
    """
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(
            "InternalQwenClient is not implemented yet. "
            "Set LLM_BACKEND=ollama in .env to use the local Ollama server. "
            "Once the internal Qwen endpoint is confirmed, implement this class."
        )


def get_llm_client() -> LLMClient:
    from ingestion.config import settings
    backend = settings.LLM_BACKEND
    if backend == "ollama":
        return OllamaClient(
            model=settings.OLLAMA_MODEL,
            host=settings.OLLAMA_HOST,
            temperature=settings.GENERATION_TEMPERATURE,
        )
    elif backend == "internal_qwen":
        return InternalQwenClient()
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend!r}. Choose 'ollama' or 'internal_qwen'.")
