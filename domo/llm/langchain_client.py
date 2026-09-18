"""
LangChain-backed LLM clients — the provider-agnostic path.

Wraps any LangChain chat model behind the DOMO `LLMClient` interface
(`generate(prompt, temperature) -> str`), so Eureka / DrEureka can drive:
  * local models served by vLLM (OpenAI-compatible endpoint)
  * OpenAI / Azure / any OpenAI-compatible gateway
  * Google Gemini via langchain-google-genai
without changing a line of the routine.

LangChain is imported lazily: the direct `GeminiClient` / `ScriptedClient`
in `client.py` remain dependency-free fallbacks. Install extras with
`pip install langchain-core langchain-openai` (+ `langchain-google-genai`
for the Gemini binding).

vLLM quickstart (serve a local model, OpenAI-compatible):
    vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --port 8000
Then:
    from domo.llm import vllm_client
    llm = vllm_client("Qwen/Qwen2.5-Coder-7B-Instruct")
    # or via the factory:  make_llm("vllm", model="...", base_url="...")

Adding a provider: write a ``*_client(...)`` constructor below that builds
the LangChain chat model and returns ``LangChainClient(chat)``, then route a
provider name to it in ``client.make_llm``.
"""

from __future__ import annotations

import logging
import os

from .client import LLMClient

__all__ = [
    "LangChainClient",
    "gemini_langchain_client",
    "openai_client",
    "vllm_client",
]

logger = logging.getLogger(__name__)

# Seconds before a single chat completion is abandoned. Reward generation
# with a thinking model or a small local GPU can legitimately take minutes.
_REQUEST_TIMEOUT_S = 600

# Output budget; matches GeminiClient so thinking models are not truncated.
_DEFAULT_MAX_TOKENS = 16384


class LangChainClient(LLMClient):
    """Adapt a LangChain BaseChatModel to the DOMO LLMClient interface.

    Args:
        chat_model: any ``langchain_core`` chat model (``invoke`` + ``bind``).
        supports_temperature: bind the per-call temperature; set False for
            endpoints that reject the parameter.
        verbose: print the exception class/message before re-raising.
    """

    def __init__(self, chat_model, supports_temperature: bool = True,
                 verbose: bool = True):
        self._model = chat_model
        self._supports_temperature = supports_temperature
        self.verbose = verbose

    @property
    def model(self):
        """The wrapped LangChain chat model."""
        return self._model

    def generate(self, prompt: str, temperature: float = 1.0) -> str:
        from langchain_core.messages import HumanMessage
        model = self._model
        if self._supports_temperature:
            try:
                model = self._model.bind(temperature=temperature)
            except Exception as e:
                # Some integrations reject unknown bind kwargs; the call is
                # still valid at the model's default temperature.
                logger.debug("bind(temperature=%s) failed (%s); using default",
                             temperature, e)
                model = self._model
        try:
            resp = model.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            if self.verbose:
                print(f"  [llm] {type(e).__name__}: {e}")
            raise
        return _content_to_text(resp.content)


def _content_to_text(content) -> str:
    """LangChain message content may be str or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


# ---------------------------------------------------------------------------
# Provider constructors
# ---------------------------------------------------------------------------

def vllm_client(model: str,
                base_url: str | None = None,
                api_key: str = "EMPTY",
                max_tokens: int = _DEFAULT_MAX_TOKENS,
                verbose: bool = True) -> LangChainClient:
    """
    Client for a local model served by vLLM's OpenAI-compatible server.
    `base_url` defaults to $VLLM_BASE_URL or http://localhost:8000/v1.
    vLLM ignores the api_key, so the sentinel "EMPTY" is fine.
    """
    from langchain_openai import ChatOpenAI
    base_url = base_url or os.environ.get("VLLM_BASE_URL",
                                          "http://localhost:8000/v1")
    chat = ChatOpenAI(model=model, base_url=base_url, api_key=api_key,
                      max_tokens=max_tokens, timeout=_REQUEST_TIMEOUT_S)
    return LangChainClient(chat, verbose=verbose)


def openai_client(model: str = "gpt-4o-mini",
                  api_key: str | None = None,
                  base_url: str | None = None,
                  max_tokens: int = _DEFAULT_MAX_TOKENS,
                  verbose: bool = True) -> LangChainClient:
    """OpenAI (or any OpenAI-compatible gateway via base_url); key from $OPENAI_API_KEY."""
    from langchain_openai import ChatOpenAI
    chat = ChatOpenAI(model=model,
                      api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                      base_url=base_url, max_tokens=max_tokens,
                      timeout=_REQUEST_TIMEOUT_S)
    return LangChainClient(chat, verbose=verbose)


def gemini_langchain_client(model: str = "gemini-2.5-flash",
                            api_key: str | None = None,
                            max_tokens: int = _DEFAULT_MAX_TOKENS,
                            verbose: bool = True) -> LangChainClient:
    """Gemini through langchain-google-genai (vs. the direct GeminiClient)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    chat = ChatGoogleGenerativeAI(
        model=model, google_api_key=api_key or os.environ.get("GEMINI_API_KEY"),
        max_output_tokens=max_tokens)
    return LangChainClient(chat, verbose=verbose)
