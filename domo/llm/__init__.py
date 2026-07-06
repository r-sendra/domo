"""
LLM layer for the developmental supervisor (M1/M2).

Clients speak one interface — `generate(prompt, temperature) -> str`:
  * GeminiClient    — direct google-genai (dependency-light default)
  * ScriptedClient  — canned responses (tests / offline)
  * LangChain-backed clients (vLLM, OpenAI, Gemini) — lazy, via `make_llm`

    from domo.llm import make_llm
    llm = make_llm("vllm", model="Qwen/Qwen2.5-Coder-7B-Instruct")
    llm = make_llm("gemini")                       # free-tier default

LangChain constructors are imported lazily so the core library needs no
LLM dependencies.
"""

from .client import (GeminiClient, LLMClient, ScriptedClient,
                     extract_code_block, extract_json_block, make_llm)

__all__ = [
    "LLMClient", "GeminiClient", "ScriptedClient", "make_llm",
    "extract_code_block", "extract_json_block",
    "vllm_client", "openai_client", "gemini_langchain_client",
    "LangChainClient",
]


def __getattr__(name):
    # Lazy access to the LangChain-backed helpers without importing langchain
    # at package import time.
    if name in ("vllm_client", "openai_client", "gemini_langchain_client",
                "LangChainClient"):
        from . import langchain_client
        return getattr(langchain_client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
