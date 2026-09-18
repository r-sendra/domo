"""
LLM client abstraction for the supervisor layers (M1/M2).

One tiny interface — `generate(prompt, temperature) → str` — with two
implementations here:

  * GeminiClient   — Gemini 2.5 Flash free tier (same setup as the voice
                     pipeline: `pip install google-genai`, GEMINI_API_KEY).
                     Retries with backoff on rate limits.
  * ScriptedClient — replays canned responses. For tests, offline
                     development, and reproducing a past run exactly.

plus the LangChain-backed clients in ``langchain_client.py`` (vLLM, OpenAI,
Gemini-via-LangChain), all reachable through ``make_llm(provider, **kw)``.

Everything above (Eureka, DrEureka, future skill-gap detection) codes
against the interface, so swapping providers is a constructor change. The
module also hosts the response-parsing helpers (``extract_code_block``,
``extract_json_block``) because every provider's text goes through them.

Adding a provider: subclass ``LLMClient`` (or wrap a LangChain chat model
with ``LangChainClient``), keep heavy SDK imports inside the constructor so
the core library stays dependency-free, and add a branch to ``make_llm``.
"""

from __future__ import annotations

import logging
import os
import re
import time

__all__ = [
    "GeminiClient",
    "LLMClient",
    "ScriptedClient",
    "extract_code_block",
    "extract_json_block",
    "make_llm",
]

logger = logging.getLogger(__name__)

# First retry wait for GeminiClient; doubles on each further attempt.
_INITIAL_RETRY_DELAY_S = 5.0

# google-genai finish_reason name meaning the output budget was exhausted.
_FINISH_MAX_TOKENS = "MAX_TOKENS"


class LLMClient:
    """Provider-agnostic interface: one prompt in, one text reply out."""

    def generate(self, prompt: str, temperature: float = 1.0) -> str:
        """Return the model's reply to ``prompt`` (single-turn, no history)."""
        raise NotImplementedError


class GeminiClient(LLMClient):
    """Direct google-genai client (no LangChain).

    Args:
        model: Gemini model id. ``gemini-2.5-flash`` is a *thinking* model:
            its reasoning tokens count against ``max_output_tokens``.
        api_key: defaults to ``$GEMINI_API_KEY``.
        max_output_tokens: default 16384 — 8192 proved too small once
            reasoning was included (empty/truncated replies). The model caps
            at 65536; raise this if MAX_TOKENS warnings recur.
        max_retries: attempts before the last exception propagates.
        verbose: print retry / truncation notices.

    Raises:
        ImportError: google-genai not installed.
        ValueError: no API key available.
    """

    def __init__(self, model: str = "gemini-2.5-flash",
                 api_key: str | None = None,
                 max_output_tokens: int = 16384,
                 max_retries: int = 4, verbose: bool = True):
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai not installed — pip install google-genai") from None
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("No Gemini API key. Set GEMINI_API_KEY or pass api_key=.")
        self._genai = genai
        self._client = genai.Client(api_key=key)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.verbose = verbose

    def generate(self, prompt: str, temperature: float = 1.0) -> str:
        from google.genai import types
        delay = _INITIAL_RETRY_DELAY_S
        for attempt in range(self.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=[types.Content(
                        role="user", parts=[types.Part(text=prompt)])],
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=self.max_output_tokens),
                )
                return self._response_text(response)
            except Exception as e:
                # Broad on purpose: free-tier rate limits (429) and transient
                # transport errors surface as different exception classes
                # across google-genai versions; all are worth one backoff.
                if attempt == self.max_retries - 1:
                    raise
                if self.verbose:
                    print(f"  [llm] {type(e).__name__}: retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
        return ""    # unreachable (the last attempt re-raises); keeps the type total

    def _response_text(self, response) -> str:
        """
        Robustly extract the answer text. `gemini-2.5-flash` is a thinking
        model: if reasoning exhausts max_output_tokens the answer is empty or
        truncated (finish_reason MAX_TOKENS). Warn clearly in that case — the
        fix is a larger token budget, not a retry.
        """
        # The SDK's `.text` accessor and the candidate/part layout have both
        # changed shape across releases, so each access is defensive and the
        # fallback chain ends in "" rather than an exception.
        text = None
        try:
            text = response.text
        except Exception as e:
            logger.debug("response.text unavailable (%s); using raw parts", e)
            text = None
        if not text:                          # fall back to raw candidate parts
            try:
                parts = response.candidates[0].content.parts or []
                text = "".join(getattr(p, "text", "") or "" for p in parts)
            except Exception as e:
                logger.debug("no candidate parts in response (%s)", e)
                text = ""
        if self.verbose:
            try:
                reason = getattr(response.candidates[0], "finish_reason", None)
                name = getattr(reason, "name", str(reason))
                if name == _FINISH_MAX_TOKENS:
                    print(f"  [llm] response hit MAX_TOKENS "
                          f"(budget {self.max_output_tokens}) — likely truncated; "
                          f"raise max_output_tokens")
            except Exception as e:
                logger.debug("could not read finish_reason (%s)", e)
        return text or ""


class ScriptedClient(LLMClient):
    """Replays a fixed list of responses (cycling if exhausted).

    ``calls`` records every prompt received, so tests can assert on what the
    routine asked. Subclass and override ``generate`` to answer by prompt
    kind (see ``examples/eureka/eureka_getup.py``).
    """

    def __init__(self, responses: list[str]):
        if not responses:
            raise ValueError("ScriptedClient needs at least one response")
        self.responses = list(responses)
        self.calls: list[str] = []       # prompts received, for inspection

    def generate(self, prompt: str, temperature: float = 1.0) -> str:
        self.calls.append(prompt)
        return self.responses[(len(self.calls) - 1) % len(self.responses)]


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

# Opening fence with an optional language tag (```python, ```py, ```Python,
# ``` alone), optional trailing spaces, then a newline (CRLF tolerated).
_FENCE_OPEN = r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n"


def extract_code_block(text: str, language: str = "python") -> str | None:
    """
    Extract a fenced code block, tolerantly: any/no language tag and
    case, `python` or `py`, CRLF, and — importantly for thinking models —
    a TRUNCATED block whose closing fence never arrived. Falls back to
    unfenced text that already reads like code.

    ``language`` is accepted for API symmetry with ``extract_json_block``
    but not enforced: the tag varies too much across models to be a filter.
    """
    if not text:
        return None
    # Closed fenced block, any/no language tag.
    m = re.search(_FENCE_OPEN + r"(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Unclosed fence (truncated response): take everything after it.
    m = re.search(_FENCE_OPEN + r"(.*)$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # No fences at all, but it already looks like code.
    if "def " in text:
        return text.strip()
    return None


def extract_json_block(text: str) -> str | None:
    """Return the first ```json fenced block, else the outermost ``{...}``."""
    m = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # bare JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0).strip() if m else None


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def make_llm(provider: str = "gemini", **kwargs) -> LLMClient:
    """
    Construct an LLM client by provider name — the single dispatch point for
    Eureka / DrEureka and any future supervisor.

      "gemini"    → direct google-genai (GeminiClient), no LangChain needed
      "vllm"      → local model via vLLM OpenAI-compatible server (LangChain)
      "openai"    → OpenAI or an OpenAI-compatible gateway (LangChain)
      "gemini-lc" → Gemini via langchain-google-genai (LangChain)
      "scripted"  → canned responses; kwargs: responses=[...]

    kwargs are forwarded to the underlying constructor (model, base_url,
    api_key, max_tokens, ...).

    Raises:
        ValueError: unknown provider name.
    """
    provider = provider.lower()
    if provider == "gemini":
        return GeminiClient(**kwargs)
    if provider == "scripted":
        return ScriptedClient(kwargs.get("responses") or ["-"])
    if provider in ("vllm", "openai", "gemini-lc", "gemini_langchain"):
        from .langchain_client import gemini_langchain_client, openai_client, vllm_client
        if provider == "vllm":
            return vllm_client(**kwargs)
        if provider == "openai":
            return openai_client(**kwargs)
        return gemini_langchain_client(**kwargs)
    raise ValueError(f"unknown LLM provider '{provider}' "
                     "(gemini | vllm | openai | gemini-lc | scripted)")
