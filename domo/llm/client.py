"""
LLM client abstraction for the supervisor layers (M1/M2).

One tiny interface — `generate(prompt, temperature) → str` — with two
implementations:

  * GeminiClient   — Gemini 2.5 Flash free tier (same setup as the voice
                     pipeline: `pip install google-genai`, GEMINI_API_KEY).
                     Retries with backoff on rate limits.
  * ScriptedClient — replays canned responses. For tests, offline
                     development, and reproducing a past run exactly.

Everything above (Eureka, DrEureka, future skill-gap detection) codes
against the interface, so swapping providers is a constructor change.
"""

from __future__ import annotations

import os
import re
import time
from typing import List, Optional

__all__ = ["LLMClient", "GeminiClient", "ScriptedClient", "extract_code_block",
           "extract_json_block"]


class LLMClient:
    def generate(self, prompt: str, temperature: float = 1.0) -> str:
        raise NotImplementedError


class GeminiClient(LLMClient):
    def __init__(self, model: str = "gemini-2.5-flash",
                 api_key: Optional[str] = None,
                 max_output_tokens: int = 8192,
                 max_retries: int = 4, verbose: bool = True):
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai not installed — pip install google-genai")
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
        delay = 5.0
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
                return response.text or ""
            except Exception as e:            # rate limits on the free tier
                if attempt == self.max_retries - 1:
                    raise
                if self.verbose:
                    print(f"  [llm] {type(e).__name__}: retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
        return ""


class ScriptedClient(LLMClient):
    """Replays a fixed list of responses (cycling if exhausted)."""

    def __init__(self, responses: List[str]):
        if not responses:
            raise ValueError("ScriptedClient needs at least one response")
        self.responses = list(responses)
        self.calls: List[str] = []       # prompts received, for inspection

    def generate(self, prompt: str, temperature: float = 1.0) -> str:
        self.calls.append(prompt)
        return self.responses[(len(self.calls) - 1) % len(self.responses)]


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------

def extract_code_block(text: str, language: str = "python") -> Optional[str]:
    """First fenced code block; tolerates a missing language tag."""
    m = re.search(rf"```{language}\s*\n(.*?)```", text, re.DOTALL)
    if m is None:
        m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_json_block(text: str) -> Optional[str]:
    m = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # bare JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0).strip() if m else None
