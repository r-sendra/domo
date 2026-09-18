"""domo.llm: parsing helpers, provider factory, and the client adapters — offline."""

from __future__ import annotations

import types

import pytest

import domo.llm as llm_pkg
from domo.llm.client import (
    GeminiClient,
    LLMClient,
    ScriptedClient,
    extract_code_block,
    extract_json_block,
    make_llm,
)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def test_extract_code_block_variants():
    assert extract_code_block("```\ndef f():\n    pass\n```") == "def f():\n    pass"
    assert extract_code_block("```python\r\ndef f(): ...\r\n```") == "def f(): ..."
    assert extract_code_block("x\n```PYTHON  \nprint(1)\n```\n```json\n{}\n```") == "print(1)"
    # unfenced but code-like text is accepted; prose is not
    assert extract_code_block("def f():\n    return 1") == "def f():\n    return 1"
    assert extract_code_block("Just words.") is None
    assert extract_code_block("") is None


def test_extract_json_block_variants():
    assert extract_json_block("```json\n{\"a\": [1, 2]}\n```") == '{"a": [1, 2]}'
    assert extract_json_block("prefix {\"a\": {\"b\": 1}} suffix") == '{"a": {"b": 1}}'
    assert extract_json_block("nothing structured") is None


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def test_base_client_is_abstract():
    with pytest.raises(NotImplementedError):
        LLMClient().generate("p")


def test_scripted_client_requires_responses():
    with pytest.raises(ValueError):
        ScriptedClient([])


def _bare_gemini(verbose: bool = True) -> GeminiClient:
    """A GeminiClient without running __init__ (no SDK, no key)."""
    client = GeminiClient.__new__(GeminiClient)
    client.verbose = verbose
    client.max_output_tokens = 123
    return client


def test_gemini_response_text_fallbacks(capsys):
    client = _bare_gemini()
    ok = types.SimpleNamespace(text="answer", candidates=[
        types.SimpleNamespace(finish_reason=types.SimpleNamespace(name="STOP"))])
    assert client._response_text(ok) == "answer"

    # `.text` empty → concatenated candidate parts; MAX_TOKENS → warning.
    parts = [types.SimpleNamespace(text="par"), types.SimpleNamespace(text="tial")]
    truncated = types.SimpleNamespace(text=None, candidates=[types.SimpleNamespace(
        content=types.SimpleNamespace(parts=parts),
        finish_reason=types.SimpleNamespace(name="MAX_TOKENS"))])
    assert client._response_text(truncated) == "partial"
    warning = capsys.readouterr().out
    assert "MAX_TOKENS" in warning and "budget 123" in warning

    # Nothing usable anywhere → "" rather than an exception.
    class Broken:
        @property
        def text(self):
            raise RuntimeError("no parts")
        candidates = []
    assert client._response_text(Broken()) == ""


def test_gemini_client_needs_key_or_sdk(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises((ImportError, ValueError)):
        make_llm("gemini")


# ---------------------------------------------------------------------------
# Provider factory + LangChain adapter
# ---------------------------------------------------------------------------

def test_make_llm_is_case_insensitive_and_forwards_kwargs():
    client = make_llm("SCRIPTED", responses=["x", "y"])
    assert isinstance(client, ScriptedClient) and client.responses == ["x", "y"]
    assert make_llm("scripted").generate("p") == "-"        # default canned reply


def test_lazy_langchain_exports():
    pytest.importorskip("langchain_core")
    assert llm_pkg.LangChainClient is llm_pkg.langchain_client.LangChainClient
    assert callable(llm_pkg.vllm_client)
    with pytest.raises(AttributeError):
        getattr(llm_pkg, "no_such_name")  # noqa: B009 — exercising module __getattr__


def test_langchain_client_binds_temperature_and_flattens_content():
    pytest.importorskip("langchain_core")
    from domo.llm.langchain_client import LangChainClient, _content_to_text

    class FakeChat:
        def __init__(self):
            self.bound = None
            self.prompts = []

        def bind(self, **kw):
            self.bound = kw
            return self

        def invoke(self, messages):
            self.prompts.append(messages[0].content)
            return types.SimpleNamespace(content=[{"type": "text", "text": "a"}, "b"])

    chat = FakeChat()
    client = LangChainClient(chat)
    assert client.model is chat
    assert client.generate("hello", temperature=0.3) == "ab"
    assert chat.bound == {"temperature": 0.3} and chat.prompts == ["hello"]

    # supports_temperature=False never binds.
    chat2 = FakeChat()
    LangChainClient(chat2, supports_temperature=False).generate("p")
    assert chat2.bound is None

    assert _content_to_text("plain") == "plain"
    assert _content_to_text(None) == ""
    assert _content_to_text([{"no_text": 1}, "z"]) == "z"


def test_langchain_client_reraises_provider_errors(capsys):
    pytest.importorskip("langchain_core")
    from domo.llm.langchain_client import LangChainClient

    class Failing:
        def bind(self, **kw):
            raise TypeError("no temperature here")   # tolerated: falls back

        def invoke(self, messages):
            raise ConnectionError("server down")

    with pytest.raises(ConnectionError):
        LangChainClient(Failing()).generate("p")
    assert "ConnectionError" in capsys.readouterr().out


def test_make_llm_vllm_builds_offline(monkeypatch):
    pytest.importorskip("langchain_openai")
    from domo.llm.langchain_client import LangChainClient
    monkeypatch.setenv("VLLM_BASE_URL", "http://example.invalid/v1")
    client = make_llm("vllm", model="some/model")
    assert isinstance(client, LangChainClient)
    assert client.model.model_name == "some/model"
    assert "example.invalid" in str(client.model.openai_api_base)
