# `domo.llm` and `domo.hri` — LLM clients and the voice interface

Two small packages that sit beside `domo.eureka` in the layer stack (see
[architecture.md](../architecture.md)). `domo.llm` is the only place the
library talks to a language model: one interface, several providers, two
parsing helpers. `domo.hri` is the M6 precursor, the human side of the
loop: a microphone thread that turns speech into a velocity command the
control loop reads every step.

Neither package is needed by the core library. Every SDK (`google-genai`,
LangChain, Whisper, `sounddevice`) is imported inside a constructor or a
function body, so `import domo.llm` and `import domo.hri` succeed on a bare
install and the tests in `tests/test_llm_client.py` run without a key or a
network.

How Eureka drives a client (prompts, retries around candidate generation,
`SkillLearningRequest.llm` / `llm_kwargs`) is documented in
[eureka.md](eureka.md) and not repeated here. Provider-specific failures
(empty Gemini replies, unreachable vLLM) are in
[troubleshooting.md, LLM providers](../troubleshooting.md#llm-providers).

## Module map

| File | Exports | Role |
|------|---------|------|
| `domo/llm/client.py` | `LLMClient`, `GeminiClient`, `ScriptedClient`, `extract_code_block`, `extract_json_block`, `make_llm` | the interface, the dependency-light default provider, the offline provider, the parsing helpers, the factory |
| `domo/llm/langchain_client.py` | `LangChainClient`, `vllm_client`, `openai_client`, `gemini_langchain_client` | any LangChain chat model behind the same interface; imported lazily |
| `domo/llm/__init__.py` | everything above | the LangChain names are served through a module `__getattr__`, so `from domo.llm import vllm_client` imports LangChain only at that moment |
| `domo/hri/voice.py` | `CommandState`, `VoiceCommander`, `parse_with_gemini`, `record_audio`, `transcribe` | the voice pipeline and its shared state |

## The LLM interface

```python
class LLMClient:
    def generate(self, prompt: str, temperature: float = 1.0) -> str
```

Single turn, no history, no tools: one prompt string in, the reply text
out. Everything above the package (Eureka, DrEureka, future skill-gap
detection) codes against this and nothing else, so changing providers is
a constructor change. The base class raises `NotImplementedError`.

## Providers

`make_llm` is the single dispatch point:

```python
def make_llm(provider: str = "gemini", **kwargs) -> LLMClient
```

The name is case-insensitive; `kwargs` are forwarded verbatim to the
constructor listed below. An unknown name raises `ValueError`.

| `provider` | Constructor | Backend | Environment | Defaults | Install |
|------------|-------------|---------|-------------|----------|---------|
| `gemini` | `GeminiClient` | `google-genai`, direct | `GEMINI_API_KEY` | `model="gemini-2.5-flash"`, `max_output_tokens=16384`, `max_retries=4`, `verbose=True` | `pip install -e ".[llm]"` |
| `vllm` | `vllm_client` | vLLM's OpenAI-compatible server through `langchain_openai.ChatOpenAI` | `VLLM_BASE_URL` (default `http://localhost:8000/v1`) | `model` required, `api_key="EMPTY"` (ignored by vLLM), `max_tokens=16384`, 600 s request timeout | `pip install -e ".[langchain]"` |
| `openai` | `openai_client` | OpenAI or any OpenAI-compatible gateway (`base_url`) through `ChatOpenAI` | `OPENAI_API_KEY` | `model="gpt-4o-mini"`, `max_tokens=16384`, 600 s timeout | `pip install -e ".[langchain]"` |
| `gemini-lc` (alias `gemini_langchain`) | `gemini_langchain_client` | `langchain_google_genai.ChatGoogleGenerativeAI` | `GEMINI_API_KEY` | `model="gemini-2.5-flash"`, `max_tokens=16384` | `pip install -e ".[langchain,gemini-langchain]"` |
| `scripted` | `ScriptedClient` | none | — | `responses=[...]`; without the kwarg a single `"-"` reply | — |

```python
from domo.llm import make_llm

llm = make_llm("gemini")                                     # free tier, GEMINI_API_KEY
llm = make_llm("vllm", model="Qwen/Qwen2.5-Coder-7B-Instruct")
llm = make_llm("openai", model="gpt-4o", base_url="https://gateway.example/v1")
llm = make_llm("scripted", responses=["first reply", "second reply"])
```

### `GeminiClient`

```python
GeminiClient(model: str = "gemini-2.5-flash",
             api_key: str | None = None,
             max_output_tokens: int = 16384,
             max_retries: int = 4,
             verbose: bool = True)
```

Raises `ImportError` when `google-genai` is missing and `ValueError` when
neither `api_key` nor `GEMINI_API_KEY` is set. `generate` sends the prompt
as a single user turn with the requested `temperature` and retries on any
exception (free-tier 429s and transport errors surface as different
classes across SDK versions) with a 5 s wait that doubles per attempt; the
last attempt's exception propagates.

`gemini-2.5-flash` is a thinking model: reasoning tokens count against
`max_output_tokens`, which is why the default is 16 384 rather than the
8 192 that produced empty replies. The reply text is extracted defensively
(`response.text`, then the concatenated candidate parts, then `""`), and
when the finish reason is `MAX_TOKENS` a warning is printed naming the
budget. The remedy is a larger `max_output_tokens` (the model caps at
65 536), not a retry.

### `LangChainClient`

```python
LangChainClient(chat_model, supports_temperature: bool = True, verbose: bool = True)
LangChainClient.model      # the wrapped chat model (property)
```

Adapts any `langchain_core` chat model (`invoke` + `bind`). `generate`
binds the per-call temperature (falling back to the model's default if the
integration rejects the kwarg, or never binding when
`supports_temperature=False`), invokes with one `HumanMessage`, flattens
the reply's content (a string or a list of text blocks) and re-raises
provider errors after printing their class and message. The three
constructors above return one of these:

```python
def vllm_client(model: str, base_url: str | None = None, api_key: str = "EMPTY",
                max_tokens: int = 16384, verbose: bool = True) -> LangChainClient
def openai_client(model: str = "gpt-4o-mini", api_key: str | None = None,
                  base_url: str | None = None, max_tokens: int = 16384,
                  verbose: bool = True) -> LangChainClient
def gemini_langchain_client(model: str = "gemini-2.5-flash", api_key: str | None = None,
                            max_tokens: int = 16384, verbose: bool = True) -> LangChainClient
```

## Parsing helpers

Every provider's text goes through these, which is why they live next to
the clients.

```python
def extract_code_block(text: str, language: str = "python") -> str | None
def extract_json_block(text: str) -> str | None
```

`extract_code_block` is deliberately tolerant, because thinking models
and small local models are sloppy with fences:

* any language tag or none (` ```python `, ` ```py `, ` ```PYTHON  `,
  bare ` ``` `), CRLF line endings; the first closed block wins;
* a block whose closing fence never arrived (a truncated reply) is
  salvaged: everything after the opening fence is returned;
* text with no fences at all is accepted if it contains `def `;
* prose and the empty string give `None`.

`language` is accepted for symmetry with `extract_json_block` but not
enforced; the tag varies too much across models to be a filter.

```python
>>> extract_code_block("```python\r\ndef f(): ...\r\n```")
'def f(): ...'
>>> extract_code_block("Here you go:\n```py\ndef compute_reward(task):\n    return r, {}")
'def compute_reward(task):\n    return r, {}'      # unclosed fence, salvaged
>>> extract_code_block("x\n```PYTHON  \nprint(1)\n```\n```json\n{}\n```")
'print(1)'                                          # first block only
>>> extract_code_block("Just words.") is None
True
```

`extract_json_block` returns the first ` ```json ` fenced block, else the
outermost `{...}` in the text (greedy, so nested objects survive), else
`None`. It returns the string; the caller runs `json.loads`.

```python
>>> extract_json_block('```json\n{"a": [1, 2]}\n```')
'{"a": [1, 2]}'
>>> extract_json_block('Sure: {"a": {"b": 1}} — done')
'{"a": {"b": 1}}'
>>> extract_json_block("nothing structured") is None
True
```

## `ScriptedClient` for offline tests

```python
ScriptedClient(responses: list[str])
ScriptedClient.responses    # the list, as given
ScriptedClient.calls        # every prompt received, in order
```

Replays `responses` in order and cycles when exhausted; an empty list
raises `ValueError`. `calls` records each prompt so a test can assert on
what the routine asked. This is how `tests/test_eureka*.py` and
`examples/eureka/eureka_getup.py --llm scripted` run the full pipeline
with no key and no network.

When one canned list is not enough, subclass and answer by prompt kind.
The Eureka example does exactly this to serve a reward for reward prompts
and a DR proposal for DR prompts:

```python
from domo.llm.client import ScriptedClient

class OfflineClient(ScriptedClient):
    def generate(self, prompt, temperature=1.0):
        self.calls.append(prompt)
        if "domain randomization" in prompt:
            return SCRIPTED_DR
        return SCRIPTED_REWARD
```

## Adding a provider

Two routes, both keeping the core library dependency-free:

1. **A LangChain integration.** Add a `*_client(...)` constructor to
   `domo/llm/langchain_client.py` that imports the integration inside the
   function, builds the chat model and returns `LangChainClient(chat)`.
   Pass `supports_temperature=False` if the endpoint rejects the
   parameter.
2. **A native SDK.** Subclass `LLMClient` in `domo/llm/client.py` (or a
   new module), import the SDK inside `__init__` and raise `ImportError`
   with the `pip install` hint, read the key from an environment variable
   with an `api_key=` override, and implement `generate`.

Then add a branch to `make_llm` and, for a lazily served name, extend
`_LAZY_LANGCHAIN_NAMES` and `__all__` in `domo/llm/__init__.py`. Keep
kwargs forwarded verbatim so `SkillLearningRequest.llm_kwargs` reaches the
constructor unchanged. Add an offline test in `tests/test_llm_client.py`
in the style of `test_make_llm_vllm_builds_offline` (build the client, do
not call it).

## The voice module (`domo.hri`)

A port of `scripts/house_scene/voice_commander.py`. One daemon thread
listens to the microphone; the only object it shares with the control loop
is a `CommandState`.

```
microphone ──record_audio──▶ 3 s mono float32 @ 16 kHz
                                  │
                                  ▼  RMS < silence_threshold → skip
                              transcribe (Whisper, local, fp16=False)
                                  │
                                  ▼  fewer than 3 characters → skip
                       parse_with_gemini(text, current)
                       gemini-2.5-flash, T=0.1, 200 tokens, SYSTEM_PROMPT
                                  │
                                  ▼  description == "no change" → keep current
                       CommandState.set(vx, vy, vyaw)   clipped, marks changed
                                  │
          control loop, every step: vx, vy, vyaw = state.get()
```

Transient errors (a mic hiccup, a network fault, non-JSON from the model)
are printed and the loop continues after a 0.5 s pause; the thread only
ends on `stop()` or with the main program.

### `CommandState`

```python
class CommandState:
    VX_MIN, VX_MAX = -1.0, 1.0
    VY_MIN, VY_MAX = -0.5, 0.5
    VYW_MIN, VYW_MAX = -1.0, 1.0

    def __init__(self, vx: float = 0.5, vy: float = 0.0, vyaw: float = 0.0)
    def set(self, vx: float, vy: float, vyaw: float)   # clipped to the bounds above
    def get(self) -> tuple                             # (vx, vy, vyaw)
    def pop_changed(self) -> bool                      # True once per change
```

`set` clips to the class bounds before storing, so a mis-parsed command can
never exceed what the locomotion policy was trained for. `pop_changed` is
a one-shot flag for control loops that want to react only on a new
command (log it, re-plan) rather than on every step; `get` is what a loop
that simply forwards the command calls. All three take the same lock.

### `VoiceCommander`

```python
VoiceCommander(state: CommandState,
               whisper_model: str = "tiny",
               chunk_duration: float = 3.0,
               silence_threshold: float = 0.01,
               api_key: str | None = None,
               verbose: bool = True)
VoiceCommander.start()      # daemon thread; Whisper loads on that thread
VoiceCommander.stop()       # sets the stop event, joins for up to 5 s
```

`whisper_model` is the openai-whisper size; `tiny` keeps latency usable on
CPU, the example accepts `tiny | base | small`. `chunk_duration` is the
seconds of audio per recognition attempt, so it is also the command
latency floor. `api_key` defaults to `GEMINI_API_KEY`.

The stages are plain functions and can be used without the thread, for a
text-only or pre-recorded pipeline:

```python
def record_audio(duration: float = 3.0, sample_rate: int = 16000) -> np.ndarray
def transcribe(audio: np.ndarray, model) -> str
def parse_with_gemini(text: str, current: tuple, api_key: str | None = None) -> tuple
```

`parse_with_gemini` sends `SYSTEM_PROMPT` plus the current command and the
transcript, and returns `(vx, vy, vyaw, description)`. The current command
is included so relative instructions ("faster", "slower") resolve; speech
that is unclear or unrelated to locomotion yields `description ==
"no change"`. It builds its own `google-genai` client rather than going
through `domo.llm` because the reply is a five-field JSON object at
temperature 0.1 with a 200-token budget, not a reward function. It raises
`ImportError`, `ValueError` (no key) or `json.JSONDecodeError`.

### Dependencies

```
pip install -e ".[voice,llm]"       # openai-whisper, sounddevice, google-genai
export GEMINI_API_KEY=<key from aistudio.google.com/app/apikey>
```

All three are imported lazily; a missing one raises `ImportError` with the
`pip install` hint at the first call that needs it (Whisper when the
thread starts, `sounddevice` at the first recording, `google-genai` at the
first parse). A microphone and internet access are required.

### Example

`examples/hri/go2_cpg_rl_voice.py` drives the frozen walk policy from
voice:

```
python examples/hri/go2_cpg_rl_voice.py --eval policies/walk.pt --whisper-model tiny
python examples/hri/go2_cpg_rl_voice.py --eval policies/walk.pt --headless
```

It builds a single-env `Go2CPGWalkTask` as a stepping harness, starts a
`VoiceCommander` on a `CommandState(vx=--vx)`, and every step copies
`state.get()` into `env.commands` before calling the policy. It runs until
Ctrl+C, prints the command and measured speed every 250 steps, resets on a
fall and saves nothing. It cannot be smoke-tested: it needs a microphone
and an API key.
