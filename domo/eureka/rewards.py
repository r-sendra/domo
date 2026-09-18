"""
Handling of LLM-generated reward code: extraction, static validation, and
loading into a callable — the M2 "reward generator" execution path.

Pipeline for one candidate (called from ``routine._sample_candidates`` in the
parent and ``worker._run_train`` in the subprocess):

    LLM text ─▶ extract_reward_code ─▶ validate_reward_code ─▶ load_reward_fn
                (fenced block that       (static guard:          (exec + output
                 defines compute_reward)  imports, syntax, ...)   checking wrapper)

Sandboxing assumptions — read before touching ``load_reward_fn``
----------------------------------------------------------------
Generated code is executed with ``exec`` in a namespace that provides only
``torch`` and ``math``. This is a guard against *accidents* (an LLM that
writes ``open()`` or ``import os`` out of habit), NOT a security sandbox:
Python builtins remain reachable and a determined adversary could escape.
The design relies on three facts instead of on isolation:

1. The code only ever runs inside the worker subprocess, alongside the
   trainer, never in the caller's process (see ``routine.run_worker``).
2. The static validator rejects the constructs an accidental escape would
   need (dunder access, file/exec/eval/subprocess, non-whitelisted imports).
3. The wrapper returned by ``load_reward_fn`` type/shape/finiteness-checks
   every output, so a malformed reward fails loudly and lands in the
   reflection instead of silently corrupting training.

If untrusted LLM providers are ever used, run the worker in a container.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import torch

from domo.llm.client import extract_code_block

__all__ = ["extract_reward_code", "load_reward_fn", "validate_reward_code"]

REWARD_FN_NAME = "compute_reward"

# torch and math are already provided to the reward namespace; importing them
# is a harmless no-op that LLMs write by habit, so allow it. Anything else is
# rejected (the guard against accidental file/network/system access).
ALLOWED_IMPORTS = {"torch", "math"}

# Substrings that no legitimate reward function needs and that an accidental
# escape from the namespace would require.
_FORBIDDEN_SUBSTRINGS = ("__", "open(", "exec(", "eval(", "subprocess")

# Matches `import x` / `from x.y import z` at line start; group 1 is the module.
_IMPORT_RE = re.compile(r"(?m)^\s*(?:import|from)\s+([A-Za-z_][\w.]*)")


def extract_reward_code(llm_response: str) -> str | None:
    """Return the code block defining ``compute_reward``, or None.

    Delegates to ``extract_code_block`` (tolerant of missing/odd language tags
    and truncated closing fences) and only accepts the block if it mentions
    the reward entry point, so prose-only answers are rejected early.
    """
    code = extract_code_block(llm_response, "python")
    if code and REWARD_FN_NAME in code:
        return code
    return None


def validate_reward_code(code: str) -> str | None:
    """Static checks. Returns an error string or None if plausible.

    Checks, in order (the first failure is reported): forbidden substrings,
    non-whitelisted imports, syntax, presence of ``compute_reward``.
    """
    for forbidden in _FORBIDDEN_SUBSTRINGS:
        if forbidden in code:
            return f"forbidden construct in reward code: '{forbidden}'"
    # Imports: only torch / math (already in the namespace) are allowed.
    for m in _IMPORT_RE.finditer(code):
        root = m.group(1).split(".")[0]
        if root not in ALLOWED_IMPORTS:
            return (f"disallowed import '{root}' — only "
                    f"{', '.join(sorted(ALLOWED_IMPORTS))} are available")
    try:
        compile(code, "<reward>", "exec")
    except SyntaxError as e:
        return f"syntax error: {e}"
    if REWARD_FN_NAME not in code:
        return f"code does not define {REWARD_FN_NAME}(task)"
    return None


def load_reward_fn(code: str) -> Callable:
    """Exec validated reward code and return a checked ``fn(task)`` wrapper.

    The wrapper enforces the reward contract — ``(reward [N] float, dict of
    per-env component tensors)`` — so a malformed candidate fails loudly (and
    lands in the reflection) instead of corrupting training silently.

    Raises:
        ValueError: if ``validate_reward_code`` rejects the code.
    """
    err = validate_reward_code(code)
    if err:
        raise ValueError(err)

    # Deliberate `exec` (see module docstring): a restricted namespace, run
    # only inside the worker subprocess.
    namespace = {"torch": torch, "math": math}
    exec(compile(code, "<reward>", "exec"), namespace)
    fn = namespace[REWARD_FN_NAME]

    def wrapped(task) -> tuple[torch.Tensor, dict]:
        out = fn(task)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise TypeError(f"{REWARD_FN_NAME} must return (reward, components)")
        reward, components = out
        if reward.shape != (task.n_envs,):
            raise ValueError(f"reward shape {tuple(reward.shape)} != ({task.n_envs},)")
        if not torch.isfinite(reward).all():
            raise ValueError("reward contains non-finite values")
        if not isinstance(components, dict):
            raise TypeError("components must be a dict[str, Tensor]")
        return reward.float(), {k: v.detach().float()
                                for k, v in components.items()}

    return wrapped
