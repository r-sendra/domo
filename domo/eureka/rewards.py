"""
Handling of LLM-generated reward code: extraction, static validation, and
loading into a callable — the M2 "reward generator" execution path.

Generated code runs with a restricted namespace (torch + math only). This
is a guard against accidents, not a security sandbox: code ultimately runs
in the worker subprocess with the trainer, never in the caller's process.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Optional, Tuple

import torch

from domo.llm.client import extract_code_block

__all__ = ["extract_reward_code", "validate_reward_code", "load_reward_fn"]

REWARD_FN_NAME = "compute_reward"

# torch and math are already provided to the reward namespace; importing them
# is a harmless no-op that LLMs write by habit, so allow it. Anything else is
# rejected (the guard against accidental file/network/system access).
ALLOWED_IMPORTS = {"torch", "math"}


def extract_reward_code(llm_response: str) -> Optional[str]:
    code = extract_code_block(llm_response, "python")
    if code and REWARD_FN_NAME in code:
        return code
    return None


def validate_reward_code(code: str) -> Optional[str]:
    """Static checks. Returns an error string or None if plausible."""
    for forbidden in ("__", "open(", "exec(", "eval(", "subprocess"):
        if forbidden in code:
            return f"forbidden construct in reward code: '{forbidden}'"
    # Imports: only torch / math (already in the namespace) are allowed.
    for m in re.finditer(r"(?m)^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", code):
        root = m.group(1).split(".")[0]
        if root not in ALLOWED_IMPORTS:
            return (f"disallowed import '{root}' — only "
                    f"{', '.join(sorted(ALLOWED_IMPORTS))} are available")
    try:
        compiled = compile(code, "<reward>", "exec")
    except SyntaxError as e:
        return f"syntax error: {e}"
    if REWARD_FN_NAME not in code:
        return f"code does not define {REWARD_FN_NAME}(task)"
    del compiled
    return None


def load_reward_fn(code: str) -> Callable:
    """
    Exec the code and wrap compute_reward with output checking, so a
    malformed candidate fails loudly (and lands in the reflection) instead
    of corrupting training silently.
    """
    err = validate_reward_code(code)
    if err:
        raise ValueError(err)

    namespace = {"torch": torch, "math": math}
    exec(compile(code, "<reward>", "exec"), namespace)
    fn = namespace[REWARD_FN_NAME]

    def wrapped(task) -> Tuple[torch.Tensor, dict]:
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
