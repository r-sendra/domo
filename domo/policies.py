"""
Stable-policy registry: the single place skills resolve their *blessed*
checkpoints from.

Training scatters many checkpoints across `runs/`; this module names the ones
we consider stable and copies live in `policies/` at the repo root, so skills
and examples point at a symbolic name ("walk", "avoid") instead of a brittle
`runs/.../checkpoint_final_something.pt` path. To promote a new checkpoint,
copy it over the corresponding file in `policies/` (see policies/README.md).

    from domo.policies import stable_go2_library, load_stable_locomotion
    walk_fn, _ = load_stable_locomotion(device)
    library     = stable_go2_library(lidar, device)     # walk (+avoid if lidar)
"""

from __future__ import annotations

import os

__all__ = ["POLICY_DIR", "STABLE", "stable_policy", "load_stable_locomotion",
           "load_stable_avoid", "stable_go2_library"]

# <repo>/policies  (this file is <repo>/domo/policies.py)
POLICY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policies")

# Symbolic skill name → blessed checkpoint filename under POLICY_DIR.
STABLE = {
    "walk": "walk.pt",      # runs/go2_cpg/checkpoint_final_coupled.pt — most stable gait
    "avoid": "avoid.pt",    # runs/go2_cpg/checkpoint_final_avoid.pt   — lidar avoidance net
}


def stable_policy(name: str) -> str:
    """Absolute path to a blessed checkpoint, by symbolic skill name."""
    if name not in STABLE:
        raise KeyError(f"no stable policy '{name}' (have {sorted(STABLE)})")
    path = os.path.join(POLICY_DIR, STABLE[name])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"stable policy '{name}' missing at {path} — copy the blessed "
            f"checkpoint there (see policies/README.md)")
    return path


def load_stable_locomotion(device: str = "cpu"):
    """(policy_fn, meta) for the stable CPG walk policy."""
    from domo.checkpoints import load_locomotion_policy
    return load_locomotion_policy(stable_policy("walk"), device)


def load_stable_avoid(device: str = "cpu"):
    """Callable obs → Δv for the stable lidar-avoidance net."""
    from domo.checkpoints import load_checkpoint
    from domo.rl import ActorCritic, clean_state_dict
    ckpt = load_checkpoint(stable_policy("avoid"), device)
    net = ActorCritic.from_state_dict(clean_state_dict(ckpt["model_state"]))
    net.eval().to(device)
    return lambda obs: net.get_action(obs, deterministic=True)[0]


def stable_go2_library(lidar=None, device: str = "cpu", avoid_deltas=None):
    """
    Build the Go2 skill library from the stable registry — walk always,
    avoid (+ blocked/clear conditions) when a `lidar` is provided. This is the
    'skills point straight at their blessed policies' convenience.
    """
    from domo.skills import make_go2_library
    walk_fn, _ = load_stable_locomotion(device)
    avoid_fn = load_stable_avoid(device) if lidar is not None else None
    kw = {} if avoid_deltas is None else {"avoid_deltas": avoid_deltas}
    return make_go2_library(walk_fn, avoid_fn, lidar, **kw)
