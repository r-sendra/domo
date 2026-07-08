"""
Eureka worker — subprocess entry point.

Each candidate training (and each DR evaluation batch) runs in its own
process because Genesis initialises once per process. The parent routine
never imports a physics engine; it only writes spec JSON, spawns

    python -m domo.eureka.worker <spec.json>

and reads back <run_dir>/results.json.

Spec schema (mode "train"):
    { "mode": "train", "task": "go2_getup",
      "task_overrides": {...},                 # config kwargs incl. "dr"
      "reward_code_file": ".../candidate.py",
      "ppo": {...},                            # PPOConfig kwargs
      "run_dir": "...", "eval_episodes": 32, "snapshots": 4 }

Spec schema (mode "dr_eval"):
    { "mode": "dr_eval", "task": "go2_getup",
      "task_overrides": {...},
      "checkpoint": ".../checkpoint_final.pt",
      "sweeps": [ {"label": "friction=0.5", "dr": {...}}, ... ],
      "eval_episodes": 32, "run_dir": "..." }
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback

import torch


def _build_task(spec, dr_override=None):
    from .spec import TASK_REGISTRY
    task_spec = TASK_REGISTRY[spec["task"]]
    module = importlib.import_module(task_spec.module)
    config_cls = getattr(module, task_spec.config_class)
    task_cls = getattr(module, task_spec.task_class)
    overrides = dict(spec.get("task_overrides", {}))
    if dr_override is not None:
        overrides["dr"] = dr_override
    return task_cls(config_cls(**overrides))


def _record_stats(records) -> dict:
    """Aggregate a list of per-episode outcome dicts (or legacy bools)."""
    if not records:
        return {"success_rate": 0.0, "fitness": 0.0, "peak_height": 0.0,
                "ever_upright_rate": 0.0, "max_hold": 0.0}
    if isinstance(records[0], dict):
        n = len(records)
        return {
            "success_rate": sum(r["success"] for r in records) / n,
            "fitness": sum(r.get("fitness", 0.0) for r in records) / n,
            "peak_height": sum(r.get("peak_height", 0.0) for r in records) / n,
            "ever_upright_rate": sum(r.get("ever_upright", False)
                                     for r in records) / n,
            "max_hold": sum(r.get("max_hold", 0) for r in records) / n,
        }
    # Legacy: list of success bools (fitness == success).
    rate = sum(bool(r) for r in records) / len(records)
    return {"success_rate": rate, "fitness": rate, "peak_height": 0.0,
            "ever_upright_rate": rate}


def _evaluate_success(task, net, episodes: int) -> dict:
    """Deterministic rollouts until `episodes` episodes finish."""
    start = len(task.episode_outcomes)
    obs, _ = task.reset()
    step_cap = episodes * task.max_episode_length + 200
    steps = 0
    while len(task.episode_outcomes) - start < episodes and steps < step_cap:
        with torch.no_grad():
            act, _, _ = net.get_action(obs, deterministic=True)
        obs, _, _, reset_buf, _ = task.step(act)
        steps += 1
    stats = _record_stats(task.episode_outcomes[start:start + episodes])
    stats["mean_ep_len"] = float(task.max_episode_length)
    return stats


def _run_train(spec) -> dict:
    from dataclasses import asdict

    from domo.rl import PPOConfig, PPOTrainer

    from .rewards import load_reward_fn

    with open(spec["reward_code_file"]) as f:
        code = f.read()
    reward_fn = load_reward_fn(code)

    task = _build_task(spec)
    task.set_reward_override(reward_fn)

    # Sanity: one step with the generated reward before committing to train.
    task.reset()
    task.step(torch.zeros(task.n_envs, task.num_actions, device=task.device))

    ppo = PPOConfig(**spec.get("ppo", {}), run_dir=spec["run_dir"])
    trainer = PPOTrainer(task, ppo, extra_checkpoint_data={
        "task": spec["task"], "task_config": spec.get("task_overrides", {}),
        "reward_code": code})

    # Reward-reflection snapshots at ~equal intervals through training.
    snapshots = []
    total_updates = max(ppo.total_steps // (ppo.rollout_steps * task.n_envs), 1)
    every = max(total_updates // max(spec.get("snapshots", 4), 1), 1)

    def on_update(tr, update):
        if update % every != 0 and update != total_updates:
            return
        comps = {k: float(v.mean()) for k, v in task.reward_components.items()}
        stats = _record_stats(task.episode_outcomes[-200:])
        snapshots.append({
            "frac": update / total_updates,
            "components": comps,
            "success_rate": stats["success_rate"],
            "fitness": stats["fitness"],
            "peak_height": stats["peak_height"],
            "ever_upright_rate": stats["ever_upright_rate"],
            "mean_ep_len": (float(sum(tr.ep_lengths[-50:]) / max(len(tr.ep_lengths[-50:]), 1))
                            if tr.ep_lengths else 0.0),
        })

    trainer.update_callback = on_update
    trainer.train()

    ckpt = os.path.join(spec["run_dir"], "checkpoint_final.pt")
    stats = _evaluate_success(task, trainer.net, spec.get("eval_episodes", 32))
    return {"error": None, "snapshots": snapshots, "checkpoint": ckpt, **stats}


def _run_dr_eval(spec) -> dict:
    from domo.rl import ActorCritic, clean_state_dict

    ckpt = torch.load(spec["checkpoint"], weights_only=False,
                      map_location="cpu")
    net = ActorCritic.from_state_dict(clean_state_dict(ckpt["model_state"]))
    net.eval()

    results = []
    task = None
    for sweep in spec["sweeps"]:
        # DR is applied per reset; one task instance suffices — swap its DR.
        if task is None:
            task = _build_task(spec, dr_override=sweep["dr"])
        else:
            from domo.robot.randomization import DomainRandomization
            task.dr = (DomainRandomization.from_dict(sweep["dr"])
                       if sweep["dr"] else None)
        net_dev = net.to(task.device)
        stats = _evaluate_success(task, net_dev, spec.get("eval_episodes", 32))
        results.append({"label": sweep["label"], "dr": sweep["dr"],
                        "success_rate": stats["success_rate"],
                        "fitness": stats["fitness"],
                        "mean_ep_len": stats["mean_ep_len"]})
        print(f"  [dr_eval] {sweep['label']}: success {stats['success_rate']:.0%} "
              f"fitness {stats['fitness']:.2f}")
    return {"error": None, "sweeps": results}


def main():
    spec_path = sys.argv[1]
    with open(spec_path) as f:
        spec = json.load(f)
    os.makedirs(spec["run_dir"], exist_ok=True)
    try:
        if spec["mode"] == "train":
            results = _run_train(spec)
        elif spec["mode"] == "dr_eval":
            results = _run_dr_eval(spec)
        else:
            raise ValueError(f"unknown mode {spec['mode']}")
    except Exception:
        results = {"error": traceback.format_exc()}
    with open(os.path.join(spec["run_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[worker] done → {spec['run_dir']}/results.json")


if __name__ == "__main__":
    main()
