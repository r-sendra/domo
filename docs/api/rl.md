# `domo.rl` — PPO over the `VecTask` API

The learning layer: a shared-trunk actor-critic, a rollout buffer with
GAE-λ, and a PPO trainer that consumes any environment implementing the
`domo.tasks.VecTask` step API (`reset()`, `step(actions)`, plus `num_envs`,
`num_obs`, `num_actions`, `device`). The package is pure torch and imports
no physics engine, so networks and checkpoints load on machines without a
simulator (deployment, evaluation, tests). It is one optional consumer of
a task; nothing in `domo.tasks` depends on it.

`main.py` at the repository root is the reference entry point: it trains
and evaluates `Go2WalkTask` with this trainer. The other training scripts
under `examples/` follow the same pattern with their own tasks
([running.md](../running.md)).

## Module map

| Module | Contents |
|--------|----------|
| `domo/rl/networks.py` | `ActorCritic`, `clean_state_dict` |
| `domo/rl/buffer.py` | `RolloutBuffer` |
| `domo/rl/ppo.py` | `PPOConfig`, `PPOTrainer` |
| `main.py` | `build_configs`, `load_checkpoint`, `build_policy`, `evaluate`, `build_parser`, `main` |

`domo.rl.__all__` = `ActorCritic`, `PPOConfig`, `PPOTrainer`,
`RolloutBuffer`, `clean_state_dict`.

## `ActorCritic`

```python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512,
                 trunk_layers: int = 2, head_hidden: int | None = None)
```

```
trunk       : trunk_layers × [Linear(hidden) → ELU]            shared
actor_head  : Linear(head_hidden) → ELU → Linear(act_dim)      Gaussian mean
critic_head : Linear(head_hidden) → ELU → Linear(1)            state value
log_std     : nn.Parameter(zeros(act_dim))                     learned, state-independent
```

`head_hidden=None` (or `0`) means `hidden`. Initialisation is orthogonal
with gain √2 everywhere, then gain `0.01` on the actor output and `1.0` on
the critic output; biases are zero. The defaults reproduce the original
locomotion network; the avoidance network of the experiment scripts is
`ActorCritic(36, 3, hidden=128, trunk_layers=3, head_hidden=64)`.

State-dict keys, the checkpoint contract: `trunk.{2i}.weight/bias`,
`actor_head.{0,2}.weight/bias`, `critic_head.{0,2}.weight/bias`, `log_std`.

| Method | Returns | Notes |
|--------|---------|-------|
| `forward(obs)` | `(mean [N, act], std [N, act], value [N])` | `std = exp(log_std)` broadcast |
| `get_action(obs, deterministic=False)` | `(action [N, act], log_prob [N], value [N])` | sampling uses `rsample`; `deterministic=True` returns the mean |
| `get_value(obs)` | `value [N]` | critic only |
| `evaluate(obs, action)` | `(log_prob [N], entropy [N], value [N])` | the PPO update path |
| `num_parameters` | int | property |
| `ActorCritic.from_state_dict(sd)` | a new network | classmethod; see below |

```python
@classmethod
def from_state_dict(cls, sd: dict) -> ActorCritic
```

Rebuilds the architecture purely from weight shapes: `obs_dim` from
`trunk.0.weight`, `act_dim` from `log_std`, `hidden` from the first trunk
layer, `trunk_layers` from the number of `trunk.*.weight` keys,
`head_hidden` from `actor_head.0.weight`. Keys are first passed through
`clean_state_dict`, so weights saved from a `torch.compile`d or
`DataParallel` model (`_orig_mod.` / `module.` prefixes) load unchanged.
This is how legacy script checkpoints, which carry no architecture config,
are loaded for inference. Raises `KeyError` if the dict is not from this
family.

```python
def clean_state_dict(sd: dict) -> dict     # strips "_orig_mod." and "module." from every key
```

## `RolloutBuffer`

```python
class RolloutBuffer:
    def __init__(self, rollout_steps: int, n_envs: int, obs_dim: int, act_dim: int, device)
```

Pre-allocated float32 storage for one rollout of `T = rollout_steps` steps
over `N = n_envs` envs: `obs [T, N, obs_dim]`, `actions [T, N, act_dim]`,
`log_probs`, `values`, `rewards`, `dones`, `advantages`, `returns` all
`[T, N]`. `ptr` is the next timestep to write.

| Method | Meaning |
|--------|---------|
| `store_step(obs, actions, log_probs, values)` | the policy's view of timestep `ptr`, **before** `env.step` (detached copies) |
| `store_outcome(rewards, dones)` | the env's response, **after** `env.step`; advances `ptr` |
| `compute_gae(last_value, gamma=0.99, lam=0.95)` | backward GAE-λ recursion into `advantages`; `returns = advantages + values`; resets `ptr` to 0 |
| `get_flat()` | `(obs, actions, log_probs, advantages, returns, values)` viewed as `[T·N, ...]` |

The recursion is the textbook one:

```
δ_t   = r_t + γ · V_{t+1} · (1 − done_t) − V_t          (V_T = last_value)
A_t   = δ_t + γ λ (1 − done_t) · A_{t+1}
```

`dones` cut the recursion, so nothing bootstraps through a reset. Time-outs
are stored as dones too (the trainer passes `reset_buf`), which is a known
simplification (see [Known limitations](#known-limitations)).
`tests/test_rl_ppo.py::test_gae_matches_reference_recursion` pins this
against an explicit loop.

## `PPOConfig`

Serialised into every checkpoint as `ppo_config`; `PPOConfig(**ckpt["ppo_config"])`
rebuilds it.

| Field | Default | Meaning |
|-------|---------|---------|
| `total_steps` | `100_000_000` | env steps to train (`num_envs × rollout_steps × updates`) |
| `rollout_steps` | `24` | `T`: steps per env per update |
| `minibatch_size` | `8192` | samples per gradient step |
| `n_epochs` | `5` | passes over each rollout |
| `gamma` | `0.99` | discount |
| `lam` | `0.95` | GAE λ |
| `clip_eps` | `0.2` | clipping range for the probability ratio **and** the value |
| `lr` | `3e-4` | Adam learning rate (Adam `eps = 1e-5`, as in the original trainer) |
| `vf_coef` | `1.0` | value-loss weight |
| `ent_coef` | `0.01` | entropy-bonus weight |
| `max_grad_norm` | `1.0` | gradient clipping |
| `hidden_size` | `512` | `ActorCritic` trunk width |
| `trunk_layers` | `2` | trunk depth |
| `head_hidden` | `None` | head width; `None` → `hidden_size` |
| `target_kl` | `None` | when set, stop the remaining epochs of an update once the mean approximate KL of an epoch exceeds `1.5 × target_kl` |
| `lr_schedule` | `"constant"` | `"constant"` or `"linear"`: decay from `lr` to `lr × lr_floor_frac` over `total_steps` |
| `lr_floor_frac` | `0.05` | floor of the linear decay |
| `guard_nonfinite` | `False` | snapshot the network before the update and restore it if any minibatch produces a non-finite loss or gradient norm |
| `vloss_skip` | `None` | when set, roll the update back if a minibatch value loss exceeds this |
| `run_dir` | `"runs/experiment"` | checkpoints and TensorBoard events |
| `log_interval` | `10` | updates between console / TensorBoard lines |
| `save_interval` | `100` | updates between checkpoints |
| `ep_stat_window` | `20` | finished episodes averaged for `mean_return` / `mean_length` |

The two guards share a mechanism: with either enabled the whole update is
treated as atomic. The network state is cloned before the epochs; a bad
minibatch breaks out and the clone is restored, so nothing of that update
survives. Guarded updates also clamp normalised advantages to `±10` and
skip the update outright if advantages or returns contain non-finite
values. The KL estimator is Schulman's `k3`, `mean((ratio − 1) − log ratio)`.

## `PPOTrainer`

```python
class PPOTrainer:
    def __init__(self, env, cfg: PPOConfig, extra_checkpoint_data: dict | None = None)
    def train(self) -> None
    def save_checkpoint(self, tag: str | None = None) -> str
    def load_state(self, ckpt: dict) -> None
    update_callback: Callable[[PPOTrainer, int], None] | None
```

`env` is any VecTask-compatible object, already constructed.
`extra_checkpoint_data` is a caller-owned dict written into every
checkpoint; the entry points store `{"task_config": asdict(task_cfg)}` so a
run can be rebuilt from its own file. The constructor builds the
`ActorCritic` from `cfg` on `env.device`, an Adam optimiser, the
`RolloutBuffer`, and opens a TensorBoard `SummaryWriter` on `run_dir`
(`tensorboard` is imported unconditionally, hence the `rl` extra).

Attributes: `net`, `opt`, `buf`, `global_step` (env steps consumed,
restored by `load_state`), `ep_returns` / `ep_lengths` (every finished
episode's undiscounted return and length, in rollout order), `writer`.

`update_callback(trainer, update_idx)` is called after every PPO update;
`domo.eureka` uses it to take reward-reflection snapshots.

### The loop

```
train():
  obs = env.reset()
  n_updates = total_steps // (rollout_steps × num_envs)
  for update in 1 .. n_updates:
      _apply_lr_schedule()                      linear decay when configured
      obs = _collect_rollout(obs)               T steps under no_grad with the stochastic policy
          │  for t in 0..T-1:
          │      action, logp, value = net.get_action(obs)
          │      buf.store_step(obs, action, logp, value)
          │      obs, _, reward, reset_buf, _ = env.step(action)
          │      buf.store_outcome(reward, reset_buf)
          │      _track_episode_stats(reward, reset_buf)
          │  buf.compute_gae(net.get_value(obs), gamma, lam)
      metrics = _ppo_update()                   n_epochs × minibatches
          │  normalise advantages over the rollout
          │  for epoch: shuffle; for minibatch:
          │      loss = policy + vf_coef · value + ent_coef · (−entropy)
          │      backward, clip_grad_norm_(max_grad_norm), opt.step()
          │      [target_kl early stop] [guards: roll back and stop]
      global_step += rollout_steps × num_envs
      update_callback(self, update)
      every log_interval  → _log(metrics)        console line + TensorBoard scalars
      every save_interval → save_checkpoint()    checkpoint_step_<step:09d>.pt
  save_checkpoint(tag="final")                   checkpoint_final.pt
```

Envs are never reset by the trainer between rollouts: the next rollout
starts from the last observation. The losses are the clipped surrogate
`−min(ratio · A, clip(ratio, 1 ± ε) · A)`, the clipped value loss
`max((V − R)², (clip(V, V_old ± ε) − R)²)` and the entropy bonus.

### Checkpoints

`save_checkpoint` writes `<run_dir>/checkpoint_<tag>.pt` or, without a
tag, `<run_dir>/checkpoint_step_<global_step:09d>.pt`. The **library
format**, read by `domo.checkpoints`, `domo.policies`, `domo.eureka.worker`
and every example:

```python
{
    "step":        int,                       # global env steps so far
    "model_state": dict,                      # ActorCritic.state_dict()
    "optim_state": dict,                      # Adam state_dict()
    "ppo_config":  dict,                      # dataclasses.asdict(PPOConfig)
    "extra":       dict,                      # caller-owned, e.g. {"task_config": asdict(task_cfg)}
    "metrics":     {"mean_return": float,     # over the last ep_stat_window finished episodes
                    "mean_length": float},
}
```

The Eureka worker stores `{"task": <registry key>, "task_config":
<task_overrides>, "reward_code": <str>}` in `extra`.

The frozen scripts under `scripts/` wrote the **legacy format**, which
`policies/walk.pt` and `policies/avoid.pt` still use:

```python
{
    "step":        int,
    "model_state": dict,                      # same ActorCritic keys, possibly with _orig_mod./module. prefixes
    "optim_state": dict,
    "config":      dict,                      # ONE flat dict mixing task fields (n_envs, dt, max_episode_steps,
                                              # headless, ...) and PPO fields (lr, clip_eps, hidden_size, ...)
    "metrics":     {"mean_return": float, "mean_length": float},
    "obs_dim":     int,                       # walk.pt only
    "act_dim":     int,                       # walk.pt only
}
```

`ActorCritic.from_state_dict(ckpt["model_state"])` loads either for
inference; `domo.checkpoints.configs_from_checkpoint(ckpt, kind)` recovers
`(task_cfg, PPOConfig)` from either, filling the task fields the flat
`config` never stored with the dataclass defaults
([world-and-services.md](world-and-services.md#domocheckpoints)).

`load_state(ckpt)` restores `model_state`, `optim_state` and `step` into a
trainer whose network was built with the same architecture; use
`PPOConfig(**ckpt["ppo_config"])` (or `configs_from_checkpoint`) to
guarantee that. It reads only `model_state`, `optim_state`, `step` and
`metrics["mean_return"]`, all of which the legacy format also carries, so
`examples/locomotion/go2_cpg_rl.py --resume` accepts both formats.

### Logging

Every `log_interval` updates, one console line

```
  step  9,830,400 | ret  21.317 | len   987 | ploss -0.0123 | vloss  0.4210 | clip 0.08 |  61,440 sps
```

and these TensorBoard scalars at `global_step`:

| Tag | Meaning |
|-----|---------|
| `train/mean_return` | mean undiscounted return of the last `ep_stat_window` finished episodes |
| `train/mean_ep_len` | mean length of the same episodes |
| `loss/policy` | mean clipped-surrogate loss over the applied minibatches |
| `loss/value` | mean clipped value loss |
| `loss/entropy` | mean policy entropy (positive) |
| `train/clip_fraction` | fraction of samples whose ratio left `1 ± clip_eps` |
| `train/steps_per_sec` | `global_step / elapsed` |
| `train/lr` | current learning rate |

When a guarded update is rolled back, or a KL stop leaves no applied
minibatch, the loss metrics for that update are zeros.

## `main.py`

```
python main.py [--n-envs 4096] [--total-steps 100000000] [--rollout-steps 24]
               [--device cpu|cuda|mps] [--run-dir runs/go2_walk] [--headless]
               [--terrain flat|rough] [--resume CKPT] [--eval CKPT]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-envs` | `4096` | parallel envs |
| `--total-steps` | `100_000_000` | env steps |
| `--rollout-steps` | `24` | steps per rollout |
| `--device` | `cuda` | `cpu`, `cuda` or `mps` |
| `--run-dir` | `runs/go2_walk` | checkpoints and TensorBoard events |
| `--headless` | on | `store_true` with default `True`: a no-op kept for CLI compatibility; training is always headless |
| `--terrain` | `flat` | `flat` or `rough` |
| `--resume` | none | checkpoint to continue from |
| `--eval` | none | checkpoint to evaluate instead of training |

Three modes:

* **train**: `build_configs(args)` returns `Go2WalkConfig(n_envs, dt=0.02,
  max_episode_steps=1000, device, headless, terrain)` and
  `PPOConfig(total_steps, rollout_steps, minibatch_size = max(n_envs ·
  rollout_steps // 4, 256), hidden_size=512, run_dir)`; then
  `PPOTrainer(env, ppo_cfg, extra_checkpoint_data={"task_config":
  asdict(task_cfg)}).train()`.
* **`--resume CKPT`**: `load_checkpoint` maps the file onto CUDA when
  available, else CPU; `PPOConfig(**ckpt["ppo_config"])` and
  `Go2WalkConfig(**ckpt["extra"]["task_config"])` rebuild both configs with
  only the device overridden; `load_state(ckpt)`; `train()`. The CLI flags
  other than `--resume` are ignored.
* **`--eval CKPT`**: `evaluate(path, n_episodes=10, headless=False)` rebuilds
  the task from the checkpoint's config with `n_envs=1` and the viewer on,
  rebuilds the network with the checkpoint's architecture via
  `build_policy(ckpt, ppo_cfg, num_obs, num_actions)`, and runs 10 episodes
  with **stochastic** actions (as in training), printing return and length
  per episode and their means. The device is chosen as for `--resume`;
  `--device` is ignored.

Helpers, all importable from the module (`tests/test_rl_networks.py`
imports `main.py` by path to pin the parser and `build_policy`):

```python
def build_configs(args) -> tuple[Go2WalkConfig, PPOConfig]
def load_checkpoint(path: str) -> tuple[dict, str]              # (ckpt, device)
def build_policy(ckpt: dict, ppo_cfg: PPOConfig, num_obs: int, num_actions: int) -> ActorCritic
def evaluate(checkpoint_path: str, n_episodes: int = 10, headless: bool = False)
def build_parser() -> argparse.ArgumentParser
def main()
```

## Known limitations

* **Resuming trains `total_steps` more.** `train()` computes
  `n_updates = total_steps // (rollout_steps × num_envs)` without
  subtracting the restored `global_step`, so `--resume` on a finished run
  trains a full budget again (with the checkpoint's own `total_steps`).
  Checkpoint step numbers keep counting from the restored value, so the
  files do not collide.
* **Time-outs are not bootstrapped.** `compute_gae` treats
  `extras["time_outs"]` like any other done; the last value of a truncated
  episode is dropped rather than bootstrapped. With 20 s episodes and
  `γ = 0.99` the effect is small, which is why the original trainer did
  the same.
* **`main.py --headless` cannot be turned off.** The flag is `store_true`
  with default `True`. Evaluation always opens the viewer (`evaluate` is
  called with its default `headless=False`).
* **`--eval` and `--resume` ignore `--device`**; they pick CUDA when
  available, otherwise CPU.
* **`tensorboard` is a hard import** of `domo.rl.ppo`. Importing `domo.rl`
  without the `rl` extra fails; `ActorCritic` and `clean_state_dict` are
  reachable through `domo.rl.networks` only after that import succeeds,
  because the package `__init__` imports `ppo` too.
* **Resuming a legacy checkpoint rebuilds the task from defaults.** The
  flat `config` only stored `n_envs`, `dt`, `max_episode_steps` and
  `headless`; every other task field (gains, command ranges, reward
  weights) comes from the current dataclass defaults, which may differ from
  what the frozen script used.
