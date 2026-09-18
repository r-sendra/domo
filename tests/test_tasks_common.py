"""
VecTask contract and the shared task helpers, engine-free.

Covers the pieces of domo.tasks that other packages depend on by name:
the reward registry / override semantics (domo.eureka), the time-out
extras, the observation builders' layouts (checkpoints), the shared
step helpers in domo.tasks.common, and the config defaults that are part
of the checkpoint contract.
"""

import math
from dataclasses import fields

import pytest
import torch

from domo.robot.state import RobotState
from domo.tasks import (
    AVOID_ACT_DIM,
    AVOID_OBS_DIM,
    CPG_ACT_DIM,
    CPG_OBS_DIM,
    GETUP_ACT_DIM,
    GETUP_OBS_DIM,
    Go2AvoidConfig,
    Go2AvoidTask,
    Go2CPGWalkConfig,
    Go2CPGWalkTask,
    Go2GetUpConfig,
    Go2GetUpTask,
    Go2WalkConfig,
    Go2WalkTask,
    VecTask,
    build_getup_observation,
    rand_uniform,
)
from domo.tasks.common import (
    envs_due_for_resample,
    fall_termination,
    sample_velocity_commands,
)

N = 4
DEVICE = torch.device("cpu")


class ToyTask(VecTask):
    """Minimal VecTask: two registered terms, no physics."""

    def __init__(self, n_envs=N, dt=0.02, max_len=5):
        super().__init__(n_envs, num_obs=3, num_actions=2, device=DEVICE,
                         dt=dt, max_episode_length=max_len)
        self.register_rewards({"one": 1.0, "two": -2.0})

    def _reward_one(self):
        return torch.ones(self.n_envs)

    def _reward_two(self):
        return torch.full((self.n_envs,), 0.5)

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def reset_idx(self, envs_idx):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# VecTask reward registry / override / bookkeeping
# ---------------------------------------------------------------------------

def test_register_rewards_scales_by_dt_and_binds_methods():
    task = ToyTask()
    assert task.reward_scales == {"one": pytest.approx(0.02), "two": pytest.approx(-0.04)}
    rew = task.compute_rewards()
    # 1 × 0.02 + 0.5 × (−0.04) = 0
    assert torch.allclose(rew, torch.zeros(N))
    assert torch.allclose(task.episode_sums["one"], torch.full((N,), 0.02))
    assert torch.allclose(task.episode_sums["two"], torch.full((N,), -0.02))
    assert rew is task.rew_buf


def test_register_rewards_unknown_term_raises():
    task = ToyTask()
    with pytest.raises(AttributeError):
        task.register_rewards({"does_not_exist": 1.0})


def test_reward_override_replaces_registry_and_tracks_components():
    task = ToyTask()
    task.set_reward_override(
        lambda t: (torch.full((t.n_envs,), 3.0), {"a": torch.ones(t.n_envs)}))
    rew = task.compute_rewards()
    assert torch.allclose(rew, torch.full((N,), 3.0))       # no dt scaling
    assert set(task.reward_components) == {"a"}
    assert torch.allclose(task.episode_sums["a"], torch.ones(N))
    # registered terms are untouched while the override is active
    assert torch.all(task.episode_sums["one"] == 0)
    task.set_reward_override(None)
    task.compute_rewards()
    assert torch.allclose(task.episode_sums["one"], torch.full((N,), 0.02))


def test_log_episode_sums_reports_per_second_and_zeroes():
    task = ToyTask()
    task.compute_rewards()
    task.log_episode_sums(torch.tensor([0, 1]), episode_length_s=2.0)
    assert task.extras["episode"]["rew_one"] == pytest.approx(0.02 / 2.0)
    assert torch.all(task.episode_sums["one"][:2] == 0)
    assert torch.all(task.episode_sums["one"][2:] > 0)


def test_mark_time_outs_flags_budget_exhaustion():
    task = ToyTask(max_len=5)
    task.episode_length_buf[:] = torch.tensor([5, 6, 0, 7], dtype=torch.int32)
    timeout = task.mark_time_outs()
    assert timeout.dtype == torch.bool
    assert timeout.tolist() == [False, True, False, True]
    assert task.extras["time_outs"].dtype == torch.float32
    assert task.extras["time_outs"].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_compute_success_not_defined_by_default():
    with pytest.raises(NotImplementedError):
        ToyTask().compute_success()


def test_rand_uniform_range_and_shape():
    torch.manual_seed(0)
    x = rand_uniform(-2.0, 3.0, (1000,), DEVICE)
    assert x.shape == (1000,) and x.min() >= -2.0 and x.max() < 3.0


# ---------------------------------------------------------------------------
# Shared helpers (domo.tasks.common)
# ---------------------------------------------------------------------------

def test_sample_velocity_commands_draw_order_matches_inline_version():
    """The helper must consume the RNG exactly like the code it replaced."""
    envs = torch.tensor([1, 3])
    ranges = ((0.3, 3.0), (-0.5, 0.5), (-1.0, 1.0))

    torch.manual_seed(7)
    cmd = torch.zeros(N, 3)
    sample_velocity_commands(cmd, envs, *ranges, DEVICE)

    torch.manual_seed(7)
    ref = torch.zeros(N, 3)
    n = (len(envs),)
    ref[envs, 0] = rand_uniform(*ranges[0], n, DEVICE)
    ref[envs, 1] = rand_uniform(*ranges[1], n, DEVICE)
    ref[envs, 2] = rand_uniform(*ranges[2], n, DEVICE)

    assert torch.equal(cmd, ref)
    assert torch.all(cmd[[0, 2]] == 0)          # untouched envs stay zero
    assert torch.all(cmd[envs, 0] >= 0.3)
    # empty index is a no-op and consumes no RNG
    torch.manual_seed(1)
    expected = torch.rand(1)
    torch.manual_seed(1)
    sample_velocity_commands(cmd, torch.tensor([], dtype=torch.long), *ranges, DEVICE)
    assert torch.equal(torch.rand(1), expected)


def test_envs_due_for_resample_period_and_zero_step():
    lengths = torch.tensor([0, 200, 199, 400, 1], dtype=torch.int32)
    due = envs_due_for_resample(lengths, resampling_time_s=4.0, dt=0.02)
    assert due.tolist() == [0, 1, 3]            # 200-step period; step 0 counts


def _state(euler, height):
    s = RobotState.zeros(len(euler), 12, 4, DEVICE)
    s.base_euler[:] = torch.tensor(euler)
    s.base_pos[:, 2] = torch.tensor(height)
    return s


def test_fall_termination_pitch_roll_height():
    s = _state([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, -1.2, 0.0], [0.0, 0.0, 3.0]],
               [0.3, 0.3, 0.3, 0.1])
    assert fall_termination(s, 1.0, 1.0).tolist() == [False, True, True, False]
    assert fall_termination(s, 1.0, 1.0, height_limit=0.2).tolist() == \
        [False, True, True, True]


# ---------------------------------------------------------------------------
# Observation builders and dimension constants
# ---------------------------------------------------------------------------

def test_getup_observation_layout():
    s = RobotState.zeros(2, 12, 4, DEVICE)
    s.base_ang_vel[:] = 4.0                        # × 0.25 → 1
    s.projected_gravity[:] = torch.tensor([0.0, 0.0, -1.0])
    s.dof_pos[:] = 1.5
    s.dof_vel[:] = 20.0                            # × 0.05 → 1
    default = torch.full((12,), 0.5)
    last = torch.full((2, 12), 7.0)
    obs = build_getup_observation(s, default, last)
    assert obs.shape == (2, GETUP_OBS_DIM)
    assert torch.allclose(obs[0, 0:3], torch.ones(3))
    assert torch.allclose(obs[0, 3:6], torch.tensor([0.0, 0.0, -1.0]))
    assert torch.allclose(obs[0, 6:18], torch.ones(12))
    assert torch.allclose(obs[0, 18:30], torch.ones(12))
    assert torch.allclose(obs[0, 30:42], torch.full((12,), 7.0))


def test_dimension_constants_match_task_classes():
    assert (Go2WalkTask.OBS_DIM, Go2WalkTask.ACT_DIM) == (45, 12)
    assert (Go2CPGWalkTask.OBS_DIM, Go2CPGWalkTask.ACT_DIM) == (CPG_OBS_DIM, CPG_ACT_DIM) == (76, 12)
    assert (Go2AvoidTask.OBS_DIM, Go2AvoidTask.ACT_DIM) == (AVOID_OBS_DIM, AVOID_ACT_DIM) == (36, 3)
    assert (Go2GetUpTask.OBS_DIM, Go2GetUpTask.ACT_DIM) == (GETUP_OBS_DIM, GETUP_ACT_DIM) == (42, 12)


# ---------------------------------------------------------------------------
# Config contract (field names + defaults are serialised into checkpoints)
# ---------------------------------------------------------------------------

def _defaults(cls):
    inst = cls()
    return {f.name: getattr(inst, f.name) for f in fields(cls)}


def test_walk_config_defaults():
    d = _defaults(Go2WalkConfig)
    assert d["n_envs"] == 4096 and d["dt"] == 0.02 and d["max_episode_steps"] == 1000
    assert d["action_scale"] == 0.25 and d["kp"] == 20.0 and d["kd"] == 0.5
    assert d["simulate_action_latency"] is True and d["terrain"] == "flat"
    assert d["lin_vel_x_range"] == (0.5, 0.5)
    assert d["termination_height"] == 0.20 and d["base_height_target"] == 0.34
    assert d["reward_scales"] == {
        "tracking_lin_vel": 1.0, "tracking_ang_vel": 0.2, "lin_vel_z": -1.0,
        "base_height": -50.0, "action_rate": -0.005, "similar_to_default": -0.1}
    assert d["obs_scales"] == {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}


def test_cpg_walk_config_defaults():
    d = _defaults(Go2CPGWalkConfig)
    assert d["kp"] == 100.0 and d["kd"] == 2.0 and d["base_init_pos"] == (0.0, 0.0, 0.35)
    assert d["lin_vel_x_range"] == (0.3, 3.0) and d["ang_vel_range"] == (-1.0, 1.0)
    assert d["termination_height"] == 0.18 and d["tracking_sigma"] == 0.5
    assert d["reward_clamp"] == 10.0
    assert d["reward_scales"] == {
        "tracking_lin_vel_x": 0.75, "tracking_lin_vel_y": 0.75, "tracking_ang_vel": 0.75,
        "lin_vel_z": -2.0, "ang_vel_xy": -0.05, "work": -0.001}


def test_avoid_config_defaults():
    d = _defaults(Go2AvoidConfig)
    assert d["scene_kind"] == "arena" and d["obs_max_range"] == 4.0
    assert (d["d_collision"], d["d_danger"], d["d_caution"], d["d_anticipate"]) == (0.25, 0.60, 0.90, 1.40)
    assert (d["delta_vx_max"], d["delta_vy_max"], d["delta_vyaw_max"]) == (0.8, 0.5, 1.5)
    assert d["base_command"] == (0.6, 0.0, 0.0)
    assert d["reward_scales"] == {"survival": 1.0, "avoidance": 5.0,
                                  "smoothness": -0.05, "command_tracking": 3.0}


def test_getup_config_defaults():
    d = _defaults(Go2GetUpConfig)
    assert d["max_episode_steps"] == 400 and d["action_scale"] == 0.5
    assert d["spawn_height"] == 0.18 and d["spawn_joint_noise"] == 0.3
    assert d["spawn_roll_range"] == (math.pi / 3, 5 * math.pi / 6)
    assert (d["success_height"], d["success_tilt"], d["success_hold_steps"]) == (0.26, 0.4, 25)
    assert d["dr"] is None


def test_reward_scale_keys_have_reward_methods():
    """Every configured term must resolve to a `_reward_<name>` method."""
    for cfg_cls, task_cls in [(Go2WalkConfig, Go2WalkTask),
                              (Go2CPGWalkConfig, Go2CPGWalkTask),
                              (Go2AvoidConfig, Go2AvoidTask)]:
        for name in cfg_cls().reward_scales:
            assert callable(getattr(task_cls, f"_reward_{name}")), (task_cls, name)
