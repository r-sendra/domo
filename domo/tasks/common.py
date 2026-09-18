"""
Helpers shared by the Go2 tasks.

Each function here is logic that was duplicated verbatim across two or more
tasks (velocity-command sampling, periodic resampling, fall detection). They
are pure tensor functions — no engine, no task state — so the per-task
`step()` bodies read as a sequence of named stages while the numerical
behaviour (including the RNG draw order that training reproducibility
depends on) is unchanged.
"""

from __future__ import annotations

import torch

from domo.robot.state import RobotState

from .base import rand_uniform

__all__ = [
    "envs_due_for_resample",
    "fall_termination",
    "sample_velocity_commands",
]


def sample_velocity_commands(commands: torch.Tensor, envs_idx: torch.Tensor,
                             lin_vel_x_range: tuple[float, float],
                             lin_vel_y_range: tuple[float, float],
                             ang_vel_range: tuple[float, float],
                             device: torch.device) -> None:
    """
    Draw new (vx, vy, vyaw) body-frame velocity commands for `envs_idx`.

    Writes into `commands` [N, 3] in place. Draw order is x, then y, then
    yaw — one `torch.rand` call each — so seeded runs stay reproducible.
    Ranges are (low, high) in m/s, m/s, rad/s.
    """
    if len(envs_idx) == 0:
        return
    n = (len(envs_idx),)
    commands[envs_idx, 0] = rand_uniform(*lin_vel_x_range, n, device)
    commands[envs_idx, 1] = rand_uniform(*lin_vel_y_range, n, device)
    commands[envs_idx, 2] = rand_uniform(*ang_vel_range, n, device)


def envs_due_for_resample(episode_length_buf: torch.Tensor,
                          resampling_time_s: float, dt: float) -> torch.Tensor:
    """
    Indices [K] of envs whose step count is a multiple of the resampling
    period (`resampling_time_s / dt` steps, truncated to int). Envs at step
    0 are excluded only because `reset_idx` resamples them itself.
    """
    resample_every = int(resampling_time_s / dt)
    return ((episode_length_buf % resample_every == 0)
            .nonzero(as_tuple=False).flatten())


def fall_termination(state: RobotState, pitch_limit: float, roll_limit: float,
                     height_limit: float | None = None) -> torch.Tensor:
    """
    Bool [N]: the base has tipped past `pitch_limit` / `roll_limit` [rad]
    (|euler| test, intrinsic xyz) or, when `height_limit` [m] is given, sunk
    below it. The height test is skipped on rough terrain, where absolute
    base height is meaningless.
    """
    fallen = torch.abs(state.base_euler[:, 1]) > pitch_limit
    fallen |= torch.abs(state.base_euler[:, 0]) > roll_limit
    if height_limit is not None:
        fallen |= state.base_pos[:, 2] < height_limit
    return fallen
