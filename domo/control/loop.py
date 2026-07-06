"""
ControlLoop: the metronome under the Skill/Controller hierarchy.

The loop is the ONLY component that knows how time advances — by stepping a
simulator or by pacing against the wall clock on hardware. Controllers and
skills run byte-identical in both. Cycle:

    command = controller.update(robot.state, dt)   # brain
    command = command_filter(command, state)       # M7 safety choke point
    robot.set_joint_targets(command)               # actuator write
    <time advances>                                # scene.step() | sleep-to-rate
    robot.refresh()                                # sensors → RobotState
    sensor.tick() ...                              # exteroceptive sensors

`command_filter` is the reserved seat for the non-learned safety layer:
every command from any layer above passes through it, unbypassably, before
reaching the actuators.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import torch

__all__ = ["SimControlLoop", "RealControlLoop"]

CommandFilter = Callable[[torch.Tensor, object], torch.Tensor]


class _ControlLoopBase:

    def __init__(self, robot, controller, dt: float,
                 sensors: Sequence = (),
                 command_filter: Optional[CommandFilter] = None):
        self.robot = robot
        self.controller = controller
        self.dt = dt
        self.sensors = list(sensors)
        self.command_filter = command_filter

    def reset(self):
        all_envs = torch.arange(self.robot.n_envs, device=self.robot.device)
        self.robot.reset_idx(all_envs)
        self.controller.reset_idx(all_envs)
        for sensor in self.sensors:
            if hasattr(sensor, "reset_idx"):
                sensor.reset_idx(all_envs)
        self.robot.refresh()
        return self.robot.state

    def _cycle_pre(self):
        command = self.controller.update(self.robot.state, self.dt)
        if self.command_filter is not None:
            command = self.command_filter(command, self.robot.state)
        self.robot.set_joint_targets(command)

    def _cycle_post(self):
        self.robot.refresh()
        for sensor in self.sensors:
            if hasattr(sensor, "tick"):
                sensor.tick()
        return self.robot.state

    def run(self, n_steps: int, callback: Optional[Callable] = None):
        """Run n control cycles; callback(step, state) after each if given."""
        state = self.robot.state
        for i in range(n_steps):
            state = self.step()
            if callback is not None:
                callback(i, state)
        return state


class SimControlLoop(_ControlLoopBase):
    """Time advances by stepping the physics scene."""

    def __init__(self, scene, robot, controller, dt: float,
                 sensors: Sequence = (),
                 command_filter: Optional[CommandFilter] = None):
        super().__init__(robot, controller, dt, sensors, command_filter)
        self.scene = scene

    def step(self):
        self._cycle_pre()
        self.scene.step()
        return self._cycle_post()


class RealControlLoop(_ControlLoopBase):
    """
    Time advances on the wall clock, paced to 1/dt Hz. `robot` must be a
    real-robot implementation of the Robot facade (sensors reading DDS,
    actuator writing low-level commands). Structural skeleton for the M4
    deployment path — untested on hardware.
    """

    def step(self):
        cycle_start = time.monotonic()
        self._cycle_pre()
        remaining = self.dt - (time.monotonic() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)
        else:
            print(f"  [loop] overrun: cycle took {self.dt - remaining:.4f}s "
                  f"(budget {self.dt:.4f}s)")
        return self._cycle_post()
