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
from collections.abc import Callable, Sequence

import torch

from .skill import all_envs

__all__ = ["RealControlLoop", "SimControlLoop"]

# command_filter(command [N, D], robot_state) → filtered command [N, D].
CommandFilter = Callable[[torch.Tensor, object], torch.Tensor]


class _ControlLoopBase:
    """Shared half-cycles of every control loop; subclasses define `step()`.

    Sensors are duck-typed: anything with `tick()` is refreshed after time
    advances, anything with `reset_idx()` is reset with the robot.
    """

    def __init__(self, robot, controller, dt: float,
                 sensors: Sequence = (),
                 command_filter: CommandFilter | None = None):
        """
        Args:
            robot: The robot facade to control (already bound).
            controller: The Controller producing joint targets each cycle.
            dt: The time step for each control cycle in seconds.
            sensors: Exteroceptive sensors (e.g. lidar) to tick each cycle.
            command_filter: Optional safety layer applied to every command
                before it reaches the actuators (M7 seat).
        """
        self.robot = robot
        self.controller = controller
        self.dt = dt
        self.sensors = list(sensors)
        self.command_filter = command_filter

    def reset(self):
        """
        Reset robot, controller and sensors for all envs (start of a run or
        episode) and return the fresh RobotState.
        """
        envs = all_envs(self.robot)
        self.robot.reset_idx(envs)
        self.controller.reset_idx(envs)
        for sensor in self.sensors:
            if hasattr(sensor, "reset_idx"):
                sensor.reset_idx(envs)
        self.robot.refresh()
        return self.robot.state

    def step(self):
        """One full control cycle: pre-half, advance time, post-half."""
        raise NotImplementedError

    def _cycle_pre(self):
        """
        The first half of a control cycle: get a command from the controller,
        pass it through the safety filter, and send it to the robot's actuators.
        """
        command = self.controller.update(self.robot.state, self.dt)
        if self.command_filter is not None:
            command = self.command_filter(command, self.robot.state)
        self.robot.set_joint_targets(command)

    def _cycle_post(self):
        """
        The second half of a control cycle: after time has advanced, refresh the
        robot's internal state from its sensors and tick any external sensors.
        """
        self.robot.refresh()
        for sensor in self.sensors:
            if hasattr(sensor, "tick"):
                sensor.tick()
        return self.robot.state

    def run(self, n_steps: int, callback: Callable | None = None):
        """
        Run a fixed number of control cycles.

        Args:
            n_steps: The number of control steps to execute.
            callback: An optional function `callback(step_index, robot_state)`
                called after each step.

        Returns:
            The RobotState after the last cycle.
        """
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
                 command_filter: CommandFilter | None = None):
        """
        Args:
            scene: The physics scene handle (`scene.step()` advances time by
                the scene's own dt — it must equal `dt`).
            robot: The robot instance in the simulation.
            controller: The controller orchestrating the robot's skills.
            dt: The control loop time step.
            sensors: External sensors to update each step.
            command_filter: Optional safety filter for outgoing commands.
        """
        super().__init__(robot, controller, dt, sensors, command_filter)
        self.scene = scene

    def step(self):
        """Executes one full control cycle by stepping the physics scene."""
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
        """Executes one full control cycle, paced against the system wall clock."""
        cycle_start = time.monotonic()
        self._cycle_pre()
        remaining = self.dt - (time.monotonic() - cycle_start)
        if remaining > 0:
            time.sleep(remaining)
        else:
            # Overruns are reported, not raised: on hardware a late cycle is
            # better than a dead loop. The M7 layer decides what to do with them.
            print(f"  [loop] overrun: cycle took {self.dt - remaining:.4f}s "
                  f"(budget {self.dt:.4f}s)")
        return self._cycle_post()
