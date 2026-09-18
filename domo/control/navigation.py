"""
Waypoint navigation on top of a velocity-tracking policy.

Port of PositionController from scripts/house_scene/evaluate_nav.py,
generalised: instead of holding an env + network, it drives two injected
callables, so the identical controller runs in simulation or on the real
robot —

    step_fn(commands: [1, 3] tensor) -> None
        advance one control step tracking (vx, vy, vyaw)
    pose_fn() -> (x, y, yaw)
        current planar pose estimate (sim ground truth / odometry / mocap)

An optional `intervention` object (see examples) can pause/abort/stop and
adjust speed; pass None to run uninterrupted.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["NavConfig", "PositionController"]

StepFn = Callable[[torch.Tensor], None]
PoseFn = Callable[[], tuple[float, float, float]]


@dataclass
class NavConfig:
    dt: float = 0.02
    max_vx: float = 0.8         # m/s
    max_vyaw: float = 0.8       # rad/s
    kp_lin: float = 0.8
    kp_ang: float = 1.5
    tol_pos: float = 0.15       # m
    tol_ang: float = 0.05       # rad
    min_vx: float = 0.15        # keep moving while far from goal
    drive_timeout_s: float = 60.0
    turn_timeout_s: float = 15.0
    verbose: bool = True


class _NullIntervention:
    paused = False
    stopped = False

    @property
    def aborted(self):
        return False

    def poll(self):
        pass

    def pop_speed_delta(self):
        return 0.0


class PositionController:
    """P-controller producing velocity commands toward planar goals."""

    def __init__(self, step_fn: StepFn, pose_fn: PoseFn,
                 cfg: NavConfig = None, intervention=None,
                 device: str = "cpu"):
        self.step_fn = step_fn
        self.pose_fn = pose_fn
        self.cfg = cfg or NavConfig()
        self.ctrl = intervention or _NullIntervention()
        self._cmd = torch.zeros(1, 3, device=device)

    # ------------------------------------------------------------------
    # High-level commands — return 'done' | 'aborted' | 'stopped' | 'timeout'
    # ------------------------------------------------------------------

    def go_forward(self, distance: float, speed: float | None = None) -> str:
        x, y, yaw = self.pose_fn()
        tx = x + distance * math.cos(yaw)
        ty = y + distance * math.sin(yaw)
        spd = speed or self.cfg.max_vx
        self._say(f"→ go_forward({distance:.2f}m @ {spd:.1f}m/s) "
                  f"target=({tx:.2f},{ty:.2f})")
        return self._drive_to(tx, ty, override_vx=spd)

    def go_backward(self, distance: float, speed: float | None = None) -> str:
        x, y, yaw = self.pose_fn()
        tx = x - distance * math.cos(yaw)
        ty = y - distance * math.sin(yaw)
        spd = speed or self.cfg.max_vx
        self._say(f"→ go_backward({distance:.2f}m @ {spd:.1f}m/s)")
        return self._drive_to(tx, ty, reverse=True, override_vx=spd)

    def turn(self, angle_deg: float) -> str:
        """Positive = left (CCW)."""
        _, _, yaw = self.pose_fn()
        target_yaw = self._wrap(yaw + math.radians(angle_deg))
        self._say(f"→ turn({angle_deg:+.1f}°) target={math.degrees(target_yaw):.1f}°")
        return self._rotate_to(target_yaw)

    def go_to(self, x: float, y: float, speed: float | None = None,
              final_yaw_deg: float | None = None) -> str:
        self._say(f"→ go_to({x:.2f},{y:.2f})")
        result = self._drive_to(x, y, override_vx=speed)
        if result == "done" and final_yaw_deg is not None:
            result = self._rotate_to(math.radians(final_yaw_deg))
        return result

    def stop(self, settle_steps: int = 20):
        self._cmd[:] = 0.0
        for _ in range(settle_steps):
            self._step()
        self._say("→ stop")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _drive_to(self, tx: float, ty: float, reverse: bool = False,
                  override_vx: float | None = None) -> str:
        cfg = self.cfg
        max_steps = int(cfg.drive_timeout_s / cfg.dt)
        vx_limit = override_vx or cfg.max_vx

        for step in range(max_steps):
            outcome = self._check_intervention()
            if outcome:
                return outcome

            delta = self.ctrl.pop_speed_delta()
            if delta != 0.0:
                vx_limit = float(np.clip(vx_limit + delta, 0.1, 3.0))
                self._say(f"  speed adjusted to {vx_limit:.1f} m/s")

            x, y, yaw = self.pose_fn()
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
            if dist < cfg.tol_pos:
                self._cmd[:] = 0.0
                return "done"

            desired_yaw = math.atan2(dy, dx)
            if reverse:
                desired_yaw = self._wrap(desired_yaw + math.pi)
            yaw_err = self._wrap(desired_yaw - yaw)
            alignment = math.cos(yaw_err)
            sign = -1.0 if reverse else 1.0

            vx = sign * float(np.clip(
                cfg.kp_lin * dist * max(alignment, 0.0), cfg.min_vx, vx_limit))
            vyaw = float(np.clip(cfg.kp_ang * yaw_err,
                                 -cfg.max_vyaw, cfg.max_vyaw))
            self._cmd[0, 0], self._cmd[0, 1], self._cmd[0, 2] = vx, 0.0, vyaw
            self._step()

            if cfg.verbose and step % 100 == 0:
                self._say(f"  dist={dist:.2f}m vx={vx:.2f} vyaw={vyaw:.2f} "
                          f"pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):.1f}°")
        return "timeout"

    def _rotate_to(self, target_yaw: float) -> str:
        cfg = self.cfg
        max_steps = int(cfg.turn_timeout_s / cfg.dt)
        for _ in range(max_steps):
            outcome = self._check_intervention()
            if outcome:
                return outcome

            _, _, yaw = self.pose_fn()
            yaw_err = self._wrap(target_yaw - yaw)
            if abs(yaw_err) < cfg.tol_ang:
                self._cmd[:] = 0.0
                return "done"

            self._cmd[0, 0], self._cmd[0, 1] = 0.0, 0.0
            self._cmd[0, 2] = float(np.clip(
                cfg.kp_ang * yaw_err, -cfg.max_vyaw, cfg.max_vyaw))
            self._step()
        return "timeout"

    def _check_intervention(self) -> str | None:
        if self.ctrl.stopped:
            self.stop()
            return "stopped"
        if self.ctrl.aborted:
            self.stop()
            return "aborted"
        while self.ctrl.paused:
            self._cmd[:] = 0.0
            self._step()
            if self.ctrl.stopped:
                return "stopped"
            if self.ctrl.aborted:
                return "aborted"
        return None

    def _step(self):
        self.ctrl.poll()
        self.step_fn(self._cmd)

    def _say(self, msg: str):
        if self.cfg.verbose:
            print(f"  {msg}")

    @staticmethod
    def _wrap(a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a
