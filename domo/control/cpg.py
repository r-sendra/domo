"""
Central Pattern Generator: amplitude-controlled phase oscillators, one per
leg, with optional Kuramoto phase coupling toward a trot.

Implements the oscillator model of Bellegarda & Ijspeert, "CPG-RL: Learning
Central Pattern Generators for Quadruped Locomotion", RA-L 2022, as used in
scripts/house_scene/go2_cpg_rl.py. A policy modulates (mu, omega, psi); the
oscillator states (r, theta, phi) are integrated at high rate and mapped to
Cartesian foot targets, then to joint angles via analytic IK.

Pure torch, no engine imports — the same controller runs in any simulator
and on the real robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from domo.robot.state import RobotState

from .kinematics import LegKinematics

__all__ = [
    "CPG_OBS_DIM",
    "CPG_OBS_SCALES",
    "CPGConfig",
    "CPGLegController",
    "CPGOscillators",
    "build_cpg_observation",
]

# The canonical observation interface of every CPG locomotion policy
# checkpoint (76 dims). Anything that drives such a policy — the training
# task, the CPGLocomotionSkill, the real robot — builds it with this exact
# function so the layout can never drift.
CPG_OBS_DIM = 76
CPG_OBS_SCALES = {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}


def build_cpg_observation(state: RobotState, commands: torch.Tensor,
                          commands_scale: torch.Tensor,
                          default_dof_pos: torch.Tensor,
                          last_actions: torch.Tensor,
                          foot_contacts: torch.Tensor,
                          oscillators: CPGOscillators) -> torch.Tensor:
    """The 76-dim observation every CPG locomotion checkpoint expects.

    Args:
        state: RobotState with base velocities, gravity, dof_pos/vel [N, 12].
        commands: body-frame (vx, vy, vyaw) [N, 3].
        commands_scale: per-channel scale [3] (lin_vel, lin_vel, ang_vel).
        default_dof_pos: stance joint angles [12].
        last_actions: previous raw policy action [N, 12].
        foot_contacts: [N, 4] float 0/1 (sensor or stance-phase proxy).
        oscillators: the CPG bank supplying its 24 phase features.

    Returns:
        obs [N, 76].
    """
    s = CPG_OBS_SCALES
    return torch.cat([
        state.base_lin_vel * s["lin_vel"],                       # 3
        state.base_ang_vel * s["ang_vel"],                       # 3
        state.projected_gravity,                                 # 3
        commands * commands_scale,                               # 3
        (state.dof_pos - default_dof_pos) * s["dof_pos"],        # 12
        state.dof_vel * s["dof_vel"],                            # 12
        last_actions,                                            # 12
        foot_contacts,                                           # 4
        oscillators.observation(),                               # 24
    ], dim=-1)

# Trot: diagonal pairs (FR+RL, FL+RR) in phase, opposite pairs π apart.
# Leg order is the canonical [FR, FL, RR, RL].
_TROT_PHASE = (0.0, math.pi, math.pi, 0.0)

# Desired phase offsets phi*[i][j] = theta_j − theta_i for the Kuramoto
# coupling term; row/col in the same leg order.
_TROT_PHI_STAR = (
    (0.0, math.pi, math.pi, 0.0),
    (math.pi, 0.0, 0.0, math.pi),
    (math.pi, 0.0, 0.0, math.pi),
    (0.0, math.pi, math.pi, 0.0),
)

# Initial direction-phase offset relative to the leg phase on reset (as in
# the reference script; small so the gait starts nearly straight).
_RESET_PHI_SCALE = 0.1


@dataclass
class CPGConfig:
    """Oscillator dynamics, policy action ranges and foot-trajectory shaping.

    Defaults are the values the bundled CPG checkpoints were trained with;
    change them only together with a retrained policy.
    """
    # Oscillator dynamics
    a_conv: float = 150.0            # convergence factor (critically damped)
    integration_dt: float = 0.001    # internal step [s] (1 kHz, as the paper)
    coupling_weight: float = 2.0     # Kuramoto coupling toward trot (0 = free)
    # Action ranges: policy tanh outputs map into these
    mu_range: tuple[float, float] = (1.0, 2.0)
    omega_range_hz: tuple[float, float] = (1.5, 3.5)
    psi_max: float = 1.5             # [rad/s]
    # Foot-trajectory shaping
    d_step: float = 0.2              # max step length scale [m]
    h_nom: float = 0.30              # nominal foot depth below hip [m]
    g_clear: float = 0.08            # max swing ground clearance [m]
    g_pen: float = 0.02              # max stance ground penetration [m]


class CPGOscillators:
    """
    Vectorised bank of 4 coupled oscillators per env.
    State: r (amplitude), rdot, theta (leg phase), phi (direction phase),
    each [n_envs, 4].
    """

    def __init__(self, cfg: CPGConfig, n_envs: int, device: torch.device):
        self.cfg = cfg
        self.n_envs = n_envs
        self.device = device
        f = torch.float32
        self.r = torch.ones((n_envs, 4), device=device, dtype=f)
        self.rdot = torch.zeros((n_envs, 4), device=device, dtype=f)
        self.theta = torch.zeros((n_envs, 4), device=device, dtype=f)
        self.phi = torch.zeros((n_envs, 4), device=device, dtype=f)
        self._trot_phase = torch.tensor(_TROT_PHASE, device=device, dtype=f)
        self._phi_star = torch.tensor(_TROT_PHI_STAR, device=device, dtype=f)
        mu_min, mu_max = cfg.mu_range
        omg_min, omg_max = cfg.omega_range_hz
        self.mu_mid, self.mu_half = 0.5 * (mu_max + mu_min), 0.5 * (mu_max - mu_min)
        self.omg_mid, self.omg_half = 0.5 * (omg_max + omg_min), 0.5 * (omg_max - omg_min)

    def map_action(self, raw: torch.Tensor):
        """Squash a raw [N, 12] policy action into (mu, omega_hz, psi), each [N, 4].

        tanh maps each 4-block into its configured range: mu ∈ mu_range,
        omega ∈ omega_range_hz, psi ∈ ±psi_max.
        """
        t = torch.tanh(raw)
        mu = self.mu_mid + self.mu_half * t[:, 0:4]
        omega_hz = self.omg_mid + self.omg_half * t[:, 4:8]
        psi = self.cfg.psi_max * t[:, 8:12]
        return mu, omega_hz, psi

    def step(self, mu: torch.Tensor, omega_hz: torch.Tensor, psi: torch.Tensor,
             control_dt: float) -> None:
        """Integrate the oscillator ODEs over one control step (Euler sub-steps).

        Amplitude follows the critically damped second-order law of the paper,
        r̈ = a (a/4 (μ − r) − ṙ); phase advances at ω plus the Kuramoto
        coupling Σ_j sin(θ_j − θ_i − φ*_ij) pulling the legs into a trot.
        """
        cfg = self.cfg
        n_sub = max(1, round(control_dt / cfg.integration_dt))
        dt_c = control_dt / n_sub
        a = cfg.a_conv
        omega = 2.0 * math.pi * omega_hz
        for _ in range(n_sub):
            if cfg.coupling_weight != 0.0:
                theta_i = self.theta.unsqueeze(2)
                theta_j = self.theta.unsqueeze(1)
                coupling = cfg.coupling_weight * torch.sum(
                    torch.sin(theta_j - theta_i - self._phi_star), dim=2)
            else:
                coupling = 0.0
            r_ddot = a * (0.25 * a * (mu - self.r) - self.rdot)
            self.rdot = self.rdot + r_ddot * dt_c
            self.r = self.r + self.rdot * dt_c
            self.theta = self.theta + (omega + coupling) * dt_c
            self.phi = self.phi + psi * dt_c
        self.theta = torch.remainder(self.theta, 2.0 * math.pi)
        self.phi = torch.remainder(self.phi, 2.0 * math.pi)

    def stance_mask(self) -> torch.Tensor:
        """[N, 4] float 1.0 where the leg is in stance (sin(theta) < 0)."""
        return (torch.sin(self.theta) < 0).float()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """Restart the given envs at unit amplitude in the trot phase pattern."""
        self.r[envs_idx] = 1.0
        self.rdot[envs_idx] = 0.0
        self.theta[envs_idx] = self._trot_phase
        self.phi[envs_idx] = self._trot_phase * _RESET_PHI_SCALE

    def observation(self) -> torch.Tensor:
        """[N, 24] oscillator features: r, rdot, cos/sin(theta), cos/sin(phi)."""
        return torch.cat([
            self.r, self.rdot,
            torch.cos(self.theta), torch.sin(self.theta),
            torch.cos(self.phi), torch.sin(self.phi),
        ], dim=-1)


class CPGLegController:
    """
    Full CPG → foot-target → IK pipeline: raw policy action [N, 12] in,
    joint position targets [N, 12] out (canonical joint order).
    """

    def __init__(self, cfg: CPGConfig, kinematics: LegKinematics,
                 n_envs: int, device: torch.device):
        self.cfg = cfg
        self.kin = kinematics
        self.oscillators = CPGOscillators(cfg, n_envs, device)

    def joint_targets(self, raw_action: torch.Tensor, control_dt: float) -> torch.Tensor:
        """Advance the oscillators one control step and return joint targets [N, 12].

        Foot targets in each hip frame follow the paper's mapping: the step
        amplitude d_step·(r − 1) swings the foot along the direction phase φ
        (cos φ forward, sin φ sideways), and the vertical profile lifts by
        g_clear during swing (sin θ > 0) and presses g_pen into the ground
        during stance, around the nominal depth h_nom. The lateral offset
        ±l_hip keeps the foot under the abduction joint.
        """
        cfg, osc = self.cfg, self.oscillators
        mu, omega_hz, psi = osc.map_action(raw_action)
        osc.step(mu, omega_hz, psi, control_dt)

        amp = cfg.d_step * (osc.r - 1.0)
        ct, st = torch.cos(osc.theta), torch.sin(osc.theta)
        cp, sp = torch.cos(osc.phi), torch.sin(osc.phi)
        px = -amp * ct * cp
        py = self.kin.side_sign * self.kin.geo.l_hip - amp * ct * sp
        z_clear = torch.where(st > 0, cfg.g_clear * st, cfg.g_pen * st)
        pz = -cfg.h_nom + z_clear

        qh, qt, qc = self.kin.ik(px, py, pz)
        targets = torch.empty((raw_action.shape[0], 12),
                              device=raw_action.device, dtype=raw_action.dtype)
        targets[:, 0::3] = qh
        targets[:, 1::3] = qt
        targets[:, 2::3] = qc
        return targets

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """Reset the oscillator bank for the given envs."""
        self.oscillators.reset_idx(envs_idx)
