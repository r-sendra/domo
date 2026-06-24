"""
go2_cpg_rl.py
=============
CPG-RL for Go2 quadruped locomotion in FREE SPACE (flat terrain, no obstacles).

Reproduces the architecture from:
  G. Bellegarda & A. Ijspeert, "CPG-RL: Learning Central Pattern Generators
  for Quadruped Locomotion", IEEE RA-L 7(4), 2022.

The policy modulates one amplitude-controlled phase oscillator per leg:
    action  a = [ mu (4) , omega (4) , psi (4) ]  in R^12
Oscillator states (r, theta, phi) are integrated, mapped to Cartesian foot
positions, converted to joint targets via closed-form inverse kinematics, and
tracked with joint PD control. Foot trajectories are smooth and rhythmic by
construction (unlike the paper's "Joint PD" baseline, which emits joint offsets
directly and produces erratic, simulator-overfit gaits).

Pipeline per control step:
    policy --(mu,omega,psi)--> CPG ODE --(r,theta,phi)--> foot pos --IK--> q_des --PD--> torque

This version includes:
  * analytic IK validated against the Go2 URDF (round-trip self-test at startup)
  * paper reward (velocity tracking + z/roll-pitch penalties + mechanical work)
  * training-stability guards: per-update KL early-stop, linear LR decay,
    advantage clamping, reward clamping, and atomic rollback on any
    non-finite / exploding update (prevents the late-training value blow-up)
  * checkpoint I/O that is prefix-tolerant (strips torch.compile `_orig_mod.`
    and DataParallel `module.` prefixes) and infers dims architecture-agnostically

Usage:
    # Train on GPU
    python go2_cpg_rl.py --n-envs 4096 --device cuda --headless

    # Evaluate / watch the gait (forward 0.5 m/s demo)
    python go2_cpg_rl.py --eval ../../runs/go2_cpg/checkpoint_final.pt

    # Resume
    python go2_cpg_rl.py --resume ../../runs/go2_cpg/checkpoint_step_000xxxxxxx.pt
"""

import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
)


# ==========================================================================
#  Constants
# ==========================================================================

# Joint order is [FR, FL, RR, RL] x [hip, thigh, calf]. Kept everywhere.
MOTOR_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
FOOT_LINK_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

DEFAULT_JOINT_ANGLES = {
    "FR_hip_joint":   0.0, "FR_thigh_joint":  0.8, "FR_calf_joint": -1.5,
    "FL_hip_joint":   0.0, "FL_thigh_joint":  0.8, "FL_calf_joint": -1.5,
    "RR_hip_joint":   0.0, "RR_thigh_joint":  1.0, "RR_calf_joint": -1.5,
    "RL_hip_joint":   0.0, "RL_thigh_joint":  1.0, "RL_calf_joint": -1.5,
}

# --- Go2 leg geometry (m), from go2_description URDF -----------------------
#   hip->thigh offset (0,-0.0955,0): L_HIP=0.0955 ; thigh & calf links 0.213.
#   If the IK self-test warns, verify these against your urdf/go2/urdf/go2.urdf.
L_HIP, L_THIGH, L_CALF = 0.0955, 0.213, 0.213
LEG_SIDE_SIGN = [-1.0, +1.0, -1.0, +1.0]   # [FR, FL, RR, RL]: right -y, left +y

# --- CPG / foot-trajectory parameters --------------------------------------
A_CONV  = 150.0   # oscillator convergence factor a (paper). Critically damped.
D_STEP  = 0.2    # max step length scale [m]
H_NOM   = 0.30    # nominal foot depth below the hip [m] -> base stands ~0.30 m
G_CLEAR = 0.08    # max swing ground clearance g_c [m]
G_PEN   = 0.02    # max stance ground penetration g_p [m]
CPG_DT  = 0.001   # CPG integration step [s] -> 1 kHz, matches the paper

# Initial phases [FR,FL,RR,RL] = trot (diagonal pairs in phase). With no
# coupling and equal omega, relative phase is set by initial conditions; this
# biases toward a clean trot while leaving the policy free to break phase.
TROT_PHASE = [0.0, math.pi, math.pi, 0.0]

# --- Action ranges (paper Sec III-A) ---------------------------------------
# NOTE: paper writes theta_dot = omega with "omega in [0,4.5] Hz". As rad/s that
# caps gait freq at ~0.7 Hz (too slow for the ~0.5 s trot they report), so omega
# is treated as Hz with theta_dot = 2*pi*omega. psi is taken as rad/s.
MU_MIN,  MU_MAX  = 1.0, 2.0
OMG_MIN, OMG_MAX = 1.5, 3.5
PSI_MAX          = 1.5
MU_MID,  MU_HALF  = 0.5 * (MU_MAX + MU_MIN),  0.5 * (MU_MAX - MU_MIN)
OMG_MID, OMG_HALF = 0.5 * (OMG_MAX + OMG_MIN), 0.5 * (OMG_MAX - OMG_MIN)

CONTACT_FORCE_THRESH = 1.0   # N

# Observation: 3 lin_vel + 3 ang_vel + 3 gravity + 3 cmd + 12 dpos + 12 dvel
#   + 12 act + 4 contacts + 24 cpg(r,rdot,cos/sin theta,cos/sin phi) = 76
OBS_DIM = 76
ACT_DIM = 12


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def clean_state_dict(sd):
    """Strip torch.compile / DataParallel key prefixes so checkpoints load anywhere."""
    return {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}


# ==========================================================================
#  Leg kinematics  (validated to match the Go2 URDF convention exactly)
# ==========================================================================
# Hip frame: origin at the abduction joint, x forward, y left, z up. Abduction
# about +x, thigh and calf about +y. Returned/consumed angles ARE the URDF
# joint values (no sign flips). IK(FK(default)) round-trips to machine epsilon.

def leg_fk(qh, qt, qc, side_sign, l1, l2, l3):
    s = side_sign
    z_s = -l2 * torch.cos(qt) - l3 * torch.cos(qt + qc)
    px = -l2 * torch.sin(qt) - l3 * torch.sin(qt + qc)
    py = s * l1 * torch.cos(qh) - z_s * torch.sin(qh)
    pz = s * l1 * torch.sin(qh) + z_s * torch.cos(qh)
    return px, py, pz


def leg_ik(px, py, pz, side_sign, l1, l2, l3):
    l1_s = side_sign * l1
    L = torch.sqrt(torch.clamp(py * py + pz * pz - l1 * l1, min=1e-8))
    z_s = -L
    qh = torch.atan2(L * py + l1_s * pz, l1_s * py - L * pz)
    D2 = px * px + L * L
    cos_knee = torch.clamp((D2 - l2 * l2 - l3 * l3) / (2.0 * l2 * l3), -1.0, 1.0)
    qc = -torch.acos(cos_knee)
    A = l2 + l3 * cos_knee
    B = l3 * torch.sin(qc)
    qt = torch.atan2(B * z_s - A * px, -B * px - A * z_s)
    return qh, qt, qc


# ==========================================================================
#  Environment
# ==========================================================================

class Go2CPGEnv:
    ACT_DIM = ACT_DIM

    def __init__(self, n_envs=4096, dt=0.02, max_episode_steps=1000,
                 headless=True, device="cuda"):
        self.n_envs   = n_envs
        self.num_envs = n_envs
        self.dt       = dt
        self.device   = torch.device(device)

        self.env_cfg = {
            "base_init_pos":  [0.0, 0.0, 0.35],
            "base_init_quat": [0.0, 0.0, 0.0, 1.0],
            "episode_length_s":  max_episode_steps * dt,
            "resampling_time_s": 4.0,
            "kp": 100.0, "kd": 2.0,                       # paper gains
            "termination_if_pitch_greater_than": 1.0,
            "termination_if_roll_greater_than":  1.0,
            "termination_if_height_less_than":   0.18,
        }
        self.obs_scales = {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}
        self.reward_cfg = {
            "tracking_sigma": 0.15,
            "reward_scales": {                            # paper weights, signed
                "tracking_lin_vel_x":  0.75,
                "tracking_lin_vel_y":  0.75,
                "tracking_ang_vel":    0.50,
                "lin_vel_z":          -2.00,
                "ang_vel_xy":         -0.05,
                "work":               -0.001,
            },
        }
        # DEFAULT = forward only (clean gait first). For omnidirectional
        # (paper Table III) widen to x:[-1,1] y:[-1,1] yaw:[-1,1].
        self.command_cfg = {
            "lin_vel_x_range": [-0.5, 2],
            "lin_vel_y_range": [-0.5, 0.5],
            "ang_vel_range":   [-1.0, 1.0],
        }

        self.reward_scales = {k: v * dt for k, v in self.reward_cfg["reward_scales"].items()}
        self.max_episode_length = math.ceil(self.env_cfg["episode_length_s"] / dt)
        self.num_commands = 3
        self.n_cpg_substeps = max(1, round(dt / CPG_DT))
        self.cpg_dt = dt / self.n_cpg_substeps

        # ── Genesis init ──────────────────────────────────────────────────
        backend = gs.cuda if torch.cuda.is_available() else gs.cpu
        gs.init(backend=backend, precision="32", logging_level="warning", performance_mode=True)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / dt),
                camera_pos=(2.0, -2.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.3),
                camera_fov=50,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=dt, constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True, enable_joint_limit=True, iterations=100,
            ),
            show_viewer=not headless,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.base_init_pos  = torch.tensor(self.env_cfg["base_init_pos"],  device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=self.base_init_pos.cpu().numpy(), quat=self.base_init_quat.cpu().numpy()))

        self.scene.build(n_envs=n_envs)

        # ── Motor DOFs + gains ────────────────────────────────────────────
        self.motor_dofs = [self.robot.get_joint(n).dof_idx_local for n in MOTOR_JOINT_NAMES]
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.ACT_DIM, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.ACT_DIM, self.motor_dofs)
        self.default_dof_pos = torch.tensor(
            [DEFAULT_JOINT_ANGLES[n] for n in MOTOR_JOINT_NAMES],
            device=self.device, dtype=gs.tc_float)

        self.side_sign  = torch.tensor(LEG_SIDE_SIGN, device=self.device, dtype=gs.tc_float)
        self.trot_phase = torch.tensor(TROT_PHASE,    device=self.device, dtype=gs.tc_float)

        # ── Foot-contact source (force if available, else stance-phase proxy) ─
        self.contact_mode, self.foot_link_idx = "phase", None
        try:
            self.foot_link_idx = [self.robot.get_link(n).idx_local for n in FOOT_LINK_NAMES]
            _ = self.robot.get_links_net_contact_force()
            self.contact_mode = "force"
        except Exception as e:
            print(f"  [contacts] net-contact-force API unavailable ({e}); using phase proxy.")

        # ── IK self-test ──────────────────────────────────────────────────
        with torch.no_grad():
            qh0 = self.default_dof_pos[0::3].view(1, 4)
            qt0 = self.default_dof_pos[1::3].view(1, 4)
            qc0 = self.default_dof_pos[2::3].view(1, 4)
            fx, fy, fz = leg_fk(qh0, qt0, qc0, self.side_sign, L_HIP, L_THIGH, L_CALF)
            rh, rt, rc = leg_ik(fx, fy, fz, self.side_sign, L_HIP, L_THIGH, L_CALF)
            err = (torch.abs(rh - qh0) + torch.abs(rt - qt0) + torch.abs(rc - qc0)).max().item()
        print(f"\n{'='*58}")
        print(f"  Go2 CPG-RL environment")
        print(f"{'='*58}")
        print(f"  Envs          : {n_envs}")
        print(f"  Control dt    : {dt}s ({1/dt:.0f} Hz) | CPG {self.n_cpg_substeps}x ({1/self.cpg_dt:.0f} Hz)")
        print(f"  Obs / Act     : {OBS_DIM} / {ACT_DIM}")
        print(f"  Foot contacts : {self.contact_mode}")
        print(f"  IK self-test  : max round-trip {err:.2e} rad"
              f"  [{'OK' if err < 1e-4 else 'WARN -> check L_HIP/L_THIGH/L_CALF'}]")
        print(f"  Nominal stance: foot z = {fz.mean().item():.3f} m  (base ~{-fz.mean().item():.2f} m)")
        print(f"{'='*58}\n")

        # ── Reward registry ───────────────────────────────────────────────
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales:
            self.reward_functions[name] = getattr(self, f"_reward_{name}")
            self.episode_sums[name] = torch.zeros((n_envs,), device=self.device, dtype=gs.tc_float)

        # ── State buffers ─────────────────────────────────────────────────
        N, f = n_envs, gs.tc_float
        self.base_lin_vel      = torch.zeros((N, 3), device=self.device, dtype=f)
        self.base_ang_vel      = torch.zeros((N, 3), device=self.device, dtype=f)
        self.projected_gravity = torch.zeros((N, 3), device=self.device, dtype=f)
        self.global_gravity    = torch.tensor([0., 0., -1.], device=self.device, dtype=f).repeat(N, 1)
        self.base_pos          = torch.zeros((N, 3), device=self.device, dtype=f)
        self.base_quat         = torch.zeros((N, 4), device=self.device, dtype=f)
        self.base_euler        = torch.zeros((N, 3), device=self.device, dtype=f)

        self.dof_pos        = torch.zeros((N, ACT_DIM), device=self.device, dtype=f)
        self.dof_vel        = torch.zeros((N, ACT_DIM), device=self.device, dtype=f)
        self.last_dof_vel   = torch.zeros((N, ACT_DIM), device=self.device, dtype=f)
        self.target_dof_pos = self.default_dof_pos.unsqueeze(0).repeat(N, 1).clone()
        self.applied_torque = torch.zeros((N, ACT_DIM), device=self.device, dtype=f)

        self.actions      = torch.zeros((N, ACT_DIM), device=self.device, dtype=f)
        self.last_actions = torch.zeros((N, ACT_DIM), device=self.device, dtype=f)

        self.cpg_r     = torch.ones( (N, 4), device=self.device, dtype=f)
        self.cpg_rdot  = torch.zeros((N, 4), device=self.device, dtype=f)
        self.cpg_theta = torch.zeros((N, 4), device=self.device, dtype=f)
        self.cpg_phi   = torch.zeros((N, 4), device=self.device, dtype=f)
        self.foot_contacts = torch.zeros((N, 4), device=self.device, dtype=f)

        self.commands       = torch.zeros((N, self.num_commands), device=self.device, dtype=f)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device, dtype=f)

        self.obs_buf            = torch.zeros((N, OBS_DIM), device=self.device, dtype=f)
        self.rew_buf            = torch.zeros((N,),         device=self.device, dtype=f)
        self.reset_buf          = torch.ones( (N,),         device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((N,),         device=self.device, dtype=gs.tc_int)
        self.extras = {}

    # ------------------------------------------------------------------
    def _map_action(self, raw):
        t = torch.tanh(raw)                              # bounded; squash lives in env
        mu       = MU_MID  + MU_HALF  * t[:, 0:4]
        omega_hz = OMG_MID + OMG_HALF * t[:, 4:8]
        psi      = PSI_MAX * t[:, 8:12]
        return mu, omega_hz, psi

    # def _integrate_cpg(self, mu, omega_hz, psi):
    #     omega = 2.0 * math.pi * omega_hz
    #     a, dt_c = A_CONV, self.cpg_dt
    #     for _ in range(self.n_cpg_substeps):
    #         r_ddot = a * (0.25 * a * (mu - self.cpg_r) - self.cpg_rdot)
    #         self.cpg_rdot = self.cpg_rdot + r_ddot * dt_c
    #         self.cpg_r    = self.cpg_r + self.cpg_rdot * dt_c
    #         self.cpg_theta = self.cpg_theta + omega * dt_c
    #         self.cpg_phi   = self.cpg_phi + psi * dt_c
    #     self.cpg_theta = torch.remainder(self.cpg_theta, 2 * math.pi)
    #     self.cpg_phi   = torch.remainder(self.cpg_phi,   2 * math.pi)
    def _integrate_cpg(self, mu, omega_hz, psi):
        omega = 2.0 * math.pi * omega_hz
        a, dt_c = A_CONV, self.cpg_dt

        # Desired phase offsets for trot [FR, FL, RR, RL]
        # FR-RL in phase, FL-RR in phase, diagonal pairs π apart
        phi_star = torch.tensor([
            [0.0,   math.pi, math.pi, 0.0],   # FR target offsets
            [math.pi, 0.0,   0.0,   math.pi], # FL
            [math.pi, 0.0,   0.0,   math.pi], # RR
            [0.0,   math.pi, math.pi, 0.0],   # RL
        ], device=self.device)

        # Coupling weight
        w = 2.0   # paper uses w=1, higher = stronger coupling

        for _ in range(self.n_cpg_substeps):
            # Kuramoto coupling: pull each leg toward desired phase offset
            theta_i = self.cpg_theta.unsqueeze(2)   # [N, 4, 1]
            theta_j = self.cpg_theta.unsqueeze(1)   # [N, 1, 4]
            coupling = w * torch.sum(
                torch.sin(theta_j - theta_i - phi_star), dim=2
            )   # [N, 4]

            r_ddot = a * (0.25 * a * (mu - self.cpg_r) - self.cpg_rdot)
            self.cpg_rdot  = self.cpg_rdot + r_ddot * dt_c
            self.cpg_r     = self.cpg_r + self.cpg_rdot * dt_c
            self.cpg_theta = self.cpg_theta + (omega + coupling) * dt_c
            self.cpg_phi   = self.cpg_phi + psi * dt_c

        self.cpg_theta = torch.remainder(self.cpg_theta, 2 * math.pi)
        self.cpg_phi   = torch.remainder(self.cpg_phi,   2 * math.pi)

    def _cpg_to_joint_targets(self):
        amp = D_STEP * (self.cpg_r - 1.0)
        ct, st = torch.cos(self.cpg_theta), torch.sin(self.cpg_theta)
        cp, sp = torch.cos(self.cpg_phi),   torch.sin(self.cpg_phi)
        px = -amp * ct * cp
        py = self.side_sign * L_HIP - amp * ct * sp
        z_clear = torch.where(st > 0, G_CLEAR * st, G_PEN * st)
        pz = -H_NOM + z_clear
        qh, qt, qc = leg_ik(px, py, pz, self.side_sign, L_HIP, L_THIGH, L_CALF)
        self.target_dof_pos[:, 0::3] = qh
        self.target_dof_pos[:, 1::3] = qt
        self.target_dof_pos[:, 2::3] = qc

    def _update_foot_contacts(self):
        if self.contact_mode == "force":
            try:
                fall = self.robot.get_links_net_contact_force()
                ff = fall[:, self.foot_link_idx, :]
                self.foot_contacts = (torch.norm(ff, dim=-1) > CONTACT_FORCE_THRESH).float()
                return
            except Exception as e:
                print(f"  [contacts] force read failed ({e}); switching to phase proxy.")
                self.contact_mode = "phase"
        self.foot_contacts = (torch.sin(self.cpg_theta) < 0).float()

    def _compute_observation(self):
        self.obs_buf = torch.cat([
            self.base_lin_vel * self.obs_scales["lin_vel"],                       # 3
            self.base_ang_vel * self.obs_scales["ang_vel"],                       # 3
            self.projected_gravity,                                               # 3
            self.commands * self.commands_scale,                                  # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],   # 12
            self.dof_vel * self.obs_scales["dof_vel"],                            # 12
            self.last_actions,                                                    # 12
            self.foot_contacts,                                                   # 4
            self.cpg_r, self.cpg_rdot,                                            # 8
            torch.cos(self.cpg_theta), torch.sin(self.cpg_theta),                 # 8
            torch.cos(self.cpg_phi),   torch.sin(self.cpg_phi),                   # 8
        ], dim=-1)

    # ------------------------------------------------------------------
    def step(self, actions):
        self.actions = actions
        mu, omega_hz, psi = self._map_action(self.actions)
        self._integrate_cpg(mu, omega_hz, psi)
        self._cpg_to_joint_targets()
        self.robot.control_dofs_position(self.target_dof_pos, self.motor_dofs)
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos[:]  = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(transform_quat_by_quat(
            torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat))
        inv_q = inv_quat(self.base_quat)
        self.base_lin_vel[:]      = transform_by_quat(self.robot.get_vel(), inv_q)
        self.base_ang_vel[:]      = transform_by_quat(self.robot.get_ang(), inv_q)
        self.projected_gravity[:] = transform_by_quat(self.global_gravity, inv_q)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.applied_torque = (self.env_cfg["kp"] * (self.target_dof_pos - self.dof_pos)
                               - self.env_cfg["kd"] * self.dof_vel)
        self._update_foot_contacts()

        resample_every = int(self.env_cfg["resampling_time_s"] / self.dt)
        envs_idx = (self.episode_length_buf % resample_every == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(envs_idx)

        self.reset_buf  = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg["termination_if_height_less_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        self.rew_buf[:] = 0.0
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        self.rew_buf = torch.clamp(self.rew_buf, -10.0, 10.0)   # safety against spikes

        self._compute_observation()
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        self.base_pos[:]  = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        inv_q = inv_quat(self.base_quat)
        self.base_lin_vel[:]      = transform_by_quat(self.robot.get_vel(), inv_q)
        self.base_ang_vel[:]      = transform_by_quat(self.robot.get_ang(), inv_q)
        self.projected_gravity[:] = transform_by_quat(self.global_gravity, inv_q)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self._update_foot_contacts()
        self._compute_observation()
        assert self.obs_buf.shape[-1] == OBS_DIM, \
            f"obs dim {self.obs_buf.shape[-1]} != OBS_DIM {OBS_DIM}"
        return self.obs_buf, None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(position=self.dof_pos[envs_idx], dofs_idx_local=self.motor_dofs,
                                     zero_velocity=True, envs_idx=envs_idx)
        self.base_pos[envs_idx]  = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx],  zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.cpg_r[envs_idx]     = 1.0
        self.cpg_rdot[envs_idx]  = 0.0
        self.cpg_theta[envs_idx] = self.trot_phase
        self.cpg_phi[envs_idx]   = self.trot_phase * 0.1

        self.last_actions[envs_idx]       = 0.0
        self.last_dof_vel[envs_idx]       = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx]          = True

        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"])
            self.episode_sums[key][envs_idx] = 0.0
        self._resample_commands(envs_idx)

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (n,), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (n,), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"],   (n,), self.device)

    # ── Reward terms (paper Sec III-C) ─────────────────────────────────
    def _reward_tracking_lin_vel_x(self):
        return torch.exp(-torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]) / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        return torch.exp(-torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1]) / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        return torch.exp(-torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2]) / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_work(self):
        dqd = self.dof_vel - self.last_dof_vel
        return torch.abs(torch.sum(self.applied_torque * dqd, dim=1))


# ==========================================================================
#  ActorCritic
# ==========================================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden,  hidden), nn.ELU(),
        )
        self.actor_head  = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, act_dim))
        self.critic_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.00)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        return mean, self.log_std.exp().expand_as(mean), value

    def get_action(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        return action, dist.log_prob(action).sum(-1), value

    def get_value(self, obs):
        return self.critic_head(self.trunk(obs)).squeeze(-1)

    def evaluate(self, obs, action):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ==========================================================================
#  RolloutBuffer
# ==========================================================================

class RolloutBuffer:
    def __init__(self, rollout_steps, n_envs, obs_dim, act_dim, device):
        self.T, self.N, self.device, self.ptr = rollout_steps, n_envs, device, 0
        b = lambda *s: torch.zeros(*s, device=device)
        self.obs        = b(rollout_steps, n_envs, obs_dim)
        self.actions    = b(rollout_steps, n_envs, act_dim)
        self.log_probs  = b(rollout_steps, n_envs)
        self.values     = b(rollout_steps, n_envs)
        self.rewards    = b(rollout_steps, n_envs)
        self.dones      = b(rollout_steps, n_envs)
        self.advantages = b(rollout_steps, n_envs)
        self.returns    = b(rollout_steps, n_envs)

    def store_step(self, obs, actions, log_probs, values):
        t = self.ptr
        self.obs[t], self.actions[t] = obs.detach(), actions.detach()
        self.log_probs[t], self.values[t] = log_probs.detach(), values.detach()

    def store_outcome(self, rewards, dones):
        t = self.ptr
        self.rewards[t], self.dones[t] = rewards.detach().float(), dones.detach().float()
        self.ptr += 1

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            next_val = last_value if t == self.T - 1 else self.values[t + 1]
            mask  = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * mask - self.values[t]
            gae   = delta + gamma * lam * mask * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values
        self.ptr = 0

    def get_flat(self):
        T, N = self.T, self.N
        return (self.obs.view(T*N, -1), self.actions.view(T*N, -1), self.log_probs.view(T*N),
                self.advantages.view(T*N), self.returns.view(T*N), self.values.view(T*N))


# ==========================================================================
#  PPO Trainer
# ==========================================================================

class PPOTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["device"]
        self.env = Go2CPGEnv(n_envs=cfg["n_envs"], dt=cfg["dt"],
                             max_episode_steps=cfg["max_episode_steps"],
                             headless=cfg["headless"], device=cfg["device"])
        self.net = ActorCritic(OBS_DIM, ACT_DIM, cfg["hidden_size"]).to(self.device)
        print(f"  Network params: {self.net.num_parameters:,}\n")
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"], eps=1e-5)
        self.buf = RolloutBuffer(cfg["rollout_steps"], cfg["n_envs"], OBS_DIM, ACT_DIM, self.device)

        os.makedirs(cfg["run_dir"], exist_ok=True)
        self.writer = SummaryWriter(cfg["run_dir"])
        self.global_step = 0
        self.start_time = time.time()
        self.ep_returns, self.ep_lengths = [], []
        self._env_ret = torch.zeros(cfg["n_envs"], device=self.device)
        self._env_len = torch.zeros(cfg["n_envs"], device=self.device, dtype=torch.int32)

    def train(self):
        cfg = self.cfg
        obs, _ = self.env.reset()
        print(f"{'='*58}")
        print(f"  CPG-RL PPO training | total {cfg['total_steps']:,} steps | {self.device}")
        print(f"{'='*58}\n")
        steps_per_rollout = cfg["rollout_steps"] * cfg["n_envs"]
        n_updates = cfg["total_steps"] // steps_per_rollout

        for update in range(1, n_updates + 1):
            # Linear LR decay to a floor: stops the converged policy from
            # chasing normalized-noise advantages and destabilizing late on.
            frac = max(1.0 - self.global_step / max(cfg["total_steps"], 1), cfg["lr_floor_frac"])
            lr = cfg["lr"] * frac
            for g in self.opt.param_groups:
                g["lr"] = lr

            obs = self._collect_rollout(obs)
            m = self._ppo_update()
            self.global_step += steps_per_rollout

            if update % cfg["log_interval"] == 0:
                sps = self.global_step / (time.time() - self.start_time)
                mret = float(np.mean(self.ep_returns[-50:])) if self.ep_returns else 0.0
                mlen = float(np.mean(self.ep_lengths[-50:])) if self.ep_lengths else 0.0
                print(f"  step {self.global_step:>11,} | ret {mret:>7.3f} | len {mlen:>5.0f} | "
                      f"ploss {m['policy_loss']:>7.4f} | vloss {m['value_loss']:>8.3f} | "
                      f"kl {m['approx_kl']:>6.4f} | clip {m['clip_frac']:>4.2f} | lr {lr:.1e} | {sps:>7,.0f} sps")
                for k, v in {"train/mean_return": mret, "train/mean_ep_len": mlen,
                             "loss/policy": m["policy_loss"], "loss/value": m["value_loss"],
                             "loss/entropy": m["entropy"], "train/approx_kl": m["approx_kl"],
                             "train/clip_fraction": m["clip_frac"], "train/lr": lr,
                             "train/sps": sps}.items():
                    self.writer.add_scalar(k, v, self.global_step)

            if update % cfg["save_interval"] == 0:
                self._save_checkpoint()

        self._save_checkpoint(tag="final")
        self.writer.close()
        print("\nTraining complete.")

    def _collect_rollout(self, obs):
        self.net.eval()
        with torch.no_grad():
            for _ in range(self.cfg["rollout_steps"]):
                action, log_prob, value = self.net.get_action(obs)
                self.buf.store_step(obs, action, log_prob, value)
                next_obs, _, reward, reset_buf, _ = self.env.step(action)
                self.buf.store_outcome(reward, reset_buf.float())
                self._env_ret += reward
                self._env_len += 1
                done_idx = reset_buf.nonzero(as_tuple=False).flatten()
                for idx in done_idx:
                    self.ep_returns.append(float(self._env_ret[idx]))
                    self.ep_lengths.append(int(self._env_len[idx]))
                self._env_ret[done_idx] = 0.0
                self._env_len[done_idx] = 0
                obs = next_obs
            last_value = self.net.get_value(obs)
            self.buf.compute_gae(last_value, self.cfg["gamma"], self.cfg["lam"])
        return obs

    def _ppo_update(self):
        self.net.train()
        cfg = self.cfg
        obs_f, act_f, lp_f, adv_f, ret_f, val_f = self.buf.get_flat()

        # Skip entirely if the rollout produced non-finite stats.
        if not (torch.isfinite(adv_f).all() and torch.isfinite(ret_f).all()):
            print("  [warn] non-finite rollout stats; skipping update.")
            return {"policy_loss": 0., "value_loss": 0., "entropy": 0., "clip_frac": 0., "approx_kl": 0.}

        backup = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        adv_f = ((adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)).clamp(-10.0, 10.0)
        total = obs_f.shape[0]
        target_kl = cfg.get("target_kl", 0.02)
        vloss_cap = cfg.get("vloss_skip", 1e3)
        m = {"policy_loss": [], "value_loss": [], "entropy": [], "clip_frac": [], "approx_kl": []}
        bad = False

        for _ in range(cfg["n_epochs"]):
            idx = torch.randperm(total, device=self.device)
            epoch_kl = []
            for start in range(0, total, cfg["minibatch_size"]):
                mb = idx[start:start + cfg["minibatch_size"]]
                new_lp, entropy, value = self.net.evaluate(obs_f[mb], act_f[mb])
                logratio = new_lp - lp_f[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                epoch_kl.append(approx_kl); m["approx_kl"].append(approx_kl)

                surr1 = ratio * adv_f[mb]
                surr2 = ratio.clamp(1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv_f[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                vclip = val_f[mb] + (value - val_f[mb]).clamp(-cfg["clip_eps"], cfg["clip_eps"])
                value_loss = torch.max((value - ret_f[mb]).pow(2), (vclip - ret_f[mb]).pow(2)).mean()
                loss = policy_loss + cfg["vf_coef"] * value_loss - cfg["ent_coef"] * entropy.mean()

                if (not torch.isfinite(loss)) or (value_loss.item() > vloss_cap):
                    print(f"  [warn] pathological minibatch (vloss={value_loss.item():.2e}); rolling back, skipping update.")
                    bad = True; break
                self.opt.zero_grad(); loss.backward()
                gnorm = nn.utils.clip_grad_norm_(self.net.parameters(), cfg["max_grad_norm"])
                if not torch.isfinite(gnorm):
                    print("  [warn] non-finite gradient; rolling back, skipping update.")
                    bad = True; break
                self.opt.step()
                m["policy_loss"].append(policy_loss.item())
                m["value_loss"].append(value_loss.item())
                m["entropy"].append(entropy.mean().item())
                m["clip_frac"].append(((ratio - 1.0).abs() > cfg["clip_eps"]).float().mean().item())

            if bad:
                break
            if np.mean(epoch_kl) > 1.5 * target_kl:      # KL early stop
                break

        if bad:
            self.net.load_state_dict(backup)             # atomic rollback
            self.opt.zero_grad(set_to_none=True)
        return {k: float(np.mean(v)) if v else 0.0 for k, v in m.items()}

    def _save_checkpoint(self, tag=None):
        name = f"checkpoint_{tag}" if tag else f"checkpoint_step_{self.global_step:09d}"
        path = os.path.join(self.cfg["run_dir"], f"{name}.pt")
        torch.save({
            "step": self.global_step,
            "model_state": clean_state_dict(self.net.state_dict()),   # always clean keys
            "optim_state": self.opt.state_dict(),
            "config": self.cfg, "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
            "metrics": {
                "mean_return": np.mean(self.ep_returns[-50:]) if self.ep_returns else 0.0,
                "mean_length": np.mean(self.ep_lengths[-50:]) if self.ep_lengths else 0.0,
            },
        }, path)
        print(f"  [ckpt] {path}")

    @classmethod
    def load_checkpoint(cls, path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, weights_only=False, map_location=device)
        trainer = cls(ckpt["config"])
        trainer.net.load_state_dict(clean_state_dict(ckpt["model_state"]))
        trainer.opt.load_state_dict(ckpt["optim_state"])
        trainer.global_step = ckpt["step"]
        print(f"  Resumed step={trainer.global_step:,} ret={ckpt['metrics']['mean_return']:.3f}")
        return trainer


# ==========================================================================
#  Evaluation  (single env, viewer, fixed forward command for a clean demo)
# ==========================================================================

def evaluate(checkpoint_path, n_episodes=3, command=(0.5, 0.0, 0.0), headless=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    sd = clean_state_dict(ckpt["model_state"])           # tolerate _orig_mod./module. prefixes

    # Architecture-agnostic dim inference: log_std is always [act_dim],
    # trunk.0.weight is always [hidden, obs_dim].
    obs_dim = sd["trunk.0.weight"].shape[1]
    act_dim = sd["log_std"].shape[0]
    print(f"  Checkpoint: obs_dim={obs_dim} act_dim={act_dim}")
    if obs_dim != OBS_DIM:
        print(f"  [warn] checkpoint obs_dim={obs_dim} != this script's OBS_DIM={OBS_DIM}.\n"
              f"        It was trained on a different observation layout and will not run here.\n"
              f"        Retrain with this script (CPG-RL converges in tens of millions of steps).")
        return

    env = Go2CPGEnv(n_envs=1, headless=headless, max_episode_steps=cfg["max_episode_steps"],
                    dt=cfg["dt"], device=device)
    try:
        net = ActorCritic(obs_dim, act_dim, cfg.get("hidden_size", 512))
    except TypeError:
        net = ActorCritic(obs_dim, act_dim)
    net.load_state_dict(sd)
    net.eval().to(device)

    cmd = torch.tensor([command], device=device, dtype=gs.tc_float)
    for ep in range(n_episodes):
        obs, _ = env.reset()
        env.commands[:] = cmd
        done = torch.zeros(1, dtype=torch.bool)
        ep_ret, ep_len, vx_err, theta_hist = 0.0, 0, [], []
        while not done[0]:
            env.commands[:] = cmd
            with torch.no_grad():
                act, _, _ = net.get_action(obs, deterministic=True)
            obs, _, reward, reset_buf, _ = env.step(act)
            ep_ret += reward[0].item(); ep_len += 1
            done = reset_buf.bool()
            vx_err.append(abs(command[0] - env.base_lin_vel[0, 0].item()))
            theta_hist.append(env.cpg_theta[0].cpu().numpy().copy())
            if ep_len % 50 == 0:
                r = env.cpg_r[0].cpu().numpy()
                print(f"    step {ep_len:4d}  vx={env.base_lin_vel[0,0].item():+.2f}/{command[0]:.2f}  "
                      f"vy={env.base_lin_vel[0,1].item():+.2f}  wz={env.base_ang_vel[0,2].item():+.2f}  "
                      f"h={env.base_pos[0,2].item():.2f}  r=[{r[0]:.2f} {r[1]:.2f} {r[2]:.2f} {r[3]:.2f}]")
        th = np.array(theta_hist)[:, 0]
        wraps = int(np.sum(np.diff(th) < -math.pi))
        period = (ep_len * env.dt / wraps) if wraps > 0 else float('nan')
        print(f"  Episode {ep+1} | return={ep_ret:7.2f} | length={ep_len:4d} | "
              f"mean |vx err|={np.mean(vx_err):.3f} m/s | gait period~{period:.2f}s\n")


# ==========================================================================
#  Entry point
# ==========================================================================

def get_config(args):
    total_buffer = args.n_envs * args.rollout_steps
    return dict(
        n_envs=args.n_envs, dt=0.02, max_episode_steps=1000, headless=args.headless,
        hidden_size=512, total_steps=args.total_steps, rollout_steps=args.rollout_steps,
        minibatch_size=max(total_buffer // 4, 256), n_epochs=5,
        gamma=0.99, lam=0.95, clip_eps=0.2, lr=3e-4, vf_coef=1.0, ent_coef=0.01,
        max_grad_norm=1.0, device=args.device, run_dir=args.run_dir,
        log_interval=10, save_interval=100,
        # stability guards
        target_kl=0.02, vloss_skip=1e3, lr_floor_frac=0.05,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs",        type=int, default=4096)
    p.add_argument("--total-steps",   type=int, default=80_000_000)
    p.add_argument("--rollout-steps", type=int, default=24)
    p.add_argument("--device",        type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--run-dir",       type=str, default="../../runs/go2_cpg")
    p.add_argument("--headless",      action="store_true", default=True)
    p.add_argument("--resume",        type=str, default=None)
    p.add_argument("--eval",          type=str, default=None)
    p.add_argument("--vx",   type=float, default=0.5,
               help="Forward velocity command for eval (m/s)")
    p.add_argument("--vy",   type=float, default=0.0,
                help="Lateral velocity command for eval (m/s)")
    p.add_argument("--vyaw", type=float, default=0.0,
                help="Yaw rate command for eval (rad/s)")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.eval:
        evaluate(args.eval, command=(args.vx, args.vy, args.vyaw))
        return
    trainer = PPOTrainer.load_checkpoint(args.resume) if args.resume else PPOTrainer(get_config(args))
    trainer.train()


if __name__ == "__main__":
    main()
