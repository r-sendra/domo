"""
go2_lidar_nav.py
================
Self-contained RL experiment: Go2 locomotion with LiDAR-based
obstacle avoidance on flat terrain with random obstacles.

Curriculum learning (2 phases):
  Phase 1 — Walk (0 → 10M steps):
    No avoidance reward. Robot reinforces walking with pretrained weights.

  Phase 2 — Avoidance (10M → end):
    Full avoidance penalty enabled. LiDAR rays are horizontal-only,
    positioned above the chassis so they detect room walls and furniture
    rather than the robot's own body.

LiDAR setup:
  - 128 horizontal rays at 0° elevation (flat ring)
  - pos_offset z=0.35m (above chassis, clears legs)
  - return_world_frame=True (stable sector ordering as robot rotates)
  - 128 rays → 36 sector minimums → 81-dim obs (45 prop + 36 LiDAR)

Usage:
    # Train in house (recommended)
    python go2_lidar_nav.py --n-envs 512 --device cuda --headless \\
        --scene data/replica_cad/configs/scenes/apt_0.scene_instance.json \\
        --asset-root data/replica_cad/ \\
        --pretrained ../../runs/go2_walk/checkpoint_final.pt

    # Evaluate on Mac
    python go2_lidar_nav.py --eval ../../runs/go2_lidar/checkpoint_final.pt

    # Resume
    python go2_lidar_nav.py --resume ../../runs/go2_lidar/checkpoint_step_010000000.pt
"""

import os
import json
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

MOTOR_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

DEFAULT_JOINT_ANGLES = {
    "FR_hip_joint":   0.0, "FR_thigh_joint":  0.8, "FR_calf_joint": -1.5,
    "FL_hip_joint":   0.0, "FL_thigh_joint":  0.8, "FL_calf_joint": -1.5,
    "RR_hip_joint":   0.0, "RR_thigh_joint":  1.0, "RR_calf_joint": -1.5,
    "RL_hip_joint":   0.0, "RL_thigh_joint":  1.0, "RL_calf_joint": -1.5,
}

BASE_INIT_POS  = [3.0, -2.0, 0.42]
BASE_INIT_QUAT = [0.0, 0.0, 0.0, 1.0]   # [x,y,z,w] identity

# LiDAR — default SphericalPattern (n_points=(64,128) = 8192 rays on GPU)
# On CPU (Mac) this is slow; training on H200 GPU is fast
LIDAR_MAX_RANGE    = 5.0    # metres
LIDAR_DANGER_ZONE  = 0.8    # metres — penalise if closer than this
LIDAR_CAUTION_ZONE = 1.5    # metres — smaller penalty

# Sector aggregation: 8192 rays → 36 sector minimums
# Each sector covers 360°/36 = 10° horizontal slice
# Min distance per sector is the relevant signal for avoidance
N_SECTORS = 36
PROP_DIM  = 45               # proprioception dims (unchanged from walking env)
OBS_DIM   = PROP_DIM + N_SECTORS   # 45 + 36 = 81

# Obstacle configuration
N_OBSTACLES        = 8      # cylinders per environment
OBSTACLE_RING_MIN  = 1.5    # metres from spawn — inner exclusion (increased from 1.0)
OBSTACLE_RING_MAX  = 4.0    # metres from spawn — outer boundary
OBSTACLE_HEIGHT    = 1.2    # metres
OBSTACLE_RADIUS    = 0.25   # metres

# Curriculum phases (in environment steps)
# Phase 1: walk only — no avoidance reward (robot reinforces walking)
# Phase 2: full avoidance reward enabled
CURRICULUM_PHASE1_STEPS = 10_000_000   # 0 → 10M: walk only (short warm-up)
# Phase 2: 10M → end: full avoidance reward


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


# ==========================================================================
#  ReplicaCAD scene loader (same as replica_rl_locomotion.py)
# ==========================================================================

def _convert_habitat_to_genesis(pos, quat_wxyz):
    x, y, z = pos
    qw, qx, qy, qz = quat_wxyz
    return [x, -z, y], [qw, qx, -qz, qy]


def _resolve_asset_path(template_path, root_dir):
    base_name  = os.path.basename(template_path)
    search_dir = os.path.join(root_dir, "configs")
    if not os.path.exists(search_dir):
        search_dir = root_dir
    for subdir, _, files in os.walk(search_dir):
        for f in files:
            if f.startswith(base_name) and f.endswith(".json"):
                path = os.path.join(subdir, f)
                try:
                    with open(path) as fp:
                        cfg = json.load(fp)
                    asset = cfg.get("urdf_filepath") or cfg.get("render_asset")
                    if asset:
                        abs_p = os.path.abspath(
                            os.path.join(os.path.dirname(path), asset)
                        )
                        if os.path.exists(abs_p):
                            return abs_p
                except Exception:
                    pass
    return None


def _spawn_entity(scene, template_name, asset_path, pos, quat, is_fixed, scale=1.0):
    try:
        if asset_path.endswith(".urdf"):
            scene.add_entity(gs.morphs.URDF(
                file=asset_path, pos=pos, quat=quat, fixed=is_fixed
            ))
        elif asset_path.endswith((".glb", ".obj")):
            scene.add_entity(gs.morphs.Mesh(
                file=asset_path, pos=pos, quat=quat, fixed=is_fixed,
                scale=(scale, scale, scale)
            ))
        print(f"  ✅  {os.path.basename(template_name)}")
    except Exception as e:
        print(f"  ❌  {os.path.basename(template_name)}: {e}")


def _load_replica_scene(scene, scene_json, asset_root):
    """Load a ReplicaCAD scene into a Genesis scene object."""
    import json as _json
    with open(scene_json) as f:
        config = _json.load(f)

    print("\n── Stage ──")
    stage = config.get("stage_instance", {})
    tmpl  = stage.get("template_name")
    if tmpl:
        asset = _resolve_asset_path(tmpl, asset_root)
        if asset:
            p, q = _convert_habitat_to_genesis(
                stage.get("translation", [0, 0, 0]),
                stage.get("rotation",    [1, 0, 0, 0]),
            )
            p[2] -= 0.05   # nudge to prevent z-fighting
            _spawn_entity(scene, tmpl, asset, p, q, True)
        else:
            print(f"  ⚠️  Could not resolve stage: {tmpl}")

    print("\n── Rigid objects ──")
    for obj in config.get("object_instances", []):
        tmpl  = obj["template_name"]
        asset = _resolve_asset_path(tmpl, asset_root)
        if asset:
            p, q = _convert_habitat_to_genesis(obj["translation"], obj["rotation"])
            _spawn_entity(scene, tmpl, asset, p, q,
                          obj.get("motion_type") == "STATIC",
                          obj.get("uniform_scale", 1.0))

    print("\n── Articulated objects ──")
    for obj in config.get("articulated_object_instances", []):
        tmpl  = obj["template_name"]
        asset = _resolve_asset_path(tmpl, asset_root)
        if asset:
            p, q = _convert_habitat_to_genesis(obj["translation"], obj["rotation"])
            _spawn_entity(scene, tmpl, asset, p, q,
                          obj.get("fixed_base", True),
                          obj.get("uniform_scale", 1.0))


# ==========================================================================
#  Environment
# ==========================================================================

class Go2LidarNavEnv:
    """
    Go2 locomotion + LiDAR obstacle avoidance environment.

    Observation: [proprioception (45)] + [LiDAR distances (N_lidar)]
    Action:      joint position offsets (12)
    Reward:      velocity tracking + collision avoidance penalty
    """

    ACT_DIM = 12

    def __init__(
        self,
        n_envs:            int   = 4096,
        dt:                float = 0.02,
        max_episode_steps: int   = 1000,
        headless:          bool  = True,
        device:            str   = "cuda",
        scene_json:        str   = None,    # path to ReplicaCAD scene JSON
        asset_root:        str   = None,    # path to ReplicaCAD root dir
    ):
        self.n_envs            = n_envs
        self.num_envs          = n_envs
        self.dt                = dt
        self.max_episode_steps = max_episode_steps
        self.device            = torch.device(device)
        self.scene_json        = scene_json
        self.asset_root        = os.path.abspath(asset_root) if asset_root else None
        self.use_house         = scene_json is not None

        self.simulate_action_latency = True

        # ── Config ────────────────────────────────────────────────────────
        self.env_cfg = {
            "num_actions":   12,
            "base_init_pos":  BASE_INIT_POS,
            "base_init_quat": BASE_INIT_QUAT,
            "episode_length_s": max_episode_steps * dt,
            "resampling_time_s": 4.0,
            "action_scale":  0.25,
            "clip_actions":  100.0,
            "kp": 20.0,
            "kd":  0.5,
            "dof_names": MOTOR_JOINT_NAMES,
            "default_joint_angles": DEFAULT_JOINT_ANGLES,
            "termination_if_pitch_greater_than": 1.0,
            "termination_if_roll_greater_than":  1.0,
        }
        self.obs_cfg = {
            "obs_scales": {
                "lin_vel": 2.0, "ang_vel": 0.25,
                "dof_pos": 1.0, "dof_vel": 0.05,
            },
        }
        self.reward_cfg = {
            "tracking_sigma":     0.25,
            "base_height_target": 0.34,
            "reward_scales": {
                "tracking_lin_vel":   1.0,
                "tracking_ang_vel":   0.2,
                "lin_vel_z":         -1.0,
                "base_height":       -50.0,
                "action_rate":       -0.005,
                "similar_to_default":-0.1,
                "obstacle_avoidance":-2.0,   # NEW: collision avoidance
            },
        }
        self.command_cfg = {
            "num_commands": 3,
            "lin_vel_x_range": [0.5, 0.5],
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range":   [0.0, 0.0],
        }

        self.obs_scales     = self.obs_cfg["obs_scales"]
        self.reward_scales  = {
            k: v * dt
            for k, v in self.reward_cfg["reward_scales"].items()
        }
        self.max_episode_length = math.ceil(
            self.env_cfg["episode_length_s"] / self.dt
        )
        self.num_commands = self.command_cfg["num_commands"]

        # Curriculum state — controlled by PPOTrainer
        self.curriculum_phase    = 1   # 1=walk only, 2=obstacles, 3=full avoidance
        self.obstacles_active    = False
        self.avoidance_active    = False

        # ── Genesis init ──────────────────────────────────────────────────
        backend = gs.cuda if torch.cuda.is_available() else gs.cpu
        gs.init(
            backend      = backend,
            precision    = "32",
            logging_level= "warning",
            performance_mode = True,
        )

        self.scene = gs.Scene(
            sim_options    = gs.options.SimOptions(dt=0.01, substeps=2),
            viewer_options = gs.options.ViewerOptions(
                max_FPS      = 60, 
                camera_pos   = (BASE_INIT_POS[0] + 2.0,
                                BASE_INIT_POS[1] - 2.0, 1.5),
                camera_lookat= (BASE_INIT_POS[0],
                                BASE_INIT_POS[1], 0.4),
                camera_fov   = 50,
            ),
            vis_options    = gs.options.VisOptions(n_rendered_envs=1),
            rigid_options  = gs.options.RigidOptions(
                dt                = self.dt,
                constraint_solver = gs.constraint_solver.Newton,
                enable_collision  = True,
                enable_joint_limit= True
            ),
            show_viewer = not headless,
        )

        # ── Scene: house or flat terrain + cylinders ─────────────────────
        if self.use_house:
            print(f"\nLoading house scene: {self.scene_json}")
            _load_replica_scene(self.scene, self.scene_json, self.asset_root)
            # In house mode obstacles are the room itself — no cylinders needed
            self.obstacles = []
            print()
        else:
            # Flat terrain
            self.scene.add_entity(
                gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
            )
            # Random cylinders — parked off-scene until Phase 2
            self.obstacles = []
            for _ in range(N_OBSTACLES):
                obs_entity = self.scene.add_entity(
                    gs.morphs.Cylinder(
                        height = OBSTACLE_HEIGHT,
                        radius = OBSTACLE_RADIUS,
                        pos    = (999.0, 999.0, OBSTACLE_HEIGHT / 2),
                        fixed  = True,
                    )
                )
                self.obstacles.append(obs_entity)

        # ── Robot ─────────────────────────────────────────────────────────
        self.base_init_pos  = torch.tensor(BASE_INIT_POS,  device=self.device)
        self.base_init_quat = torch.tensor(BASE_INIT_QUAT, device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file = "urdf/go2/urdf/go2.urdf",
                pos  = self.base_init_pos.cpu().numpy(),
                quat = self.base_init_quat.cpu().numpy(),
            )
        )

        # ── LiDAR sensor ──────────────────────────────────────────────────
        # Horizontal-only rays at robot body height to avoid hitting own legs.
        # pos_offset z=0.35 clears the robot chassis.
        # return_world_frame=True so sector ordering is stable as robot rotates.
        self.lidar = self.scene.add_sensor(
            gs.sensors.Lidar(
                pattern = gs.sensors.SphericalPattern(
                    fov      = (360.0, 0.0),   # flat horizontal — avoids hitting legs
                    n_points = (128, 1),        # 128 rays × 1 elevation = 128 rays
                ),
                entity_idx         = self.robot.idx,
                pos_offset         = (0.0, 0.0, 0.35),   # above chassis
                return_world_frame = True,                # stable sector ordering
                draw_debug         = (not headless),
            )
        )

        # ── Build ─────────────────────────────────────────────────────────
        self.scene.build(n_envs=n_envs)

        # Determine LiDAR output dimension from a dummy read after build
        # SphericalPattern default: 64 horizontal × 128 vertical = 8192
        _dummy           = self.lidar.read()
        _dummy_flat      = _dummy.distances.reshape(_dummy.distances.shape[0], -1)
        self.n_lidar_raw     = _dummy_flat.shape[-1]
        self.rays_per_sector = self.n_lidar_raw // N_SECTORS
        self.n_trim          = self.rays_per_sector * N_SECTORS
        self.OBS_DIM         = OBS_DIM   # 45 + 36 = 81
        print(f"  LiDAR raw rays : {self.n_lidar_raw}")
        print(f"  Rays/sector    : {self.rays_per_sector}  (trimmed to {self.n_trim})")
        print(f"  LiDAR sectors  : {N_SECTORS}  (min per sector)")
        print(f"  Total obs dim  : {self.OBS_DIM}")

        # ── Motor DOFs ────────────────────────────────────────────────────
        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local
            for name in self.env_cfg["dof_names"]
        ]
        self.robot.set_dofs_kp(
            [self.env_cfg["kp"]] * self.ACT_DIM, self.motor_dofs
        )
        self.robot.set_dofs_kv(
            [self.env_cfg["kd"]] * self.ACT_DIM, self.motor_dofs
        )

        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][n]
             for n in self.env_cfg["dof_names"]],
            device=self.device, dtype=gs.tc_float,
        )

        # ── Reward registry ───────────────────────────────────────────────
        self.reward_functions = {}
        self.episode_sums     = {}
        for name in self.reward_scales:
            self.reward_functions[name] = getattr(self, f"_reward_{name}")
            self.episode_sums[name]     = torch.zeros(
                (n_envs,), device=self.device, dtype=gs.tc_float
            )

        # ── State buffers ─────────────────────────────────────────────────
        N, f = n_envs, gs.tc_float
        self.base_lin_vel      = torch.zeros((N, 3), device=self.device, dtype=f)
        self.base_ang_vel      = torch.zeros((N, 3), device=self.device, dtype=f)
        self.projected_gravity = torch.zeros((N, 3), device=self.device, dtype=f)
        self.global_gravity    = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device, dtype=f
        ).repeat(N, 1)

        self.obs_buf            = torch.zeros((N, self.OBS_DIM), device=self.device, dtype=f)
        self.rew_buf            = torch.zeros((N,),              device=self.device, dtype=f)
        self.reset_buf          = torch.ones( (N,),              device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((N,),              device=self.device, dtype=gs.tc_int)

        self.commands       = torch.zeros((N, self.num_commands), device=self.device, dtype=f)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"],
             self.obs_scales["lin_vel"],
             self.obs_scales["ang_vel"]],
            device=self.device, dtype=f,
        )

        self.actions      = torch.zeros((N, self.ACT_DIM), device=self.device, dtype=f)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos      = torch.zeros_like(self.actions)
        self.dof_vel      = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos     = torch.zeros((N, 3), device=self.device, dtype=f)
        self.base_quat    = torch.zeros((N, 4), device=self.device, dtype=f)

        # LiDAR sector buffer — shape [N, N_SECTORS]
        # Aggregated from raw rays: min distance per 36 horizontal sectors
        self.lidar_sectors = torch.zeros(
            (N, N_SECTORS), device=self.device, dtype=f
        )

        self.extras = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, actions):
        self.actions = torch.clip(
            actions,
            -self.env_cfg["clip_actions"],
             self.env_cfg["clip_actions"],
        )

        exec_actions   = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # ── Update state ──────────────────────────────────────────────────
        self.episode_length_buf += 1
        self.base_pos[:]  = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler   = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            )
        )
        inv_base_quat              = inv_quat(self.base_quat)
        self.base_lin_vel[:]       = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:]       = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity[:]  = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:]            = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:]            = self.robot.get_dofs_velocity(self.motor_dofs)

        # ── LiDAR read + sector aggregation ──────────────────────────────
        # scene.step() already computed all rays — this is just a memory read
        lidar_data = self.lidar.read()
        raw_       = lidar_data.distances   # [N, channels, horizontal] or [N, n_rays]

        # One-time debug print to show actual runtime shape
        if not hasattr(self, '_lidar_shape_printed'):
            print(f"  [LiDAR runtime] distances.shape = {raw_.shape}")
            self._lidar_shape_printed = True

        # Flatten to [N, total_rays] regardless of whether Genesis returns
        # a 2D [N, rays] or 3D [N, channels, horizontal] tensor
        raw = raw_.reshape(raw_.shape[0], -1)   # [N, total_rays]

        # Aggregate raw rays into N_SECTORS sector minimums.
        # Handles any ray count gracefully — no assumption about divisibility.
        batch, n_raw = raw.shape
        if n_raw >= N_SECTORS:
            # Enough rays — reshape into N_SECTORS groups and take min
            rays_ps = n_raw // N_SECTORS
            n_trim  = rays_ps * N_SECTORS
            self.lidar_sectors[:] = raw[
                :, :n_trim
            ].view(batch, N_SECTORS, rays_ps).min(dim=2).values
        else:
            # Fewer rays than sectors — repeat-pad to fill sectors
            # Each sector gets the same small set of rays
            padded = raw.repeat(1, math.ceil(N_SECTORS / n_raw))[:, :N_SECTORS]
            self.lidar_sectors[:] = padded

        # ── Resample commands ─────────────────────────────────────────────
        resample_every = int(self.env_cfg["resampling_time_s"] / self.dt)
        envs_idx = (
            (self.episode_length_buf % resample_every == 0)
            .nonzero(as_tuple=False).flatten()
        )
        self._resample_commands(envs_idx)

        # ── Termination ───────────────────────────────────────────────────
        self.reset_buf  = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False).flatten()
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=self.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # ── Reward ────────────────────────────────────────────────────────
        self.rew_buf[:] = 0.0
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # ── Observation ───────────────────────────────────────────────────
        # Proprioception (45) + sector minimums normalised to [0,1] (36)
        lidar_norm = torch.clamp(
            self.lidar_sectors / LIDAR_MAX_RANGE, 0.0, 1.0
        )
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],                  # 3
                self.projected_gravity,                                            # 3
                self.commands * self.commands_scale,                               # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],                         # 12
                self.actions,                                                      # 12
                lidar_norm,                                                        # 36 sectors
            ],
            dim=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        return self.obs_buf, None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # Reset joints
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position      = self.dof_pos[envs_idx],
            dofs_idx_local= self.motor_dofs,
            zero_velocity = True,
            envs_idx      = envs_idx,
        )

        # Reset base pose
        self.base_pos[envs_idx]  = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx],  zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # Randomise obstacle positions for reset envs
        self._randomise_obstacles(envs_idx)

        # Reset buffers
        self.last_actions[envs_idx]       = 0.0
        self.last_dof_vel[envs_idx]       = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx]          = True

        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)
        self.commands[envs_idx, 0] = gs_rand_float(
            *self.command_cfg["lin_vel_x_range"], (n,), self.device
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg["lin_vel_y_range"], (n,), self.device
        )
        self.commands[envs_idx, 2] = gs_rand_float(
            *self.command_cfg["ang_vel_range"],   (n,), self.device
        )

    def set_curriculum_phase(self, phase: int):
        """
        Set the curriculum phase from the trainer.

        Two phases:
          Phase 1: walk only — no avoidance reward (short warm-up)
          Phase 2: full avoidance reward enabled

        Flat mode: cylinders parked in Phase 1, randomised in Phase 2.
        House mode: walls always present, only avoidance reward changes.
        """
        if phase == self.curriculum_phase:
            return
        self.curriculum_phase = phase
        if self.use_house:
            self.obstacles_active = False       # no movable cylinders in house
            self.avoidance_active = phase >= 2
        else:
            self.obstacles_active = phase >= 2
            self.avoidance_active = phase >= 2
        print(f"\n  [Curriculum] → Phase {phase}  "
              f"mode={'house' if self.use_house else 'flat'}  "
              f"avoidance={'ON' if self.avoidance_active else 'OFF'}\n")

    def _randomise_obstacles(self, envs_idx):
        """
        Place obstacles at random positions around each reset env's spawn.
        In Phase 1: park all obstacles far off-scene (no obstacles).
        In Phase 2+: randomise within the ring around spawn.
        """
        if len(envs_idx) == 0:
            return

        if not self.obstacles_active:
            # Phase 1 — park obstacles far away, out of simulation range
            for obs_entity in self.obstacles:
                far = torch.full(
                    (len(envs_idx), 3), 999.0, device=self.device
                )
                far[:, 2] = OBSTACLE_HEIGHT / 2
                obs_entity.set_pos(far, envs_idx=envs_idx)
            return

        # Phase 2 and 3 — random ring placement
        for obs_entity in self.obstacles:
            angles = torch.rand(len(envs_idx), device=self.device) * 2 * math.pi
            radii  = (
                OBSTACLE_RING_MIN
                + torch.rand(len(envs_idx), device=self.device)
                * (OBSTACLE_RING_MAX - OBSTACLE_RING_MIN)
            )
            # Place relative to spawn position
            spawn = self.base_init_pos
            x = spawn[0] + radii * torch.cos(angles)
            y = spawn[1] + radii * torch.sin(angles)
            z = torch.full((len(envs_idx),), OBSTACLE_HEIGHT / 2, device=self.device)
            pos = torch.stack([x, y, z], dim=-1)
            obs_entity.set_pos(pos, envs_idx=envs_idx)

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    def _reward_tracking_lin_vel(self):
        error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_base_height(self):
        return torch.square(
            self.base_pos[:, 2] - self.reward_cfg["base_height_target"]
        )

    def _reward_action_rate(self):
        return torch.sum(
            torch.square(self.last_actions - self.actions), dim=1
        )

    def _reward_similar_to_default(self):
        return torch.sum(
            torch.abs(self.dof_pos - self.default_dof_pos), dim=1
        )

    def _reward_obstacle_avoidance(self):
        """
        Penalty based on minimum LiDAR distance.
        Only active in Phase 2+. Returns zero in Phase 1.
        """
        if not self.avoidance_active:
            return torch.zeros(self.n_envs, device=self.device, dtype=gs.tc_float)

        min_dist = self.lidar_sectors.min(dim=1).values
        min_dist = torch.clamp(min_dist, 0.0, LIDAR_MAX_RANGE)

        danger_pen = torch.where(
            min_dist < LIDAR_DANGER_ZONE,
            torch.square(min_dist - LIDAR_DANGER_ZONE),
            torch.zeros_like(min_dist),
        )
        caution_pen = torch.where(
            (min_dist >= LIDAR_DANGER_ZONE) & (min_dist < LIDAR_CAUTION_ZONE),
            (LIDAR_CAUTION_ZONE - min_dist) * 0.2,
            torch.zeros_like(min_dist),
        )
        return danger_pen + caution_pen


# ==========================================================================
#  ActorCritic (identical to main.py — cross-compatible checkpoints)
# ==========================================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden,  hidden), nn.ELU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, act_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.00)

    def forward(self, obs):
        h     = self.trunk(obs)
        mean  = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        std   = self.log_std.exp().expand_as(mean)
        return mean, std, value

    def get_action(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        dist     = torch.distributions.Normal(mean, std)
        action   = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def get_value(self, obs):
        return self.critic_head(self.trunk(obs)).squeeze(-1)

    def evaluate(self, obs, action):
        mean, std, value = self.forward(obs)
        dist     = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy, value

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ==========================================================================
#  RolloutBuffer (identical to main.py)
# ==========================================================================

class RolloutBuffer:
    def __init__(self, rollout_steps, n_envs, obs_dim, act_dim, device):
        self.T, self.N, self.device, self.ptr = rollout_steps, n_envs, device, 0

        def buf(*shape):
            return torch.zeros(*shape, device=device)

        self.obs        = buf(rollout_steps, n_envs, obs_dim)
        self.actions    = buf(rollout_steps, n_envs, act_dim)
        self.log_probs  = buf(rollout_steps, n_envs)
        self.values     = buf(rollout_steps, n_envs)
        self.rewards    = buf(rollout_steps, n_envs)
        self.dones      = buf(rollout_steps, n_envs)
        self.advantages = buf(rollout_steps, n_envs)
        self.returns    = buf(rollout_steps, n_envs)

    def store_step(self, obs, actions, log_probs, values):
        t = self.ptr
        self.obs[t]       = obs.detach()
        self.actions[t]   = actions.detach()
        self.log_probs[t] = log_probs.detach()
        self.values[t]    = values.detach()

    def store_outcome(self, rewards, dones):
        t = self.ptr
        self.rewards[t] = rewards.detach().float()
        self.dones[t]   = dones.detach().float()
        self.ptr += 1

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            next_val = last_value if t == self.T - 1 else self.values[t + 1]
            mask     = 1.0 - self.dones[t]
            delta    = self.rewards[t] + gamma * next_val * mask - self.values[t]
            gae      = delta + gamma * lam * mask * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values
        self.ptr = 0

    def get_flat(self):
        T, N = self.T, self.N
        return (
            self.obs.view(T*N, -1),
            self.actions.view(T*N, -1),
            self.log_probs.view(T*N),
            self.advantages.view(T*N),
            self.returns.view(T*N),
            self.values.view(T*N),
        )


# ==========================================================================
#  PPO Trainer
# ==========================================================================

class PPOTrainer:

    def __init__(self, cfg):
        self.cfg    = cfg
        self.device = cfg["device"]

        self.env = Go2LidarNavEnv(
            n_envs            = cfg["n_envs"],
            dt                = cfg["dt"],
            max_episode_steps = cfg["max_episode_steps"],
            headless          = cfg["headless"],
            device            = cfg["device"],
            scene_json        = cfg.get("scene_json"),
            asset_root        = cfg.get("asset_root"),
        )

        obs_dim = self.env.OBS_DIM
        self.net = ActorCritic(
            obs_dim = obs_dim,
            act_dim = Go2LidarNavEnv.ACT_DIM,
            hidden  = cfg["hidden_size"],
        ).to(self.device)
        print(f"  Network params : {self.net.num_parameters:,}")
        print(f"  Obs dim        : {obs_dim}  (45 proprioception + {N_SECTORS} LiDAR sectors)")

        # Load pretrained walking policy if provided
        # The pretrained network has obs_dim=45, ours is 81.
        # We load only the trunk and actor/critic heads — the extra 36 LiDAR
        # input weights are initialised to near-zero so the policy ignores
        # LiDAR initially and gradually learns to use it.
        if cfg.get("pretrained"):
            self._load_pretrained(cfg["pretrained"])

        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=cfg["lr"], eps=1e-5
        )
        self.buf = RolloutBuffer(
            rollout_steps = cfg["rollout_steps"],
            n_envs        = cfg["n_envs"],
            obs_dim       = obs_dim,
            act_dim       = Go2LidarNavEnv.ACT_DIM,
            device        = self.device,
        )

        os.makedirs(cfg["run_dir"], exist_ok=True)
        self.writer      = SummaryWriter(cfg["run_dir"])
        self.global_step = 0
        self.start_time  = time.time()
        self.ep_returns  = []
        self.ep_lengths  = []
        self._env_ret    = torch.zeros(cfg["n_envs"], device=self.device)
        self._env_len    = torch.zeros(cfg["n_envs"], device=self.device, dtype=torch.int32)

    def _load_pretrained(self, path: str):
        """
        Load trunk + heads from a pretrained flat-terrain walking checkpoint.
        The pretrained network has obs_dim=45; ours is 81 (45+36 LiDAR).
        Strategy: copy matching weights, leave the 36 LiDAR input weights
        near zero so the policy initially ignores sensor data and relies
        on the walking behaviour it already learned.
        """
        print(f"\n  Loading pretrained weights: {path}")
        ckpt        = torch.load(path, weights_only=False, map_location=self.device)
        src_state   = ckpt["model_state"]
        dst_state   = self.net.state_dict()

        n_loaded = 0
        for key, dst_param in dst_state.items():
            if key not in src_state:
                continue
            src_param = src_state[key]
            if src_param.shape == dst_param.shape:
                # Exact match — copy directly (trunk layers 1+, heads)
                dst_state[key] = src_param.clone()
                n_loaded += 1
            elif key == "trunk.0.weight":
                # First trunk layer: [hidden, obs_dim_new] vs [hidden, obs_dim_old]
                # Copy the proprioception columns, leave LiDAR columns at ~0
                dst_param[:, :src_param.shape[1]] = src_param.clone()
                dst_state[key] = dst_param
                n_loaded += 1

        self.net.load_state_dict(dst_state)
        print(f"  Loaded {n_loaded}/{len(dst_state)} parameter tensors from pretrained\n")

    def _update_curriculum(self, global_step: int):
        """Update curriculum phase based on global step count."""
        if global_step < CURRICULUM_PHASE1_STEPS:
            self.env.set_curriculum_phase(1)
        else:
            self.env.set_curriculum_phase(2)

    def train(self):
        cfg = self.cfg
        obs, _ = self.env.reset()

        print(f"\n{'='*55}")
        print(f"  Go2 LiDAR Navigation PPO Training")
        print(f"{'='*55}")
        print(f"  Envs          : {cfg['n_envs']}")
        print(f"  Obstacles/env : {N_OBSTACLES}")
        print(f"  Total steps   : {cfg['total_steps']:,}")
        print(f"  Device        : {self.device}")
        print(f"  Run dir       : {cfg['run_dir']}")
        print(f"  Curriculum:")
        print(f"    Phase 1 (walk only) :    0 → {CURRICULUM_PHASE1_STEPS:,}")
        print(f"    Phase 2 (avoidance) : {CURRICULUM_PHASE1_STEPS:,} → end")
        print(f"{'='*55}\n")

        steps_per_rollout = cfg["rollout_steps"] * cfg["n_envs"]
        n_updates         = cfg["total_steps"] // steps_per_rollout

        for update in range(1, n_updates + 1):
            # Update curriculum phase before collecting rollout
            self._update_curriculum(self.global_step)

            obs     = self._collect_rollout(obs)
            metrics = self._ppo_update()
            self.global_step += steps_per_rollout

            if update % cfg["log_interval"] == 0:
                elapsed   = time.time() - self.start_time
                sps       = self.global_step / elapsed
                mean_ret  = float(np.mean(self.ep_returns[-20:])) if self.ep_returns else 0.0
                mean_len  = float(np.mean(self.ep_lengths[-20:])) if self.ep_lengths else 0.0

                print(
                    f"  step {self.global_step:>10,} | "
                    f"ph {self.env.curriculum_phase} | "
                    f"ret {mean_ret:>7.3f} | "
                    f"len {mean_len:>5.0f} | "
                    f"ploss {metrics['policy_loss']:>7.4f} | "
                    f"vloss {metrics['value_loss']:>7.4f} | "
                    f"clip {metrics['clip_frac']:>4.2f} | "
                    f"{sps:>7,.0f} sps"
                )
                self.writer.add_scalar("train/mean_return",   mean_ret,               self.global_step)
                self.writer.add_scalar("train/mean_ep_len",   mean_len,               self.global_step)
                self.writer.add_scalar("loss/policy",         metrics["policy_loss"], self.global_step)
                self.writer.add_scalar("loss/value",          metrics["value_loss"],  self.global_step)
                self.writer.add_scalar("loss/entropy",        metrics["entropy"],     self.global_step)
                self.writer.add_scalar("train/clip_fraction", metrics["clip_frac"],   self.global_step)
                self.writer.add_scalar("train/sps",           sps,                    self.global_step)
                self.writer.add_scalar("curriculum/phase",    self.env.curriculum_phase, self.global_step)

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

                next_obs, _, reward, reset_buf, extras = self.env.step(action)
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
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        total   = obs_f.shape[0]
        metrics = {"policy_loss":[], "value_loss":[], "entropy":[], "clip_frac":[]}

        for _ in range(cfg["n_epochs"]):
            idx = torch.randperm(total, device=self.device)
            for start in range(0, total, cfg["minibatch_size"]):
                mb         = idx[start : start + cfg["minibatch_size"]]
                new_lp, entropy, value = self.net.evaluate(obs_f[mb], act_f[mb])
                ratio      = (new_lp - lp_f[mb]).exp()
                surr1      = ratio * adv_f[mb]
                surr2      = ratio.clamp(1-cfg["clip_eps"], 1+cfg["clip_eps"]) * adv_f[mb]
                policy_loss= -torch.min(surr1, surr2).mean()

                vclip      = val_f[mb] + (value - val_f[mb]).clamp(-cfg["clip_eps"], cfg["clip_eps"])
                value_loss = torch.max(
                    (value - ret_f[mb]).pow(2),
                    (vclip  - ret_f[mb]).pow(2),
                ).mean()

                loss = policy_loss + cfg["vf_coef"] * value_loss - cfg["ent_coef"] * entropy.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg["max_grad_norm"])
                self.opt.step()

                clip_frac = ((ratio - 1.0).abs() > cfg["clip_eps"]).float().mean()
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.mean().item())
                metrics["clip_frac"].append(clip_frac.item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def _save_checkpoint(self, tag=None):
        name = f"checkpoint_step_{self.global_step:09d}" if not tag else f"checkpoint_{tag}"
        path = os.path.join(self.cfg["run_dir"], f"{name}.pt")
        torch.save({
            "step":        self.global_step,
            "model_state": self.net.state_dict(),
            "optim_state": self.opt.state_dict(),
            "config":      self.cfg,
            "obs_dim":          self.env.OBS_DIM,
            "n_sectors":        N_SECTORS,
            "curriculum_phase": self.env.curriculum_phase,
            "metrics": {
                "mean_return": np.mean(self.ep_returns[-20:]) if self.ep_returns else 0.0,
                "mean_length": np.mean(self.ep_lengths[-20:]) if self.ep_lengths else 0.0,
            },
        }, path)
        print(f"  [ckpt] Saved {path}")

    @classmethod
    def load_checkpoint(cls, path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt   = torch.load(path, weights_only=False, map_location=device)
        trainer= cls(ckpt["config"])
        trainer.net.load_state_dict(ckpt["model_state"])
        trainer.opt.load_state_dict(ckpt["optim_state"])
        trainer.global_step = ckpt["step"]
        print(f"  Resumed step={trainer.global_step:,}  "
              f"ret={ckpt['metrics']['mean_return']:.3f}")
        return trainer


# ==========================================================================
#  Evaluation
# ==========================================================================

def evaluate(checkpoint_path, n_episodes=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg    = ckpt["config"]

    # Read actual obs_dim from checkpoint weights — don't rely on env
    obs_dim = ckpt["model_state"]["trunk.0.weight"].shape[1]
    act_dim = ckpt["model_state"]["actor_head.2.weight"].shape[0]
    hidden  = cfg.get("hidden_size", 512)
    print(f"  Checkpoint: obs_dim={obs_dim} act_dim={act_dim} hidden={hidden}")

    env = Go2LidarNavEnv(
        n_envs            = 1,
        headless          = False,
        max_episode_steps = cfg["max_episode_steps"],
        dt                = cfg["dt"],
        device            = device,
        scene_json        = cfg.get("scene_json"),
        asset_root        = cfg.get("asset_root"),
    )

    net = ActorCritic(obs_dim, act_dim, hidden)
    net.load_state_dict(ckpt["model_state"])
    net.eval().to(device)

    for ep in range(n_episodes):
        obs, _ = env.reset()

        # Pad or trim obs to match checkpoint obs_dim if env returns different size
        # This handles the case where LiDAR gives different ray counts on CPU vs GPU
        if obs.shape[-1] != obs_dim:
            print(f"  ⚠️  obs dim mismatch: env={obs.shape[-1]} checkpoint={obs_dim}")
            print(f"     Padding/trimming to match checkpoint.")
            if obs.shape[-1] < obs_dim:
                pad = torch.zeros(obs.shape[0], obs_dim - obs.shape[-1], device=device)
                obs = torch.cat([obs, pad], dim=-1)
            else:
                obs = obs[:, :obs_dim]

        # One-time full sector breakdown at start of episode 1
        if ep == 0:
            sectors = env.lidar_sectors[0].cpu().numpy()
            print(f"\n  LiDAR sector breakdown (episode 1 start):")
            print(f"  {'Sector':>6}  {'Angle':>7}  {'Distance':>10}")
            for i, d in enumerate(sectors):
                angle = i * (360.0 / N_SECTORS)
                flag  = " ← CLOSE" if d < 1.0 else " ← caution" if d < 2.0 else ""
                print(f"  {i:>6}  {angle:>6.0f}°  {d:>8.2f}m{flag}")
            print(f"  Min: {sectors.min():.2f}m  Mean: {sectors.mean():.2f}m  "
                  f"Max: {sectors.max():.2f}m")
            print(f"  All at max range ({LIDAR_MAX_RANGE}m)? "
                  f"{'YES — sensor may not be detecting walls' if sectors.min() > LIDAR_MAX_RANGE * 0.99 else 'NO — sensor working'}\n")

        done   = torch.zeros(1, dtype=torch.bool)
        ep_ret, ep_len = 0.0, 0
        lidar_mins = []

        while not done[0]:
            with torch.no_grad():
                act, _, _ = net.get_action(obs, deterministic=True)
            obs, _, reward, reset_buf, _ = env.step(act)

            # Apply same dim correction each step
            if obs.shape[-1] != obs_dim:
                if obs.shape[-1] < obs_dim:
                    pad = torch.zeros(obs.shape[0], obs_dim - obs.shape[-1], device=device)
                    obs = torch.cat([obs, pad], dim=-1)
                else:
                    obs = obs[:, :obs_dim]

            # Track LiDAR diagnostics
            min_dist  = env.lidar_sectors.min().item()
            mean_dist = env.lidar_sectors.mean().item()
            lidar_mins.append(min_dist)

            ep_ret += reward[0].item()
            ep_len += 1
            done    = reset_buf.bool()

            # Print LiDAR every 50 steps
            if ep_len % 50 == 0:
                print(f"    step {ep_len:4d}  "
                      f"min_lidar={min_dist:.2f}m  "
                      f"mean_lidar={mean_dist:.2f}m  "
                      f"reward={reward[0].item():.3f}")

        print(f"  Episode {ep+1} | return={ep_ret:.2f} | length={ep_len}")
        if lidar_mins:
            print(f"    LiDAR stats — min={min(lidar_mins):.2f}m  "
                  f"mean={sum(lidar_mins)/len(lidar_mins):.2f}m  "
                  f"max={max(lidar_mins):.2f}m")


# ==========================================================================
#  Entry point
# ==========================================================================

def get_config(args):
    total_buffer   = args.n_envs * args.rollout_steps
    minibatch_size = max(total_buffer // 4, 256)
    return dict(
        n_envs             = args.n_envs,
        dt                 = 0.02,
        max_episode_steps  = 1000,
        headless           = args.headless,
        hidden_size        = 512,
        total_steps        = args.total_steps,
        rollout_steps      = args.rollout_steps,
        minibatch_size     = minibatch_size,
        n_epochs           = 5,
        gamma              = 0.99,
        lam                = 0.95,
        clip_eps           = 0.2,
        lr                 = 3e-4,
        vf_coef            = 1.0,
        ent_coef           = 0.01,
        max_grad_norm      = 1.0,
        device             = args.device,
        run_dir            = args.run_dir,
        log_interval       = 10,
        save_interval      = 100,
        pretrained         = args.pretrained,
        scene_json         = args.scene,
        asset_root         = args.asset_root,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs",        type=int,  default=4096)
    parser.add_argument("--total-steps",   type=int,  default=200_000_000)
    parser.add_argument("--rollout-steps", type=int,  default=24)
    parser.add_argument("--device",        type=str,  default="cuda",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--run-dir",       type=str,  default="../../runs/go2_lidar")
    parser.add_argument("--headless",      action="store_true", default=True)
    parser.add_argument("--resume",        type=str,  default=None)
    parser.add_argument("--eval",          type=str,  default=None)
    parser.add_argument("--pretrained",    type=str,  default=None,
                        help="Path to flat-terrain walking checkpoint for warm-start")
    parser.add_argument("--scene",         type=str,  default=None,
                        help="Path to ReplicaCAD scene JSON (e.g. data/replica_cad/configs/scenes/apt_0.scene_instance.json)")
    parser.add_argument("--asset-root",    type=str,  default="data/replica_cad/",
                        help="Root directory of ReplicaCAD dataset")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval)
        return

    if args.resume:
        trainer = PPOTrainer.load_checkpoint(args.resume)
    else:
        cfg     = get_config(args)
        trainer = PPOTrainer(cfg)

    trainer.train()


if __name__ == "__main__":
    main()
