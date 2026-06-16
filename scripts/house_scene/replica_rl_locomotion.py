"""
replica_rl_locomotion.py
------------------------
Loads a ReplicaCAD room scene and runs the trained Go2 RL locomotion
policy inside it.

Based on your working replica_cad scene loader, with the Go2 robot
and trained PPO policy added on top. Nothing in the scene loading
logic has been changed.

Requirements:
    - A trained checkpoint from main.py
    - The ReplicaCAD dataset at ./data/replica_cad/
    - genesis-world installed

Usage:
    python replica_rl_locomotion.py \
        --checkpoint runs/go2_walk/checkpoint_final.pt \
        --scene     data/replica_cad/configs/scenes/apt_0.scene_instance.json

Controls:
    The robot runs with a fixed forward velocity command (vx=0.4 m/s).
    Close the viewer window to exit.

Notes:
    - Scene uses gs.cpu (no GPU rendering on H200). If running on a
      machine with display and GPU, change gs.cpu → gs.gpu below.
    - The RL policy runs on CPU for compatibility with the scene backend.
      This is slower but correct for single-env evaluation.
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import genesis as gs
from genesis.utils.geom import transform_by_quat, inv_quat


# ==========================================================================
#  Configuration
# ==========================================================================

# Go2 joint order — must match training
MOTOR_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

# Default standing pose (must match training)
DEFAULT_JOINT_POS = torch.tensor([
    0.0,  0.8, -1.5,   # FR
    0.0,  0.8, -1.5,   # FL
    0.0,  1.0, -1.5,   # RR
    0.0,  1.0, -1.5,   # RL
], dtype=torch.float32)

# Go2 spawn position — place robot in a clear area of the room
# Adjust these if the robot spawns inside furniture
GO2_SPAWN_POS  = [1.0, 1.0, 0.45]    # x, y, z (metres, Genesis Z-up)
GO2_SPAWN_QUAT = [0.0, 0.0, 0.0, 1.0]  # [x,y,z,w] identity

# Observation and action dimensions (must match training)
OBS_DIM = 45
ACT_DIM = 12
ACTION_SCALE = 0.25

# Observation scales (must match training in go2.py)
OBS_SCALES = {
    "ang_vel":  0.25,
    "dof_pos":  1.0,
    "dof_vel":  0.05,
    "lin_vel":  2.0,
}


# ==========================================================================
#  Policy network (must match main.py ActorCritic exactly)
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

    def get_action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation."""
        return self.actor_head(self.trunk(obs))


def load_policy(checkpoint_path: str, device: str) -> ActorCritic:
    print(f"Loading policy: {checkpoint_path}")
    ckpt   = torch.load(checkpoint_path, weights_only=False,
                        map_location=device)
    cfg    = ckpt["config"]
    hidden = cfg.get("hidden_size", 512)
    policy = ActorCritic(OBS_DIM, ACT_DIM, hidden)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()
    policy = policy.to(device)
    print(f"  Policy loaded — hidden={hidden}  "
          f"step={ckpt.get('step', '?'):,}")
    return policy


# ==========================================================================
#  Scene loading (your original code, unchanged)
# ==========================================================================

def convert_habitat_to_genesis(pos, quat_wxyz):
    """Swizzles Habitat Y-up → Genesis Z-up."""
    x, y, z   = pos
    qw, qx, qy, qz = quat_wxyz
    return [x, -z, y], [qw, qx, -qz, qy]


def resolve_asset_path(template_path, root_dir):
    """Hunts for the matching Habitat config file and returns the asset path."""
    base_name  = os.path.basename(template_path)
    search_dir = os.path.join(root_dir, "configs")
    if not os.path.exists(search_dir):
        search_dir = root_dir

    for subdir, dirs, files in os.walk(search_dir):
        for file in files:
            if file.startswith(base_name) and file.endswith(".json"):
                json_path = os.path.join(subdir, file)
                try:
                    with open(json_path, "r") as f:
                        obj_config = json.load(f)
                    asset_rel = (obj_config.get("urdf_filepath") or
                                 obj_config.get("render_asset"))
                    if asset_rel:
                        abs_path = os.path.abspath(
                            os.path.join(os.path.dirname(json_path), asset_rel)
                        )
                        if os.path.exists(abs_path):
                            return abs_path
                except Exception:
                    pass
    return None


def spawn_entity(scene, template_name, asset_path, pos, quat,
                 is_fixed, scale=1.0):
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


def build_scene(scene_json: str, asset_root: str, scene) -> None:
    # scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.1)))
    """Load the ReplicaCAD scene into an existing Genesis scene object."""
    with open(scene_json, "r") as f:
        config = json.load(f)

    # Stage (room shell)
    print("\n── Stage ──")
    stage_info = config.get("stage_instance", {})
    stage_tmpl = stage_info.get("template_name")
    if stage_tmpl:
        room_asset = resolve_asset_path(stage_tmpl, asset_root)
        if room_asset:
            rp, rq = convert_habitat_to_genesis(
                stage_info.get("translation", [0, 0, 0]),
                stage_info.get("rotation",    [1, 0, 0, 0]),
            )
            rp[2] -= 0.05  # Nudge up to prevent z-fighting with the floor plane
            spawn_entity(scene, stage_tmpl, room_asset,
                         pos=rp, quat=rq, is_fixed=True)
        else:
            print(f"  ⚠️  Could not resolve stage: {stage_tmpl}")

    # Rigid objects
    print("\n── Rigid objects ──")
    for obj in config.get("object_instances", []):
        tmpl  = obj["template_name"]
        asset = resolve_asset_path(tmpl, asset_root)
        if asset:
            np_, nq = convert_habitat_to_genesis(
                obj["translation"], obj["rotation"]
            )
            spawn_entity(scene, tmpl, asset,
                         pos=np_, quat=nq,
                         is_fixed=(obj.get("motion_type") == "STATIC"),
                         scale=obj.get("uniform_scale", 1.0))

    # Articulated objects
    print("\n── Articulated objects ──")
    for obj in config.get("articulated_object_instances", []):
        tmpl  = obj["template_name"]
        asset = resolve_asset_path(tmpl, asset_root)
        if asset:
            np_, nq = convert_habitat_to_genesis(
                obj["translation"], obj["rotation"]
            )
            spawn_entity(scene, tmpl, asset,
                         pos=np_, quat=nq,
                         is_fixed=obj.get("fixed_base", True),
                         scale=obj.get("uniform_scale", 1.0))


# ==========================================================================
#  Observation builder
# ==========================================================================

def get_obs(
    robot,
    motor_dofs:    list,
    commands:      torch.Tensor,   # [1, 3]
    last_actions:  torch.Tensor,   # [1, 12]
    device:        str,
) -> torch.Tensor:
    """
    Build the 45-dim observation vector matching go2.py training format.
    Runs on CPU (single env evaluation).
    """
    base_quat = robot.get_quat()                          # [1, 4]
    inv_q     = inv_quat(base_quat)

    ang_vel   = transform_by_quat(robot.get_ang(), inv_q) # [1, 3]
    grav_w    = torch.tensor([[0., 0., -1.]], device=base_quat.device)
    proj_grav = transform_by_quat(grav_w, inv_q)          # [1, 3]

    dof_pos   = robot.get_dofs_position(motor_dofs)       # [1, 12]
    dof_vel   = robot.get_dofs_velocity(motor_dofs)       # [1, 12]

    default   = DEFAULT_JOINT_POS.to(base_quat.device).unsqueeze(0)
    dof_pos_rel = dof_pos - default

    obs = torch.cat([
        ang_vel     * OBS_SCALES["ang_vel"],
        proj_grav,
        commands.to(base_quat.device),
        dof_pos_rel * OBS_SCALES["dof_pos"],
        dof_vel     * OBS_SCALES["dof_vel"],
        last_actions.to(base_quat.device),
    ], dim=-1)

    return obs.clamp(-5.0, 5.0)


# ==========================================================================
#  Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Go2 RL locomotion inside a ReplicaCAD room"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to trained .pt checkpoint from main.py"
    )
    parser.add_argument(
        "--scene", "-s",
        default="data/replica_cad/configs/scenes/apt_0.scene_instance.json",
        help="Path to ReplicaCAD scene JSON"
    )
    parser.add_argument(
        "--asset-root",
        default="data/replica_cad/",
        help="Root directory of the ReplicaCAD dataset"
    )
    parser.add_argument(
        "--vx",   type=float, default=0.4,
        help="Forward velocity command (m/s)"
    )
    parser.add_argument(
        "--vy",   type=float, default=0.0,
        help="Lateral velocity command (m/s)"
    )
    parser.add_argument(
        "--vyaw", type=float, default=0.0,
        help="Yaw rate command (rad/s)"
    )
    parser.add_argument(
        "--spawn-x", type=float, default=GO2_SPAWN_POS[0],
        help="Robot spawn X position (metres)"
    )
    parser.add_argument(
        "--spawn-y", type=float, default=GO2_SPAWN_POS[1],
        help="Robot spawn Y position (metres)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without viewer"
    )
    args = parser.parse_args()

    asset_root = os.path.abspath(args.asset_root)

    # ── Genesis init ───────────────────────────────────────────────────────
    # Using cpu to match your working scene loader.
    # Change to gs.gpu if your machine supports it.
    gs.init(backend=gs.cpu, logging_level="warning")

    scene = gs.Scene(
        show_viewer = not args.headless,
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (args.spawn_x + 2.0,
                             args.spawn_y - 2.0,
                             1.5),
            camera_lookat = (args.spawn_x,
                             args.spawn_y,
                             0.4),
            camera_fov    = 50,
            max_FPS       = 60,
        ),
        sim_options = gs.options.SimOptions(dt=0.02, substeps=2),
        rigid_options = gs.options.RigidOptions(
            multiplier_collision_broad_phase = 50,
            max_collision_pairs              = 10000,
            iterations                       = 100,
        ),
    )

    # ── Load room scene ────────────────────────────────────────────────────
    print(f"\nLoading scene: {args.scene}")
    build_scene(args.scene, asset_root, scene)

    # ── Add Go2 robot ──────────────────────────────────────────────────────
    print(f"\n── Go2 robot ──")
    spawn_pos = [args.spawn_x, args.spawn_y, GO2_SPAWN_POS[2]]
    robot = scene.add_entity(
        gs.morphs.URDF(
            file = "urdf/go2/urdf/go2.urdf",
            pos  = spawn_pos,
            quat = GO2_SPAWN_QUAT,
        )
    )
    print(f"  ✅  Go2 at {spawn_pos}")

    # ── Build scene ────────────────────────────────────────────────────────
    print("\nBuilding scene (this may take a moment)...")
    scene.build(n_envs=1)

    # ── Motor DOFs ─────────────────────────────────────────────────────────
    motor_dofs = [
        robot.get_joint(name).dof_idx_local
        for name in MOTOR_JOINT_NAMES
    ]

    # PD gains matching training
    robot.set_dofs_kp([20.0] * ACT_DIM, dofs_idx_local=motor_dofs)
    robot.set_dofs_kv([ 0.5] * ACT_DIM, dofs_idx_local=motor_dofs)

    # Set initial standing pose
    init_pos = DEFAULT_JOINT_POS.unsqueeze(0)
    robot.set_dofs_position(init_pos, dofs_idx_local=motor_dofs,
                            zero_velocity=True)

    # Warm up — let the robot settle
    print("Settling robot into standing pose...")
    target = init_pos.clone()
    for _ in range(100):
        robot.control_dofs_position(target, dofs_idx_local=motor_dofs)
        scene.step()

    # ── Load policy ────────────────────────────────────────────────────────
    # Policy runs on CPU for compatibility with gs.cpu scene
    device = "cpu"
    policy = load_policy(args.checkpoint, device)

    commands     = torch.tensor([[args.vx, args.vy, args.vyaw]])
    last_actions = torch.zeros(1, ACT_DIM)

    print(f"\nRunning policy — vx={args.vx} vy={args.vy} vyaw={args.vyaw}")
    print("Close the viewer window to exit.\n")

    step = 0
    while True:
        obs = get_obs(robot, motor_dofs, commands, last_actions, device)

        with torch.no_grad():
            action = policy.get_action_mean(obs)

        target_pos = action * ACTION_SCALE + DEFAULT_JOINT_POS.unsqueeze(0)
        robot.control_dofs_position(
            target_pos, dofs_idx_local=motor_dofs
        )
        scene.step()
        last_actions = action.clone()
        if step == 300:
            pos  = robot.get_pos()[0].cpu().numpy()
            dpos = robot.get_dofs_position(motor_dofs)[0].cpu().numpy()
            print(f"Base Z: {pos[2]:.4f}")

        # if scene.viewer is not None and step % 5 == 0:
        #     pos = robot.get_pos()[0].cpu().numpy()
        #     scene.viewer.set_camera_pose(
        #         pos    = pos + np.array([-1.5, -1.0, 0.8]),
        #         lookat = pos,
        #     )
        #
        if step % 250 == 0:
            pos = robot.get_pos()[0].cpu().numpy()
            print(f"  step {step:6d}  "
                  f"x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f}")

        step += 1


if __name__ == "__main__":
    main()
