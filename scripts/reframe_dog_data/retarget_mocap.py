"""
scripts/retarget_mocap.py
--------------------------
Stage 1: Convert Zhang 2018 dog BVH files into a Go2-compatible
AMP reference dataset using the sparse foot-position representation
(Path A from Escontrela 2022).

Fill in the configuration section below AFTER running diagnose_bvh.py.

Usage:
    # Step 1: run diagnosis on one file
    python scripts/diagnose_bvh.py data/dog_mocap/walk_000.bvh

    # Step 2: fill in the config below, then run
    python scripts/retarget_mocap.py

Output:
    data/amp_go2_reference.npy   shape: [N_frames, 22]  float32

State layout (22 values per frame, Path A sparse representation):
    [0:3]   FL foot position in Go2 base frame (metres, Z-up)
    [3:6]   FR foot position in Go2 base frame
    [6:9]   RL foot position in Go2 base frame
    [9:12]  RR foot position in Go2 base frame
    [12:15] projected gravity vector in base frame (unit vector)
    [15]    base height (metres)
    [16:19] base linear velocity in base frame (m/s)
    [19:22] base angular velocity in base frame (rad/s)
"""

import numpy as np
import bvhio
from pathlib import Path


# ==========================================================================
#  CONFIGURATION — fill in after running diagnose_bvh.py
# ==========================================================================

BVH_DIR     = "data/dog_mocap/"
OUTPUT_DIR = "data/amp_reference_clips/"

# Unit conversion: 0.01 if BVH is in centimetres, 1.0 if already metres
# diagnose_bvh.py will tell you which one
UNIT_SCALE       = 0.01          # centimetres → metres
DOG_LEG_LENGTH   = 0.467         # from diagnosis (root height ≈ hip-to-ground)

# ── Foot joint names ──────────────────────────────────────────────────────
# Fill these in from the diagnose_bvh.py output.
# These are the BVH joint names for each of the four feet.
# They are the leaf joints at the bottom of each leg chain.
FOOT_JOINT_NAMES = {
    "FL": "LeftHand",     # Front-Left  — LeftForeArm end = front left foot
    "FR": "RightHand",    # Front-Right — RightForeArm end = front right foot
    "RL": "LeftFoot",     # Rear-Left
    "RR": "RightFoot",    # Rear-Right
}

# ── Physical scale ────────────────────────────────────────────────────────
# These come from the robot URDF and the BVH skeleton T-pose.
# Go2 hip-to-foot length (upper leg + lower leg):
GO2_LEG_LENGTH = 0.426   # metres (0.213 + 0.213)

# German Shepherd hip-to-foot length (measure from BVH T-pose after unit
# conversion). Run diagnose_bvh.py and look at the root-to-foot distance
# in a standing frame. Typical value for this dataset: ~0.52m.
DOG_LEG_LENGTH = 0.52    # metres — adjust based on diagnosis output

# Scale factor applied to all foot positions after unit conversion
MORPHOLOGY_SCALE = GO2_LEG_LENGTH / DOG_LEG_LENGTH

# ── Filtering ────────────────────────────────────────────────────────────
# Skip BVH files matching these substrings (e.g. terrain sequences)
SKIP_PATTERNS = ["terrain", "uneven", "obstacle"]

# Minimum frames to process a file (skip very short clips)
MIN_FRAMES = 30


# ==========================================================================
#  Coordinate conversion utilities
# ==========================================================================

def bvh_to_genesis(pos: np.ndarray) -> np.ndarray:
    """
    Convert position from BVH convention to Genesis convention.

    BVH (bvhio):  Y-up,  Z-forward  (right-handed, OpenGL)
    Genesis:       Z-up,  Y-forward  (right-handed, robotics)

    Mapping:
        BVH X (right)    → Genesis X (right)    unchanged
        BVH Y (up)       → Genesis Z (up)
        BVH Z (forward)  → Genesis -Y (forward, flip sign due to handedness)

    pos: array of shape (..., 3)
    """
    return np.stack([
         pos[..., 0],   # X unchanged
        -pos[..., 2],   # BVH Z → Genesis -Y
         pos[..., 1],   # BVH Y → Genesis Z
    ], axis=-1)


def quat_bvh_to_genesis(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from BVH/bvhio convention to Genesis convention.
    bvhio returns quaternions as (w, x, y, z).
    We apply the same axis remap as bvh_to_genesis to the xyz part.

    q: array of shape (4,) — (w, x, y, z)
    returns: array of shape (4,) — (w, x, y, z) in Genesis convention
    """
    w, x, y, z = q
    # Apply same coordinate remap: X→X, Y→Z, Z→-Y
    # new_x = x, new_y = -z, new_z = y
    return np.array([w, x, -z, y], dtype=np.float32)


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate world-frame vector v into body frame.
    Equivalent to R(q)^T @ v.

    q_wxyz: (4,)  quaternion in (w, x, y, z) convention
    v:      (..., 3)
    returns: (..., 3)
    """
    w, x, y, z = q_wxyz
    vx = v[..., 0]
    vy = v[..., 1]
    vz = v[..., 2]
    # Rotate by conjugate quaternion (w, -x, -y, -z)
    bx = (1-2*(y*y+z*z))*vx +   2*(x*y+w*z)*vy +   2*(x*z-w*y)*vz
    by =   2*(x*y-w*z)*vx + (1-2*(x*x+z*z))*vy +   2*(y*z+w*x)*vz
    bz =   2*(x*z+w*y)*vx +   2*(y*z-w*x)*vy + (1-2*(x*x+y*y))*vz
    return np.stack([bx, by, bz], axis=-1)


# ==========================================================================
#  Per-file processing
# ==========================================================================

def process_bvh(filepath: str) -> np.ndarray | None:
    """
    Process one BVH file into a sequence of AMP states.

    Returns array of shape [T, 22] or None if file should be skipped.
    """
    # Check skip patterns
    name = Path(filepath).name.lower()
    for pattern in SKIP_PATTERNS:
        if pattern in name:
            print(f"    Skipping (pattern '{pattern}'): {name}")
            return None

    root = bvhio.readAsHierarchy(filepath)

    # Count frames
    n_frames = len(root.Keyframes)
    if n_frames < MIN_FRAMES:
        print(f"    Skipping (too short: {n_frames} frames): {name}")
        return None

    # Collect world positions per frame for root and feet
    root_positions_raw  = np.zeros((n_frames, 3), dtype=np.float32)
    root_quats_raw      = np.zeros((n_frames, 4), dtype=np.float32)
    foot_positions_raw  = {k: np.zeros((n_frames, 3), dtype=np.float32)
                           for k in ["FL", "FR", "RL", "RR"]}

    # Find foot joints
    foot_joints = {}
    for leg, joint_name in FOOT_JOINT_NAMES.items():
        matches = root.filter(joint_name)
        if not matches:
            print(f"    ERROR: joint '{joint_name}' not found in {name}")
            print(f"    Run diagnose_bvh.py to see available joint names.")
            return None
        foot_joints[leg] = matches[0]

    # Extract per-frame data
    for i in range(n_frames):
        root.loadPose(i)

        # Root
        rp = np.array(root.PositionWorld, dtype=np.float32)
        rq = np.array(root.RotationWorld, dtype=np.float32)  # (w,x,y,z)
        root_positions_raw[i] = rp
        root_quats_raw[i]     = rq

        # Feet
        for leg, joint in foot_joints.items():
            fp = np.array(joint.PositionWorld, dtype=np.float32)
            foot_positions_raw[leg][i] = fp

    # ── Unit conversion ──────────────────────────────────────────
    root_positions_raw  *= UNIT_SCALE
    for leg in foot_positions_raw:
        foot_positions_raw[leg] *= UNIT_SCALE

    # ── Coordinate system conversion (Y-up → Z-up) ───────────────
    root_pos  = bvh_to_genesis(root_positions_raw)   # [T, 3]
    root_quat = np.array([quat_bvh_to_genesis(q) for q in root_quats_raw])
    foot_pos  = {leg: bvh_to_genesis(foot_positions_raw[leg])
                 for leg in foot_positions_raw}

    # ── Morphology scale ─────────────────────────────────────────
    # Scale foot positions relative to root so they match Go2 proportions
    for leg in foot_pos:
        rel = foot_pos[leg] - root_pos
        rel *= MORPHOLOGY_SCALE
        foot_pos[leg] = root_pos + rel

    # ── Compute foot positions in base frame ─────────────────────
    foot_base = {}
    for leg in foot_pos:
        rel_world = foot_pos[leg] - root_pos          # [T, 3]
        # Rotate into body frame using inverse of root quaternion
        foot_base[leg] = np.array([
            quat_rotate_inverse(root_quat[i], rel_world[i])
            for i in range(n_frames)
        ])                                             # [T, 3]

    # ── Projected gravity in body frame ──────────────────────────
    gravity_world = np.tile([0.0, 0.0, -1.0], (n_frames, 1))
    proj_gravity  = np.array([
        quat_rotate_inverse(root_quat[i], gravity_world[i])
        for i in range(n_frames)
    ])                                                 # [T, 3]

    # ── Base height ───────────────────────────────────────────────
    base_height = root_pos[:, 2:3]                     # [T, 1]

    # ── Velocities via finite difference ─────────────────────────
    dt = 1.0 / 30.0   # Zhang 2018 is 30 fps

    # Linear velocity: derivative of world position, rotated to body frame
    lin_vel_world = np.gradient(root_pos, dt, axis=0)  # [T, 3]
    lin_vel_base  = np.array([
        quat_rotate_inverse(root_quat[i], lin_vel_world[i])
        for i in range(n_frames)
    ])                                                 # [T, 3]

    # Angular velocity: from derivative of projected gravity (approximation)
    # This is simpler than computing from quaternion derivatives and
    # sufficient for the AMP discriminator
    ang_vel_base = np.gradient(proj_gravity, dt, axis=0)  # [T, 3]

    # ── Assemble state vector ─────────────────────────────────────
    # [FL_pos(3), FR_pos(3), RL_pos(3), RR_pos(3),
    #  proj_grav(3), height(1), lin_vel(3), ang_vel(3)] = 22 values
    states = np.concatenate([
        foot_base["FL"],   # [T, 3]
        foot_base["FR"],   # [T, 3]
        foot_base["RL"],   # [T, 3]
        foot_base["RR"],   # [T, 3]
        proj_gravity,      # [T, 3]
        base_height,       # [T, 1]
        lin_vel_base,      # [T, 3]
        ang_vel_base,      # [T, 3]
    ], axis=1)             # [T, 22]

    return states.astype(np.float32)


# ==========================================================================
#  Main pipeline
# ==========================================================================

def build_dataset():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    bvh_files = sorted(Path(BVH_DIR).glob("*.bvh"))
    saved = []

    for bvh_file in bvh_files:
        print(f"  Processing: {bvh_file.name}")
        states = process_bvh(str(bvh_file))
        if states is None:
            continue

        # Zero velocity at clip boundaries — prevents spike artifacts
        states[:3,  16:22] = 0.0
        states[-3:, 16:22] = 0.0

        # Save one file per clip
        out_path = Path(OUTPUT_DIR) / (bvh_file.stem + ".npy")
        np.save(str(out_path), states)
        saved.append(str(out_path))
        print(f"    → {len(states)} frames → {out_path.name}")

    # Also save a manifest listing all clip paths
    manifest = Path(OUTPUT_DIR) / "manifest.txt"
    manifest.write_text("\n".join(saved))
    print(f"\nSaved {len(saved)} clips to {OUTPUT_DIR}")
    print(f"Manifest: {manifest}")

if __name__ == "__main__":
    build_dataset()
