"""
scripts/reframe_dog_data/visualize_skeleton.py
-----------------------------------------------
Render the Zhang 2018 dog BVH files as a simplified skeleton video.
Shows the raw MoCap data BEFORE retargeting so you can verify the
skeleton structure and motion quality.

Usage:
    # Render one BVH file to MP4
    python scripts/reframe_dog_data/visualize_skeleton.py \
        data/dog_mocap/D1_001_KAN01_001.bvh \
        --output data/skeleton_preview.mp4

    # Interactive viewer (no output file)
    python scripts/reframe_dog_data/visualize_skeleton.py \
        data/dog_mocap/D1_001_KAN01_001.bvh

Requires:
    pip install bvhio matplotlib numpy
    pip install ffmpeg-python   # for MP4 export
    # or: brew install ffmpeg   # on Mac
"""

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import bvhio
from pathlib import Path


# ── Skeleton definition ────────────────────────────────────────────────────
# Bones to draw: list of (parent_joint, child_joint) pairs.
# Based on the Zhang 2018 BVH joint hierarchy from diagnosis output.
BONES = [
    # Spine
    ("Hips",   "Spine"),
    ("Spine",  "Spine1"),
    ("Spine1", "Neck"),
    ("Neck",   "Head"),
    # Tail
    ("Hips",   "Tail"),
    ("Tail",   "Tail1"),
    # Front-Left leg (named as left arm in BVH)
    ("Spine1",      "LeftShoulder"),
    ("LeftShoulder","LeftArm"),
    ("LeftArm",     "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    # Front-Right leg (named as right arm in BVH)
    ("Spine1",       "RightShoulder"),
    ("RightShoulder","RightArm"),
    ("RightArm",     "RightForeArm"),
    ("RightForeArm", "RightHand"),
    # Rear-Left leg
    ("Hips",      "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg",   "LeftFoot"),
    # Rear-Right leg
    ("Hips",       "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg",   "RightFoot"),
]

# Joint colours by body part
JOINT_COLORS = {
    "Hips":          "black",
    "Spine":         "gray",
    "Spine1":        "gray",
    "Neck":          "gray",
    "Head":          "saddlebrown",
    "Tail":          "gray",
    "Tail1":         "gray",
    # Front legs (blue)
    "LeftShoulder":  "royalblue",
    "LeftArm":       "royalblue",
    "LeftForeArm":   "royalblue",
    "LeftHand":      "blue",         # FL foot
    "RightShoulder": "tomato",
    "RightArm":      "tomato",
    "RightForeArm":  "tomato",
    "RightHand":     "red",          # FR foot
    # Rear legs
    "LeftUpLeg":     "deepskyblue",
    "LeftLeg":       "deepskyblue",
    "LeftFoot":      "cyan",         # RL foot
    "RightUpLeg":    "salmon",
    "RightLeg":      "salmon",
    "RightFoot":     "orange",       # RR foot
}

# Foot joint names (highlighted as larger dots)
FOOT_JOINTS = {"LeftHand", "RightHand", "LeftFoot", "RightFoot"}


def load_skeleton(bvh_path: str, unit_scale: float = 0.01):
    """
    Load all joint world positions for all frames.
    Returns:
        joint_names: list of str
        positions:   dict {joint_name: np.ndarray [T, 3]}  in metres, Z-up
    """
    root = bvhio.readAsHierarchy(bvh_path)
    joints_layout = list(root.layout())
    joint_names   = [j.Name for j, _, _ in joints_layout]
    n_frames      = len(root.Keyframes)

    print(f"  Joints: {len(joint_names)}")
    print(f"  Frames: {n_frames}")

    # Pre-allocate
    positions = {name: np.zeros((n_frames, 3), dtype=np.float32)
                 for name in joint_names}

    # Build a fast lookup: name → joint object
    joint_map = {j.Name: j for j, _, _ in joints_layout}

    for i in range(n_frames):
        root.loadPose(i)
        for name, joint in joint_map.items():
            p = np.array(joint.PositionWorld, dtype=np.float32)
            # Unit conversion (cm → m)
            p = p * unit_scale
            # Coordinate conversion: bvhio Y-up → Genesis Z-up
            # X stays, Y→-Z (forward), Z→Y (up... wait, bvhio is Y-up)
            # BVH: X=right, Y=up, Z=back(negative forward)
            # Genesis: X=right, Y=forward, Z=up
            positions[name][i] = np.array([
                 p[0],    # X unchanged (lateral)
                -p[2],    # BVH -Z → Genesis Y (forward)
                 p[1],    # BVH Y → Genesis Z (up)
            ])

        if i % 100 == 0:
            print(f"  Loaded frame {i}/{n_frames}", end="\r")

    print(f"  Loaded {n_frames} frames OK          ")
    return joint_names, positions, n_frames


def render_video(
    bvh_path:   str,
    output:     str = None,
    unit_scale: float = 0.01,
    fps:        float = 30.0,
    max_frames: int   = None,
    speed:      float = 1.0,
):
    print(f"\nLoading: {bvh_path}")
    joint_names, positions, n_frames = load_skeleton(bvh_path, unit_scale)

    if max_frames:
        n_frames = min(n_frames, max_frames)

    # ── Compute scene bounds from all frames ───────────────────────────
    all_pos = np.concatenate(list(positions.values()), axis=0)
    # Use root position range for centering
    root_pos = positions["Hips"]
    cx = root_pos[:n_frames, 0].mean()
    cy = root_pos[:n_frames, 1].mean()
    cz = root_pos[:n_frames, 2].mean()

    spread = 0.6   # metres — half-width of view
    xlim = (cx - spread, cx + spread)
    ylim = (cy - spread, cy + spread)
    zlim = (0.0, cz + spread)   # floor at 0

    # ── Figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 7), facecolor="white")
    ax  = fig.add_subplot(111, projection="3d")

    ax.set_facecolor("white")
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)
    ax.set_zlabel("Z (m)", fontsize=9)
    ax.set_title(f"Dog MoCap — {Path(bvh_path).name}", fontsize=10)

    # Ground grid
    grid_x = np.linspace(xlim[0], xlim[1], 5)
    grid_y = np.linspace(ylim[0], ylim[1], 5)
    for gx in grid_x:
        ax.plot([gx, gx], [ylim[0], ylim[1]], [0, 0],
                color="lightgray", linewidth=0.5, alpha=0.5)
    for gy in grid_y:
        ax.plot([xlim[0], xlim[1]], [gy, gy], [0, 0],
                color="lightgray", linewidth=0.5, alpha=0.5)

    # ── Create animated elements ──────────────────────────────────────
    bone_lines = {}
    for parent_name, child_name in BONES:
        if parent_name in positions and child_name in positions:
            color = JOINT_COLORS.get(child_name, "gray")
            line, = ax.plot([], [], [], "-",
                            color=color, linewidth=2.5, alpha=0.85)
            bone_lines[(parent_name, child_name)] = line

    joint_dots = {}
    for name in joint_names:
        if name in positions:
            color    = JOINT_COLORS.get(name, "gray")
            is_foot  = name in FOOT_JOINTS
            size     = 60 if is_foot else 20
            marker   = "o"
            dot = ax.scatter([], [], [],
                             c=color, s=size, marker=marker,
                             depthshade=False, zorder=5)
            joint_dots[name] = dot

    info_text = ax.text2D(
        0.02, 0.95, "", transform=ax.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    legend_items = [
        plt.Line2D([0], [0], color="blue",   linewidth=2, label="FL leg"),
        plt.Line2D([0], [0], color="red",    linewidth=2, label="FR leg"),
        plt.Line2D([0], [0], color="cyan",   linewidth=2, label="RL leg"),
        plt.Line2D([0], [0], color="orange", linewidth=2, label="RR leg"),
        plt.Line2D([0], [0], color="gray",   linewidth=2, label="spine/tail"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8)

    def set_axes():
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

    set_axes()
    ax.view_init(elev=20, azim=-60)

    # ── Update function ───────────────────────────────────────────────
    def update(frame_idx):
        i = int(frame_idx) % n_frames

        # Update bones
        for (pname, cname), line in bone_lines.items():
            if pname in positions and cname in positions:
                pp = positions[pname][i]
                cp = positions[cname][i]
                line.set_data([pp[0], cp[0]], [pp[1], cp[1]])
                line.set_3d_properties([pp[2], cp[2]])

        # Update joint dots
        for name, dot in joint_dots.items():
            p = positions[name][i]
            dot._offsets3d = ([p[0]], [p[1]], [p[2]])

        # Info text
        root_h = positions["Hips"][i, 2]
        info_text.set_text(
            f"Frame: {i}/{n_frames}\n"
            f"Time:  {i/fps:.2f}s\n"
            f"Hip Z: {root_h:.3f}m"
        )

        set_axes()
        return list(bone_lines.values()) + list(joint_dots.values()) + [info_text]

    # ── Animate ───────────────────────────────────────────────────────
    interval_ms = (1000.0 / fps) / speed

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if output:
        print(f"\nSaving video to: {output}")
        print("(This may take a minute...)")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        writer = animation.FFMpegWriter(
            fps=fps * speed,
            metadata={"title": Path(bvh_path).stem},
            bitrate=2000,
        )
        ani.save(output, writer=writer, dpi=120)
        print(f"Saved: {output}")
    else:
        print("\nShowing interactive viewer...")
        print("Controls: close the window to exit")
        plt.tight_layout()
        plt.show()

    return ani


def main():
    parser = argparse.ArgumentParser(
        description="Visualise dog BVH skeleton as video or interactive viewer"
    )
    parser.add_argument("bvh_file", help="Path to .bvh file")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output MP4 path. If not given, shows interactive viewer."
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Playback frame rate (default: 30)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0, try 0.25 for slow motion)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit number of frames to render (default: all)"
    )
    parser.add_argument(
        "--unit-scale", type=float, default=0.01,
        help="Unit scale factor (default: 0.01 for cm→m)"
    )
    args = parser.parse_args()

    if not Path(args.bvh_file).exists():
        print(f"File not found: {args.bvh_file}")
        sys.exit(1)

    render_video(
        bvh_path   = args.bvh_file,
        output     = args.output,
        fps        = args.fps,
        speed      = args.speed,
        max_frames = args.max_frames,
        unit_scale = args.unit_scale,
    )


if __name__ == "__main__":
    main()
