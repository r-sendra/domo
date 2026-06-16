"""
scripts/reframe_dog_data/visualize_retargeted_go2.py
-----------------------------------------------------
Render the retargeted AMP reference dataset as a Go2-proportioned
stick figure. Shows exactly what the AMP discriminator sees during
training.

Reads:  data/amp_go2_reference.npy  (output of retarget_mocap.py)
Output: interactive viewer or MP4 video

State layout (22 values per frame):
    [0:3]   FL foot position in base frame
    [3:6]   FR foot position in base frame
    [6:9]   RL foot position in base frame
    [9:12]  RR foot position in base frame
    [12:15] projected gravity vector
    [15]    base height
    [16:19] base linear velocity
    [19:22] base angular velocity

Usage:
    # Interactive
    python scripts/reframe_dog_data/visualize_retargeted_go2.py

    # Save MP4
    python scripts/reframe_dog_data/visualize_retargeted_go2.py \
        --input  data/amp_go2_reference.npy \
        --output data/go2_retargeted.mp4

    # Slow motion
    python scripts/reframe_dog_data/visualize_retargeted_go2.py \
        --speed 0.25

Requires:
    pip install numpy matplotlib
    brew install ffmpeg   # only for MP4 export
"""

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from pathlib import Path


# ── Go2 physical dimensions ────────────────────────────────────────────────
GO2_BODY_LENGTH  = 0.47   # m  front-to-rear hip distance
GO2_BODY_WIDTH   = 0.12   # m  left-right hip offset
GO2_NOMINAL_HEIGHT = 0.34  # m

# Hip positions in base frame (fixed, from Go2 URDF)
# These are where the legs attach to the body
HIP_POSITIONS = {
    "FL": np.array([ 0.19,  0.11, 0.0]),   # Front-Left
    "FR": np.array([ 0.19, -0.11, 0.0]),   # Front-Right
    "RL": np.array([-0.19,  0.11, 0.0]),   # Rear-Left
    "RR": np.array([-0.19, -0.11, 0.0]),   # Rear-Right
}

LEG_COLORS = {
    "FL": "royalblue",
    "FR": "tomato",
    "RL": "deepskyblue",
    "RR": "salmon",
}

FOOT_COLORS = {
    "FL": "blue",
    "FR": "red",
    "RL": "cyan",
    "RR": "orange",
}


def load_reference(npy_path: str):
    data = np.load(npy_path)
    assert data.shape[1] == 22, \
        f"Expected 22-dim state, got {data.shape[1]}. " \
        f"Check retarget_mocap.py output."
    return data


def render(
    npy_path:   str,
    output:     str   = None,
    fps:        float = 30.0,
    speed:      float = 1.0,
    max_frames: int   = None,
):
    print(f"\nLoading: {npy_path}")
    data = load_reference(npy_path)
    n_frames = min(len(data), max_frames) if max_frames else len(data)
    print(f"Frames: {n_frames}  ({n_frames/fps:.1f}s at {fps}Hz)")

    # Extract columns
    foot_pos_base = {
        "FL": data[:n_frames, 0:3],
        "FR": data[:n_frames, 3:6],
        "RL": data[:n_frames, 6:9],
        "RR": data[:n_frames, 9:12],
    }
    proj_gravity = data[:n_frames, 12:15]
    base_height  = data[:n_frames, 15]
    lin_vel      = data[:n_frames, 16:19]

    # ── Figure layout ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor="white")
    fig.suptitle("Go2 Retargeted Reference (AMP Dataset)", fontsize=11)

    ax3d  = fig.add_subplot(121, projection="3d")
    ax_ft = fig.add_subplot(222)   # foot heights
    ax_vel= fig.add_subplot(224)   # base velocity

    # ── 3D axes setup ──────────────────────────────────────────────────
    spread = 0.55
    ax3d.set_xlim(-spread, spread)
    ax3d.set_ylim(-spread, spread)
    ax3d.set_zlim(-0.05,   spread * 1.5)
    ax3d.set_xlabel("X →  (right)", fontsize=8)
    ax3d.set_ylabel("Y →  (forward)", fontsize=8)
    ax3d.set_zlabel("Z ↑  (up)", fontsize=8)
    ax3d.set_title("Go2 Stick Figure (base frame)", fontsize=9)
    ax3d.view_init(elev=20, azim=-55)

    # Ground plane
    gx = np.linspace(-spread, spread, 4)
    gy = np.linspace(-spread, spread, 4)
    for x in gx:
        ax3d.plot([x, x], [-spread, spread], [0, 0],
                  color="lightgray", linewidth=0.5, alpha=0.4)
    for y in gy:
        ax3d.plot([-spread, spread], [y, y], [0, 0],
                  color="lightgray", linewidth=0.5, alpha=0.4)

    # ── Static 2D plots ────────────────────────────────────────────────
    t_ax = np.arange(n_frames) / fps

    ax_ft.set_title("Foot Heights in Base Frame", fontsize=9)
    ax_ft.set_xlabel("Time (s)", fontsize=8)
    ax_ft.set_ylabel("Z (m)", fontsize=8)
    ax_ft.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for leg, color in FOOT_COLORS.items():
        ax_ft.plot(t_ax, foot_pos_base[leg][:, 2],
                   color=color, linewidth=0.7, alpha=0.8, label=leg)
    ax_ft.legend(loc="upper right", fontsize=8)
    ax_ft.grid(True, alpha=0.3)
    vline_ft = ax_ft.axvline(0, color="black", linewidth=1.0, alpha=0.6)

    ax_vel.set_title("Base Linear Velocity (base frame)", fontsize=9)
    ax_vel.set_xlabel("Time (s)", fontsize=8)
    ax_vel.set_ylabel("m/s", fontsize=8)
    ax_vel.plot(t_ax, lin_vel[:, 0], color="blue",  linewidth=0.7, label="X (lateral)")
    ax_vel.plot(t_ax, lin_vel[:, 1], color="green", linewidth=0.7, label="Y (forward)")
    ax_vel.plot(t_ax, lin_vel[:, 2], color="red",   linewidth=0.7, label="Z (vertical)")
    ax_vel.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_vel.legend(loc="upper right", fontsize=8)
    ax_vel.grid(True, alpha=0.3)
    vline_vel = ax_vel.axvline(0, color="black", linewidth=1.0, alpha=0.6)

    # ── Animated 3D elements ───────────────────────────────────────────
    # Body rectangle (4 hip positions connected)
    body_corners_base = np.array([
        HIP_POSITIONS["FL"],
        HIP_POSITIONS["FR"],
        HIP_POSITIONS["RR"],
        HIP_POSITIONS["RL"],
        HIP_POSITIONS["FL"],   # close the rectangle
    ])

    body_line, = ax3d.plot([], [], [], "k-", linewidth=3, alpha=0.8, zorder=5)

    # Base centre dot
    base_dot, = ax3d.plot([], [], [], "ko", markersize=10, zorder=6)

    # Gravity arrow — drawn each frame
    grav_arrow_container = [None]

    # Per-leg: hip dot, thigh line (hip→mid), shin line (mid→foot), foot dot
    hip_dots    = {}
    thigh_lines = {}
    shin_lines  = {}
    foot_dots   = {}
    mid_dots    = {}

    for leg in ["FL", "FR", "RL", "RR"]:
        lc = LEG_COLORS[leg]
        fc = FOOT_COLORS[leg]
        hip_dots[leg],    = ax3d.plot([], [], [], "o",
                                       color=lc, markersize=7, zorder=5)
        mid_dots[leg],    = ax3d.plot([], [], [], "o",
                                       color=lc, markersize=5, alpha=0.6, zorder=5)
        foot_dots[leg],   = ax3d.plot([], [], [], "o",
                                       color=fc, markersize=10, zorder=6)
        thigh_lines[leg], = ax3d.plot([], [], [], "-",
                                       color=lc, linewidth=2.5, alpha=0.9)
        shin_lines[leg],  = ax3d.plot([], [], [], "-",
                                       color=fc, linewidth=2.5, alpha=0.9)

    info_text = ax3d.text2D(
        0.02, 0.96, "", transform=ax3d.transAxes, fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    )

    legend_items = [
        plt.Line2D([0], [0], color="royalblue", linewidth=2, label="FL (front-left)"),
        plt.Line2D([0], [0], color="tomato",    linewidth=2, label="FR (front-right)"),
        plt.Line2D([0], [0], color="deepskyblue", linewidth=2, label="RL (rear-left)"),
        plt.Line2D([0], [0], color="salmon",    linewidth=2, label="RR (rear-right)"),
    ]
    ax3d.legend(handles=legend_items, loc="upper right", fontsize=8)

    # ── Update function ────────────────────────────────────────────────
    def update(i):
        i = int(i) % n_frames
        bh = float(base_height[i])

        # Base is at (0, 0, bh) in world frame
        # All foot positions are given in base frame → just offset by bh in Z
        base_world = np.array([0.0, 0.0, bh])

        # Body rectangle
        body_world = body_corners_base.copy()
        body_world[:, 2] += bh
        body_line.set_data(body_world[:, 0], body_world[:, 1])
        body_line.set_3d_properties(body_world[:, 2])

        # Base centre
        base_dot.set_data([0.0], [0.0])
        base_dot.set_3d_properties([bh])

        # Gravity arrow
        if grav_arrow_container[0] is not None:
            try:
                grav_arrow_container[0].remove()
            except Exception:
                pass
        grav = proj_gravity[i] * 0.15
        grav_arrow_container[0] = ax3d.quiver(
            0, 0, bh,
            grav[0], grav[1], grav[2],
            color="purple", linewidth=1.5,
            arrow_length_ratio=0.3, alpha=0.8
        )

        # Each leg
        for leg in ["FL", "FR", "RL", "RR"]:
            hip_base   = HIP_POSITIONS[leg]
            hip_world  = hip_base + base_world

            foot_base  = foot_pos_base[leg][i]
            foot_world = foot_base + base_world

            # Midpoint approximation (knee/elbow) — halfway between hip and foot
            mid_world = (hip_world + foot_world) / 2.0

            # Hip dot
            hip_dots[leg].set_data([hip_world[0]], [hip_world[1]])
            hip_dots[leg].set_3d_properties([hip_world[2]])

            # Knee dot
            mid_dots[leg].set_data([mid_world[0]], [mid_world[1]])
            mid_dots[leg].set_3d_properties([mid_world[2]])

            # Foot dot
            foot_dots[leg].set_data([foot_world[0]], [foot_world[1]])
            foot_dots[leg].set_3d_properties([foot_world[2]])

            # Thigh: hip → knee
            thigh_lines[leg].set_data(
                [hip_world[0], mid_world[0]],
                [hip_world[1], mid_world[1]]
            )
            thigh_lines[leg].set_3d_properties([hip_world[2], mid_world[2]])

            # Shin: knee → foot
            shin_lines[leg].set_data(
                [mid_world[0], foot_world[0]],
                [mid_world[1], foot_world[1]]
            )
            shin_lines[leg].set_3d_properties([mid_world[2], foot_world[2]])

        # Time cursors
        t_cur = i / fps
        vline_ft.set_xdata([t_cur, t_cur])
        vline_vel.set_xdata([t_cur, t_cur])

        # Info
        fz = {leg: float(foot_pos_base[leg][i, 2]) for leg in ["FL","FR","RL","RR"]}
        info_text.set_text(
            f"Frame: {i}/{n_frames}  t={t_cur:.2f}s\n"
            f"Base H: {bh:.3f}m\n"
            f"FL: {fz['FL']:+.3f}m  FR: {fz['FR']:+.3f}m\n"
            f"RL: {fz['RL']:+.3f}m  RR: {fz['RR']:+.3f}m"
        )

        return (
            [body_line, base_dot, info_text, vline_ft, vline_vel]
            + list(hip_dots.values())
            + list(mid_dots.values())
            + list(foot_dots.values())
            + list(thigh_lines.values())
            + list(shin_lines.values())
        )

    # ── Run animation ──────────────────────────────────────────────────
    interval_ms = (1000.0 / fps) / speed

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()

    if output:
        print(f"\nSaving to: {output}  (may take a minute...)")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        writer = animation.FFMpegWriter(
            fps        = fps * speed,
            metadata   = {"title": "Go2 AMP Reference"},
            bitrate    = 2000,
            extra_args = ["-vcodec", "libx264"],
        )
        ani.save(output, writer=writer, dpi=120)
        print(f"Saved: {output}")
    else:
        print("\nShowing interactive viewer — close window to exit")
        plt.show()

    return ani


def main():
    parser = argparse.ArgumentParser(
        description="Visualise retargeted Go2 AMP reference dataset"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/amp_go2_reference.npy",
        help="Path to amp_go2_reference.npy (default: data/amp_go2_reference.npy)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output MP4 path. If not given, shows interactive viewer."
    )
    parser.add_argument(
        "--fps",   type=float, default=30.0,
        help="Frame rate (default: 30)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed (default: 1.0, try 0.25 for slow motion)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit frames rendered (default: all)"
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"File not found: {args.input}")
        print("Run retarget_mocap.py first.")
        sys.exit(1)

    render(
        npy_path   = args.input,
        output     = args.output,
        fps        = args.fps,
        speed      = args.speed,
        max_frames = args.max_frames,
    )


if __name__ == "__main__":
    main()
