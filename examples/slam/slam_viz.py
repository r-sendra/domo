"""
Shared visualisation helpers for the SLAM examples:
  * `slam_snapshot`     — a small JSON-serialisable telemetry dict for the
                          live dashboard (pose, velocity, map, cloud, lidar).
  * `go2_robot_manifest`— the 3D-viewer manifest entry + asset root for the
                          Go2 URDF (served by the dashboard under /assets).
  * `save_point_cloud`  — a static 3D point-cloud scatter PNG.
  * `SlamLiveView`      — a LIVE matplotlib window that redraws the
                          reconstruction as the Genesis sim steps (run Genesis
                          headless; this is the separate animated plot).
  * `SlamRecorder`      — collects frames and writes a GIF at the end.

Backend note: the *examples* pick the matplotlib backend once at startup
(Agg when only saving; a GUI backend when `--live`). These helpers never call
`matplotlib.use()`, so both paths coexist. matplotlib itself is imported
lazily inside the plotting helpers.

Kept out of the library (domo/) because it pulls in matplotlib — it is
example-side tooling. Both slam_demo.py and slam_replica_house.py import it
(`from slam_viz import …` works because the script's directory is on
sys.path). It holds no web code, so importing it never loads the dashboard.
"""

from __future__ import annotations

import contextlib
import os

import torch

MAP_ROWS = 60               # occupancy grid is max-pooled down to ~this many rows


def slam_snapshot(step, state, slam, lidar=None, cloud_pts=500, status="running"):
    """
    Build a small, JSON-serialisable telemetry dict for the live dashboard —
    pose/velocity/height/coverage, lidar sectors, a downsampled occupancy map,
    and a subsampled top-down point cloud. Cheap (vectorised max-pool + a few
    `.tolist()`); safe to call from the sim loop at a few Hz.

    Keys match the dashboard page's expectations (see domo/dashboard.py):
    t, status, pose, base, dof, vel, height, coverage, lidar, map, cloud, bounds.
    """
    snap = {
        "t": int(step),
        "status": status,
        "pose": [round(float(state.base_pos[0, 0]), 2),
                 round(float(state.base_pos[0, 1]), 2),
                 round(float(state.base_euler[0, 2]), 2)],
        # Full base pose (x,y,z, qw,qx,qy,qz) + joint angles for the 3D viewer.
        "base": [round(float(state.base_pos[0, 0]), 3),
                 round(float(state.base_pos[0, 1]), 3),
                 round(float(state.base_pos[0, 2]), 3)]
                + [round(float(v), 4) for v in state.base_quat[0].tolist()],
        "dof": [round(float(v), 4) for v in state.dof_pos[0].tolist()],
        "vel": [round(float(state.base_lin_vel[0, 0]), 2),
                round(float(state.base_lin_vel[0, 1]), 2),
                round(float(state.base_ang_vel[0, 2]), 2)],
        "height": round(float(state.base_pos[0, 2]), 3),
        "coverage": round(float(slam.coverage()[0]) * 100, 1),
    }
    if lidar is not None:
        # Before the first scan a backend may have no buffer to read; the
        # exact exception is backend-specific, so any failure just omits the
        # lidar panel for this snapshot.
        with contextlib.suppress(Exception):
            snap["lidar"] = [round(float(v), 2) for v in lidar.read()[0].tolist()]

    snap["map"] = _occupancy_rows(slam)

    cloud = slam.point_cloud()
    if cloud.shape[0] > 0:
        if cloud.shape[0] > cloud_pts:
            cloud = cloud[torch.randperm(cloud.shape[0])[:cloud_pts]]
        snap["cloud"] = [[round(x, 2), round(y, 2), round(z, 2)]
                         for x, y, z in cloud.tolist()]
        c = slam.cfg
        snap["bounds"] = [c.origin[0] - c.half_extent, c.origin[0] + c.half_extent,
                          c.origin[1] - c.half_extent, c.origin[1] + c.half_extent]
    return snap


def _occupancy_rows(slam) -> list[str]:
    """Max-pooled occupancy → rows of '#' (occupied), '.' (free), ' ' (unknown)."""
    g = slam.grid[0]
    n = slam.n_cells
    st = max(1, n // MAP_ROWS)
    H = (n // st) * st
    gp = g[:H, :H].reshape(H // st, st, H // st, st).amax(dim=(1, 3))
    occ = (torch.sigmoid(gp) > slam.cfg.occ_threshold).tolist()
    free = (gp < 0.0).tolist()
    return ["".join("#" if occ[r][c] else "." if free[r][c] else " "
                    for c in range(len(occ[0])))
            for r in range(len(occ))]


def go2_robot_manifest(spec):
    """
    Scene-manifest entry + asset root for rendering a robot's URDF in the 3D
    dashboard viewer: served under /assets/robot/, joints in dof order.
    Returns (robot_dict, root_dir).
    """
    urdf = spec.urdf_path
    if not os.path.isabs(urdf):
        import genesis  # heavy: only to locate its bundled asset dir
        urdf = os.path.join(os.path.dirname(genesis.__file__), "assets", urdf)
    urdf = os.path.abspath(urdf)
    root = os.path.dirname(os.path.dirname(urdf))       # <robot>/ (urdf/ + meshes/)
    rel = os.path.relpath(urdf, root).replace(os.sep, "/")
    return {"urdf": "/assets/robot/" + rel,
            "dof_names": list(spec.joint_names)}, root


def _subsample(cloud: torch.Tensor, max_pts: int) -> torch.Tensor:
    """Random subset of at most `max_pts` rows (plots stay responsive)."""
    if cloud.shape[0] > max_pts:
        return cloud[torch.randperm(cloud.shape[0])[:max_pts]]
    return cloud


def save_point_cloud(cloud, path, title="SLAM point cloud"):
    """3D scatter of the accumulated world-frame cloud, coloured by height."""
    if cloud.shape[0] == 0:
        print("  [viz] empty point cloud — nothing to plot")
        return
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (register 3d)
    c = cloud.cpu().numpy()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    s = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=c[:, 2], cmap="viridis",
                   s=1.0, linewidths=0)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"{title} ({len(c):,} points)")
    with contextlib.suppress(AttributeError, ValueError):   # older matplotlib
        ax.set_box_aspect((1, 1, 0.35))
    fig.colorbar(s, ax=ax, label="height z (m)", shrink=0.5, pad=0.1)
    ax.view_init(elev=38, azim=-60)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  [viz] saved 3D point cloud → {path}")


class SlamLiveView:
    """
    A LIVE, separate matplotlib window that redraws SLAM's reconstruction as
    the Genesis sim advances — a top-down view of the accumulating point cloud
    (coloured by height) with the robot marker + trail. Call `update(slam,
    state)` every control step; it refreshes every `every` steps.

    Intended combo: run Genesis HEADLESS (no Genesis window) and watch this
    window fill in — one GUI, so no toolkit clashes. The hosting example must
    have selected an interactive matplotlib backend (i.e. NOT forced Agg);
    construction fails gracefully (returns a no-op) if no display is available.
    """

    def __init__(self, every: int = 40, max_pts: int = 6000,
                 title: str = "SLAM reconstruction (live)"):
        self.ok = False
        self.every = every
        self.max_pts = max_pts
        self.title = title
        self._t = 0
        self.trail_x, self.trail_y = [], []
        import matplotlib
        if matplotlib.get_backend().lower() == "agg":
            print("  [viz] --live needs a display, but matplotlib is on the "
                  "non-GUI 'agg' backend — skipping the live window.")
            return
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.ax.set_aspect("equal")
            self.ax.set_title(title)
            self.fig.canvas.manager.set_window_title("DOMO SLAM — live")
            self.fig.show()
            self.plt.pause(0.001)
            self.ok = True
        except Exception as e:
            # Which exception a missing display raises depends on the GUI
            # toolkit (Tk, Qt, macOS); all of them mean "no live window".
            print(f"  [viz] live view unavailable ({type(e).__name__}: {e}); "
                  f"continuing without it")

    def update(self, slam, state) -> None:
        if not self.ok:
            return
        self._t += 1
        if self._t % self.every != 0:
            return
        cloud = slam.point_cloud()
        if cloud.shape[0] == 0:
            return
        c = _subsample(cloud, self.max_pts).cpu().numpy()
        rx, ry = float(state.base_pos[0, 0]), float(state.base_pos[0, 1])
        self.trail_x.append(rx)
        self.trail_y.append(ry)
        self.ax.clear()
        self.ax.scatter(c[:, 0], c[:, 1], c=c[:, 2], cmap="viridis",
                        s=2, linewidths=0)
        self.ax.plot(self.trail_x, self.trail_y, "-", color="red",
                     lw=1.0, alpha=0.6)
        self.ax.plot(rx, ry, "^", color="red", ms=10)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.set_title(f"{self.title} — step {self._t}  ({len(c):,} pts)")
        try:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.001)                    # pump the GUI event loop
        except Exception:
            self.ok = False        # window closed by the user → stop drawing (toolkit-specific error)

    def keep_open(self) -> None:
        """Block at the end so the final reconstruction stays on screen."""
        if not self.ok:
            return
        self.plt.ioff()
        print("  [viz] close the live window to exit.")
        self.plt.show()


class SlamRecorder:
    """
    Snapshots SLAM's accumulating point cloud during a run and writes a
    top-down animation (GIF) showing the map forming in real time — points
    appear as the robot drives, coloured by height, with the robot position
    marked. Call `capture(slam, state)` every control step; `save_gif(path)`
    at the end. Enable it only when requested (it holds frames in memory).
    """

    def __init__(self, every: int = 250, max_pts: int = 15000):
        self.every = every
        self.max_pts = max_pts
        self.frames = []          # list of (points[N,3], (rx, ry), step)
        self._t = 0

    def capture(self, slam, state) -> None:
        self._t += 1
        if self._t % self.every != 0:
            return
        cloud = slam.point_cloud()
        if cloud.shape[0] == 0:
            return
        self.frames.append((_subsample(cloud, self.max_pts).cpu().numpy(),
                            (float(state.base_pos[0, 0]),
                             float(state.base_pos[0, 1])),
                            self._t))

    def save_gif(self, path: str, fps: int = 6,
                 title: str = "SLAM point cloud forming") -> None:
        if not self.frames:
            print("  [viz] no frames captured — animation skipped")
            return
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.animation import FuncAnimation, PillowWriter

        # Fixed axes/colour limits across frames so the animation doesn't jump.
        allpts = np.concatenate([f[0] for f in self.frames], axis=0)
        xlim = (allpts[:, 0].min(), allpts[:, 0].max())
        ylim = (allpts[:, 1].min(), allpts[:, 1].max())
        zlo, zhi = float(allpts[:, 2].min()), float(allpts[:, 2].max())
        trail_x, trail_y = [], []

        fig, ax = plt.subplots(figsize=(7, 7))

        def draw(i):
            ax.clear()
            pts, (rx, ry), step = self.frames[i]
            trail_x.append(rx)
            trail_y.append(ry)
            ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap="viridis",
                       s=2, vmin=zlo, vmax=zhi, linewidths=0)
            ax.plot(trail_x, trail_y, "-", color="red", lw=1.0, alpha=0.6)
            ax.plot(rx, ry, "^", color="red", ms=10)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(f"{title} — step {step}  ({len(pts):,} pts)")

        anim = FuncAnimation(fig, draw, frames=len(self.frames),
                             interval=1000 / fps)
        anim.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"  [viz] saved animation ({len(self.frames)} frames) → {path}")
