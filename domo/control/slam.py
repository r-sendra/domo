"""
SLAM as a perception LAYER of the control hierarchy.

`SlamSkill` is a command skill whose velocity contribution is always zero — it
observes and maps but never steers, so it layers on top of any driving stack
(`slam @ avoid @ goto(...) @ walk`). It folds the lidar + pose into a log-odds
occupancy grid and accumulates a 3D point cloud. Pure torch; engine-free (it
only reads a RobotState and a lidar's read()/read_points()).

Kept in its own module (not skill.py) because it is a self-contained
perception subsystem with its own grid, cloud, and localisation-seam code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .skill import CommandSkill, planar_pose

__all__ = ["SlamConfig", "SlamSkill"]

# Returns closer than this (m, planar) are the robot's own body/legs — never
# mapped as obstacles.
_MIN_HIT_DIST = 0.05
# |log-odds| above which a cell counts as "known" for coverage().
_KNOWN_LOG_ODDS = 0.5
# Target width (columns) of the default render_ascii() downsample.
_ASCII_COLS = 48


@dataclass
class SlamConfig:
    """Occupancy-grid + localisation parameters for SlamSkill.

    Log-odds model: each cell starts at 0 (P = 0.5, unknown); hits add
    `l_occ`, free-space samples add `l_free`, and |value| is clamped to
    `l_clamp`. P(occupied) = sigmoid(log-odds).
    """
    resolution: float = 0.15             # metres per map cell
    half_extent: float = 8.0             # map spans origin ± this (m)
    origin: tuple[float, float] = (0.0, 0.0)   # world centre of the grid
    update_interval: int = 5             # control ticks between scan integrations
    l_occ: float = 0.85                  # log-odds added to a hit cell
    l_free: float = -0.4                 # log-odds added to a free (ray) cell
    l_clamp: float = 6.0                 # clamp on |log-odds| (bounds confidence)
    sticky_occ: float = 0.5              # a cell already this occupied-confident
                                         # is NOT carved free — stops beams that
                                         # skim over short walls (in the 2D
                                         # projection) from erasing them
    occ_first: bool = True               # hits are applied before free-carving
                                         # (the only order implemented; kept
                                         # for config compatibility)
    occ_threshold: float = 0.65          # P(occupied) above → "occupied"
    free_threshold: float = 0.35         # P(occupied) below → "free"
    # High-fidelity mode: consume the lidar's raw 3D world points (if exposed)
    # instead of pooled 2D sectors. Maps a robot-height slice into the grid and
    # accumulates the full 3D cloud for visualisation.
    use_points: bool = True              # use read_points() when available
    z_band: tuple[float, float] = (0.15, 1.5)  # height slice → 2D occupancy (m)
    map_max_range: float = 10.0          # cap mapping range (device range >> map)
    cloud_max_points: int = 150000       # safety cap on the accumulated cloud
    cloud_stride: int = 2                # keep 1/stride of each scan before dedup
    cloud_voxel: float = 0.05            # voxel size for uniform-density dedup (m)


class SlamSkill(CommandSkill):
    """
    Passive SLAM as a PERCEPTION layer — it maps, it does not move the robot.

    SlamSkill is a command skill whose velocity contribution is always ZERO
    (additive, delta = 0). It therefore *layers on top of* a driving stack
    without ever affecting it:

        slam @ avoid @ goto(x=8, y=5) @ walk

    Right-associative, so the motor skill (walk) stays at the bottom; `goto`
    authors the goal-seeking velocity, `avoid` corrects it around obstacles,
    and `slam` sits on top observing. It reads the lidar and the robot's pose
    each tick and folds them into a world-frame occupancy grid via the standard
    log-odds inverse-sensor model (cells along each beam → free, the beam's hit
    → occupied). It never does locomotion or avoidance itself.

    Localisation seam: mapping uses `state.base_pos`/`base_euler` — the same
    pose the rest of the stack trusts (sim ground truth / odometry / mocap /
    the robot's own estimator). A pure-odometry pose is integrated alongside
    and exposed as `odometry_drift()`, marking exactly where a real system
    would fuse scan-matching to correct that drift (not done here — this
    implements the Mapping half against a given pose estimate).

    Query the result with `occupancy_prob()`, `coverage()`, `render_ascii()`,
    and the accumulated 3D cloud with `point_cloud()`.
    """

    name = "slam"
    channel = "velocity"
    additive = True                 # contributes zero, so it only observes

    def __init__(self, lidar, cfg: SlamConfig | None = None):
        """
        Args:
            lidar: sensor with read() → [N, n_sectors] plus `n_sectors` and
                `max_range` attributes; optionally read_points() →
                ([N, P, 3] world points, [N, P] validity) for point mode.
            cfg: grid / cloud parameters.
        """
        self.lidar = lidar
        self.cfg = cfg or SlamConfig()

    def setup(self, robot) -> None:
        super().setup(robot)
        c = self.cfg
        n, dev = robot.n_envs, robot.device
        self.n_cells = round(2 * c.half_extent / c.resolution)
        # Per-env log-odds occupancy grid [N, rows(y), cols(x)]; 0 = unknown.
        self.grid = torch.zeros(n, self.n_cells, self.n_cells, device=dev)
        self._zero = torch.zeros(n, 3, device=dev)
        k = self.lidar.n_sectors
        # Sector centre bearings in the body frame (uniform 360° sectors).
        self._bearings = (torch.arange(k, device=dev) + 0.5) * (2 * math.pi / k)
        self._max_range = float(self.lidar.max_range)
        # High-fidelity 3D-points mode when the lidar exposes per-beam points.
        self._use_points = c.use_points and hasattr(self.lidar, "read_points")
        # Distances sampled along every beam for free-space carving [S]; one
        # sample per cell so no cell along a ray is skipped.
        ray_max = c.map_max_range if self._use_points else self._max_range
        self._n_samp = int(ray_max / c.resolution) + 1
        self._ray_d = torch.arange(self._n_samp, device=dev) * c.resolution
        self._cloud = None               # accumulated world point cloud (env 0)
        self._step = 0
        # Odometry-only pose estimate (the 'localisation' concept), lazily
        # seeded from the first observed pose.
        self._odom = torch.zeros(n, 2, device=dev)
        self._need_init = torch.ones(n, dtype=torch.bool, device=dev)

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        # NOTE: LayerNode.enter() calls this, so `slam @ leg @ walk` per leg
        # wipes the map every leg. Host SLAM persistently (tick it from a
        # Controller) when the map must survive a whole mission.
        self.grid[envs_idx] = 0.0
        self._need_init[envs_idx] = True
        if 0 in envs_idx.tolist():
            self._cloud = None

    # -- the layer's control-side contribution: nothing --------------------

    def update_command(self, state, dt: float) -> torch.Tensor:
        """Integrate the scan every `update_interval` ticks; return zeros [N, 3]."""
        self._track_odometry(state, dt)
        self._step += 1
        if self._step % self.cfg.update_interval == 0:
            if self._use_points:
                self._integrate_points(state)
            else:
                self._integrate_scan(state)
        return self._zero                # never steer — pure observer

    # -- localisation seam (odometry dead-reckoning + drift readout) -------

    def _track_odometry(self, state, dt: float) -> None:
        """Dead-reckon the planar pose from body-frame velocity (no correction)."""
        xy, yaw = planar_pose(state)
        if bool(self._need_init.any()):
            m = self._need_init
            self._odom[m] = xy[m]
            self._need_init[m] = False
        v = state.base_lin_vel[:, :2]                 # body-frame planar velocity
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        vx_w = cy * v[:, 0] - sy * v[:, 1]
        vy_w = sy * v[:, 0] + cy * v[:, 1]
        self._odom[:, 0] += vx_w * dt
        self._odom[:, 1] += vy_w * dt

    def odometry_drift(self, state) -> torch.Tensor:
        """||odometry estimate − reference pose|| per env (sim diagnostic)."""
        return torch.linalg.norm(self._odom - state.base_pos[:, :2], dim=1)

    # -- mapping -----------------------------------------------------------

    def _to_cell(self, x, y):
        """World (x, y) → (col, row, in_bounds); x indexes columns, y rows.

        Out-of-range points are clamped to the edge cell, so callers must
        mask with `in_bounds` before writing.
        """
        c = self.cfg
        x0 = c.origin[0] - c.half_extent
        y0 = c.origin[1] - c.half_extent
        ci = ((x - x0) / c.resolution).long()
        ri = ((y - y0) / c.resolution).long()
        inb = (ci >= 0) & (ci < self.n_cells) & (ri >= 0) & (ri < self.n_cells)
        return ci.clamp(0, self.n_cells - 1), ri.clamp(0, self.n_cells - 1), inb

    def _mark_hits(self, grid: torch.Tensor, hx: torch.Tensor, hy: torch.Tensor,
                   valid: torch.Tensor | None = None) -> None:
        """Add l_occ to the cells under the hit points (hx, hy) [K].

        `grid` is one env's flattened log-odds [H*W]; `valid` [K] restricts
        to beams that actually returned.
        """
        ci, ri, inb = self._to_cell(hx, hy)
        mask = inb if valid is None else (valid & inb)
        idx = (ri * self.n_cells + ci)[mask]
        grid.index_add_(0, idx,
                        torch.full_like(idx, self.cfg.l_occ, dtype=grid.dtype))

    def _carve_free(self, grid: torch.Tensor, x0, y0, cos_b: torch.Tensor,
                    sin_b: torch.Tensor, ranges: torch.Tensor) -> None:
        """Add l_free along each beam strictly before its hit.

        Beams start at (x0, y0) with direction (cos_b, sin_b) [K] and length
        `ranges` [K]; samples are taken every cell (`_ray_d`, [S]) up to one
        resolution short of the hit. A cell already ≥ sticky_occ is never
        carved: in the 2D projection a beam skimming OVER a short wall would
        otherwise erase the wall's own cell. Call after `_mark_hits` so the
        current scan's hits are protected too.
        """
        c = self.cfg
        d = self._ray_d
        xs = x0 + d.unsqueeze(0) * cos_b.unsqueeze(1)              # [K, S]
        ys = y0 + d.unsqueeze(0) * sin_b.unsqueeze(1)
        ci, ri, inb = self._to_cell(xs, ys)
        free = (d.unsqueeze(0) < (ranges.unsqueeze(1) - c.resolution)) & inb
        fidx = (ri * self.n_cells + ci)[free]
        fidx = fidx[grid[fidx] < c.sticky_occ]
        grid.index_add_(0, fidx,
                        torch.full_like(fidx, c.l_free, dtype=grid.dtype))

    def _integrate_scan(self, state) -> None:
        """2D update from sector minima: one beam per sector at its centre bearing."""
        c = self.cfg
        ranges = self.lidar.read().clamp(max=self._max_range)      # [N, K]
        xy, yaw = planar_pose(state)
        px, py = xy[:, 0], xy[:, 1]
        bear = yaw.unsqueeze(1) + self._bearings.unsqueeze(0)       # [N, K]
        cos_b, sin_b = torch.cos(bear), torch.sin(bear)

        for e in range(self.grid.shape[0]):
            r = ranges[e]                                          # [K]
            grid = self.grid[e].view(-1)
            # Occupied FIRST: the hit cell, only for beams that returned.
            self._mark_hits(grid, px[e] + r * cos_b[e], py[e] + r * sin_b[e],
                            valid=r < self._max_range)
            self._carve_free(grid, px[e], py[e], cos_b[e], sin_b[e], r)

        self.grid.clamp_(-c.l_clamp, c.l_clamp)

    def _integrate_points(self, state) -> None:
        """High-fidelity update from raw 3D world points: a robot-height slice
        drives the 2D occupancy grid, and every hit feeds the 3D cloud."""
        c = self.cfg
        try:
            pts, valid = self.lidar.read_points()          # [N,P,3], [N,P]
        except NotImplementedError:
            # The handle advertised read_points but cannot serve it: drop to
            # sector mode for good (no per-tick retry cost).
            self._use_points = False
            return self._integrate_scan(state)
        xy, _ = planar_pose(state)
        px, py = xy[:, 0], xy[:, 1]
        for e in range(self.grid.shape[0]):
            v = valid[e]
            if not bool(v.any()):
                continue
            p = pts[e][v]                                   # [M, 3] world hits
            dxy = torch.hypot(p[:, 0] - px[e], p[:, 1] - py[e])
            keep = dxy < c.map_max_range
            p, dxy = p[keep], dxy[keep]
            if p.numel() == 0:
                continue
            self._accumulate_cloud(e, p)

            # Robot-height slice: near-horizontal returns off walls/obstacles
            # (excludes the floor below and anything overhead).
            band = ((p[:, 2] > c.z_band[0]) & (p[:, 2] < c.z_band[1])
                    & (dxy > _MIN_HIT_DIST))
            pb, db = p[band], dxy[band]
            if pb.numel() == 0:
                continue
            grid = self.grid[e].view(-1)

            # Mark hits occupied FIRST, so the free-carve below sees them.
            self._mark_hits(grid, pb[:, 0], pb[:, 1])
            bear = torch.atan2(pb[:, 1] - py[e], pb[:, 0] - px[e])
            self._carve_free(grid, px[e], py[e], torch.cos(bear), torch.sin(bear), db)

        self.grid.clamp_(-c.l_clamp, c.l_clamp)

    def _accumulate_cloud(self, e: int, pts: torch.Tensor) -> None:
        """
        Grow the (env-0) 3D cloud with voxel-grid downsampling that keeps ONE
        ORIGINAL point per voxel (at its true, sensor-noisy position) rather
        than snapping to the voxel centre. This gives uniform density — dense
        near-robot returns are deduped, sparse far/thin returns are preserved —
        without collapsing the cloud onto a visible lattice.
        """
        if e != 0:
            return
        c = self.cfg
        if c.cloud_stride > 1:
            pts = pts[::c.cloud_stride]
        self._cloud = pts if self._cloud is None else \
            torch.cat([self._cloud, pts], dim=0)
        # One real point per occupied voxel: bucket by integer voxel index, then
        # keep a representative original point for each (last occurrence wins).
        keys = torch.floor(self._cloud / c.cloud_voxel).to(torch.int64)
        _, inverse = torch.unique(keys, dim=0, return_inverse=True)
        rep = torch.empty(int(inverse.max()) + 1, dtype=torch.long,
                          device=self._cloud.device)
        rep[inverse] = torch.arange(inverse.shape[0], device=self._cloud.device)
        self._cloud = self._cloud[rep]
        if self._cloud.shape[0] > c.cloud_max_points:
            idx = torch.randperm(self._cloud.shape[0],
                                 device=self._cloud.device)[:c.cloud_max_points]
            self._cloud = self._cloud[idx]

    # -- queries -----------------------------------------------------------

    def point_cloud(self) -> torch.Tensor:
        """Accumulated world-frame 3D hit points [M, 3] (env 0)."""
        return self._cloud if self._cloud is not None else \
            torch.zeros(0, 3, device=self.grid.device)

    def occupancy_prob(self) -> torch.Tensor:
        """[N, H, W] probability each cell is occupied (0.5 = unknown)."""
        return torch.sigmoid(self.grid)

    def coverage(self) -> torch.Tensor:
        """Fraction of cells confidently known (free or occupied), per env [N]."""
        known = self.grid.abs() > _KNOWN_LOG_ODDS
        return known.float().mean(dim=(1, 2))

    def render_ascii(self, env: int = 0, step: int | None = None) -> str:
        """
        Coarse top-down view: '#' occupied, '.' free, ' ' unknown. Downsamples
        by MAX-pooling log-odds over each block (not point-sampling), so a
        one-cell-thin wall survives the downsample instead of aliasing away.

        Args:
            env: which env's grid to render.
            step: cells per character; default fits ~48 columns.
        """
        c = self.cfg
        g = self.grid[env]
        step = step or max(1, self.n_cells // _ASCII_COLS)
        rows = []
        for ri in range(0, self.n_cells, step):
            line = []
            for ci in range(0, self.n_cells, step):
                block = g[ri:ri + step, ci:ci + step]
                hi = float(torch.sigmoid(block.max()))   # most-occupied subcell
                if hi > c.occ_threshold:
                    line.append("#")                     # any occupied → wall
                elif float(block.max()) < 0.0:
                    line.append(".")                     # all subcells net-free
                else:
                    line.append(" ")                     # unobserved / uncertain
            rows.append("".join(line))
        return "\n".join(rows)
