"""SlamSkill mapping numerics on fakes — no physics engine.

Pins the log-odds inverse-sensor model (hit → l_occ, ray → l_free), the
`sticky_occ` no-erosion rule, the robot-height point slice, the voxel
downsampling that keeps ONE ORIGINAL point per voxel (never snapped), the
max-pooled ASCII render and the zero-velocity contract of the layer.
"""

import math

import pytest
import torch

from domo.control import SlamConfig, SlamSkill
from domo.robot import GO2
from domo.robot.state import RobotState

N = 2
DEVICE = torch.device("cpu")

# A tiny, exactly-computable map: 16 × 16 cells of 0.25 m centred on (0, 0),
# so world (0, 0) falls in cell (row 8, col 8) and the cell index of any
# point is floor((coord + 2) / 0.25).
CFG = SlamConfig(resolution=0.25, half_extent=2.0, update_interval=1,
                 use_points=False)


class FakeRobot:
    def __init__(self):
        self.spec = GO2
        self.n_envs = N
        self.device = DEVICE
        self.default_dof_pos = torch.tensor(GO2.default_dof_angles)
        self.state = RobotState.zeros(N, 12, 4, DEVICE)


class SectorLidar:
    """read() only: 8 uniform sectors, no per-beam points."""

    n_sectors = 8
    max_range = 4.0

    def __init__(self):
        self.dist = torch.full((N, self.n_sectors), self.max_range)

    def read(self):
        return self.dist


class PointLidar(SectorLidar):
    """Also exposes read_points(): world-frame hits + validity mask."""

    def __init__(self, points, valid):
        super().__init__()
        self.points = points          # [N, P, 3]
        self.valid = valid            # [N, P]

    def read_points(self):
        return self.points, self.valid


def _cell(x: float, y: float) -> tuple[int, int]:
    """(row, col) of a world point under CFG."""
    return (math.floor((y + 2.0) / 0.25), math.floor((x + 2.0) / 0.25))


def _slam(lidar, cfg=CFG):
    robot = FakeRobot()
    skill = SlamSkill(lidar, cfg)
    skill.setup(robot)
    skill.reset_idx(torch.arange(N))
    return skill, robot


# ---------------------------------------------------------------------------
# Layer contract
# ---------------------------------------------------------------------------

def test_slam_contributes_zero_velocity_and_respects_interval():
    cfg = SlamConfig(resolution=0.25, half_extent=2.0, update_interval=3,
                     use_points=False)
    skill, robot = _slam(SectorLidar(), cfg)
    for tick in range(1, 7):
        delta = skill.update_command(robot.state, 0.02)
        assert delta.shape == (N, 3) and (delta == 0).all()
        # The grid only changes on integration ticks (multiples of interval).
        touched = bool((skill.grid != 0).any())
        assert touched == (tick >= 3)
    assert skill.occupancy_prob().shape == (N, 16, 16)
    assert skill.coverage().shape == (N,)


# ---------------------------------------------------------------------------
# Sector (2D) inverse-sensor model
# ---------------------------------------------------------------------------

def test_sector_scan_marks_hit_occupied_and_ray_free():
    lidar = SectorLidar()
    lidar.dist[0, 0] = 1.0                    # one return at bearing 22.5°
    skill, robot = _slam(lidar)
    skill.update_command(robot.state, 0.02)

    bearing = 0.5 * (2 * math.pi / 8)
    hit_row, hit_col = _cell(math.cos(bearing), math.sin(bearing))
    assert skill.grid[0, hit_row, hit_col].item() == pytest.approx(CFG.l_occ)
    # Free votes on the robot's own cell: the d = 0 sample of all 8 beams,
    # plus the d = 0.25 sample of the two beams in the +x/+y quadrant
    # (0.25·cos 22.5° < 0.25 keeps them inside cell (8, 8)) → 10 votes.
    assert skill.grid[0, 8, 8].item() == pytest.approx(10 * CFG.l_free)
    # Env 1 saw no return: only free space, nothing occupied.
    assert (skill.grid[1] <= 0).all()


def test_sector_scan_never_erodes_a_confident_cell():
    lidar = SectorLidar()                      # all beams at max range
    skill, robot = _slam(lidar)
    bearing = 0.5 * (2 * math.pi / 8)
    row, col = _cell(math.cos(bearing), math.sin(bearing))  # on beam 0
    skill.grid[0, row, col] = CFG.sticky_occ           # confidently occupied
    skill.grid[1, row, col] = CFG.sticky_occ - 0.1     # not yet confident
    skill.update_command(robot.state, 0.02)
    assert skill.grid[0, row, col].item() == pytest.approx(CFG.sticky_occ)
    assert skill.grid[1, row, col].item() == pytest.approx(
        CFG.sticky_occ - 0.1 + CFG.l_free)


# ---------------------------------------------------------------------------
# Point (3D) mode
# ---------------------------------------------------------------------------

def _point_cfg(**kw):
    base = dict(resolution=0.25, half_extent=2.0, update_interval=1,
                use_points=True, cloud_stride=1, cloud_voxel=0.05)
    base.update(kw)
    return SlamConfig(**base)


def test_points_mode_slices_robot_height_into_grid():
    pts = torch.zeros(N, 3, 3)
    pts[0, 0] = torch.tensor([1.0, 0.0, 0.5])     # wall at robot height
    pts[0, 1] = torch.tensor([1.0, 0.0, 0.05])    # floor: cloud only
    pts[0, 2] = torch.tensor([0.5, 0.0, 0.5])     # invalid return
    valid = torch.tensor([[True, True, False], [False, False, False]])
    skill, robot = _slam(PointLidar(pts, valid), _point_cfg())
    skill.update_command(robot.state, 0.02)

    g = skill.grid[0]
    assert g[8, 12].item() == pytest.approx(0.85)          # hit cell
    for col in (8, 9, 10):                                # d = 0, 0.25, 0.5
        assert g[8, col].item() == pytest.approx(-0.4)
    assert g[8, 11].item() == 0.0            # within one cell of the hit
    assert (skill.grid[1] == 0).all()
    # Only the two valid points entered the cloud (env 0).
    assert skill.point_cloud().shape == (2, 3)


def test_cloud_voxel_downsampling_keeps_original_points():
    p1 = torch.tensor([1.001, 0.002, 0.5])
    p2 = torch.tensor([1.012, 0.008, 0.5])        # same 5 cm voxel as p1
    p3 = torch.tensor([1.001, 0.5, 0.5])          # different voxel
    pts = torch.stack([p1, p2, p3]).unsqueeze(0).repeat(N, 1, 1)
    valid = torch.ones(N, 3, dtype=torch.bool)
    valid[1] = False
    skill, robot = _slam(PointLidar(pts, valid), _point_cfg())
    skill.update_command(robot.state, 0.02)

    cloud = skill.point_cloud()
    assert cloud.shape == (2, 3)
    originals = torch.stack([p1, p2, p3])
    for p in cloud:
        # A real sensor point survives — never a lattice-snapped position.
        assert any(torch.equal(p, o) for o in originals)
    assert any(torch.equal(p, p3) for p in cloud)


def test_cloud_cap_and_reset():
    pts = torch.rand(N, 50, 3) + torch.tensor([0.5, 0.0, 0.3])
    valid = torch.ones(N, 50, dtype=torch.bool)
    skill, robot = _slam(PointLidar(pts, valid),
                         _point_cfg(cloud_max_points=10))
    skill.update_command(robot.state, 0.02)
    assert skill.point_cloud().shape[0] <= 10
    skill.reset_idx(torch.tensor([1]))
    assert skill.point_cloud().shape[0] > 0            # env 0 untouched
    skill.reset_idx(torch.tensor([0]))
    assert skill.point_cloud().shape == (0, 3)
    assert (skill.grid == 0).all()


def test_points_mode_falls_back_to_sectors_without_read_points():
    class NoPoints(SectorLidar):
        def read_points(self):
            raise NotImplementedError

    lidar = NoPoints()
    lidar.dist[0, 0] = 1.0
    skill, robot = _slam(lidar, _point_cfg())
    skill.update_command(robot.state, 0.02)
    bearing = 0.5 * (2 * math.pi / 8)
    row, col = _cell(math.cos(bearing), math.sin(bearing))
    assert skill.grid[0, row, col].item() == pytest.approx(0.85)
    skill.update_command(robot.state, 0.02)             # stays in sector mode
    assert skill.grid[0, row, col].item() == pytest.approx(1.7)


# ---------------------------------------------------------------------------
# Localisation seam + render
# ---------------------------------------------------------------------------

def test_odometry_dead_reckoning_tracks_body_velocity():
    skill, robot = _slam(SectorLidar())
    robot.state.base_euler[:, 2] = math.pi / 2                 # facing +y
    robot.state.base_lin_vel[:, 0] = 1.0                       # body-forward
    for _ in range(10):
        skill.update_command(robot.state, 0.1)
    # Odometry moved 1 m along +y while the reference pose stayed at origin.
    assert skill.odometry_drift(robot.state)[0].item() == pytest.approx(1.0, abs=1e-5)


def test_render_ascii_max_pools_thin_walls():
    skill, _ = _slam(SectorLidar())
    skill.grid[0, 3, 5] = 6.0                  # 1-cell wall at ODD indices
    skill.grid[0, 0:2, 0:2] = -1.0             # a known-free block
    rows = skill.render_ascii(env=0, step=2).split("\n")
    assert len(rows) == 8 and all(len(r) == 8 for r in rows)
    assert rows[1][2] == "#"                   # survives the 2× downsample
    assert rows[0][0] == "."
    assert rows[7][7] == " "
