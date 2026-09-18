"""
Engine-free World test: a fake backend registered through
`domo.sim.register_backend` must be enough to stand up the full twin
(engine → scene → arena → robot → lidar → build → bind → device-model lidar).
This is the guarantee that the physics engine stays swappable.
"""

import pytest
import torch
from test_foundation_robot import FakeArticulation

from domo.robot import SimulatedLidar
from domo.robot.lidar_models import hesai_xt16
from domo.scenes import ObstacleArena
from domo.sim import (
    LidarConfig,
    LidarSensorHandle,
    PhysicsEngine,
    RigidObject,
    Scene,
    SimConfig,
    register_backend,
)
from domo.world import World, WorldConfig


class FakeRigidObject(RigidObject):
    def __init__(self):
        self.pos = None

    def set_position(self, pos, envs_idx=None):
        self.pos = pos.clone()


class FakeLidarHandle(LidarSensorHandle):
    """Constant 2 m field, already in the contract layout."""

    def __init__(self, cfg: LidarConfig, n_envs: int):
        self.config = cfg
        self.n_envs = n_envs

    def read_ranges(self):
        return torch.full((self.n_envs, self.config.n_vertical, self.config.n_horizontal), 2.0)

    def read_sector_distances(self):
        return self.read_ranges().min(dim=1).values


class FakeScene(Scene):
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.n_envs = 0
        self.ground = []
        self.bodies = []
        self.articulation = None
        self.lidar = None
        self.built = False

    def _body(self):
        if self.built:
            raise RuntimeError("entities must be added before build()")
        b = FakeRigidObject()
        self.bodies.append(b)
        return b

    def add_ground(self, height=0.0):
        self.ground.append(height)

    def add_terrain(self, cfg): ...
    def add_mesh(self, *a, **k): return self._body()
    def add_urdf_prop(self, *a, **k): return self._body()
    def add_box(self, size, pos, fixed=True): return self._body()
    def add_cylinder(self, radius, height, pos, fixed=True): return self._body()
    def add_sphere(self, radius, pos, fixed=True): return self._body()

    def add_articulation(self, urdf_path, pos, quat_wxyz):
        self.spawn = (pos, quat_wxyz)
        self.articulation = FakeArticulation(0, 12)     # resized in build()
        return self.articulation

    def add_lidar(self, articulation, cfg):
        assert articulation is self.articulation
        self.lidar = FakeLidarHandle(cfg, 0)
        return self.lidar

    def build(self, n_envs):
        self.built = True
        self.n_envs = n_envs
        self.articulation.__init__(n_envs, 12)
        if self.lidar is not None:
            self.lidar.n_envs = n_envs

    def step(self): ...


class FakeEngine(PhysicsEngine):
    name = "fake"

    def __init__(self, device="cpu"):
        self._device = torch.device("cpu")
        self.scenes = []

    @property
    def device(self):
        return self._device

    def create_scene(self, cfg):
        scene = FakeScene(cfg)
        self.scenes.append(scene)
        return scene


register_backend("fake-world-test", lambda **kw: FakeEngine(**kw))


def _world(**kw):
    cfg = WorldConfig(engine="fake-world-test", device="cpu", headless=True, **kw)
    return World(cfg, n_envs=2)


def test_arena_world_composes_all_layers():
    w = _world(scene_kind="arena", base_init_yaw_deg=90.0)
    scene = w.scene
    assert scene.built and scene.n_envs == 2 and w.robot.n_envs == 2
    assert scene.ground == [0.0]
    assert isinstance(w.arena, ObstacleArena)
    a = w.cfg.arena
    n_expected = 4 + 4 * a.n_chairs + a.n_sofas + a.n_pillars + a.n_steps + a.n_balls
    assert len(scene.bodies) == n_expected
    # Spawn pose: yaw 90° about z as a wxyz quaternion.
    pos, quat = scene.spawn
    assert pos == (0.0, 0.0, 0.35)
    assert quat[0] == pytest.approx(quat[3]) and quat[1] == quat[2] == 0.0
    # Lidar: device model wraps the backend handle with the world's dt.
    assert isinstance(w.lidar, SimulatedLidar) and w.sensors == [w.lidar]
    assert w.lidar.update_interval == 5                      # 10 Hz @ 50 Hz
    assert w.camera is None


def test_lidar_reads_flow_through_device_model():
    w = _world(scene_kind="flat", lidar_model=hesai_xt16(n_horizontal=72))
    assert w.arena is None and w.lidar.n_sectors == 36
    for _ in range(w.lidar.update_interval):
        w.lidar.tick()
    # Noise is ~1 cm so sectors sit near the 2 m fill.
    assert (w.lidar.read() - 2.0).abs().max() < 0.1


def test_no_lidar_and_randomise_is_noop_outside_arena():
    w = _world(scene_kind="flat", lidar_model=None)
    assert w.lidar is None and w.sensors == []
    w.randomise_obstacles()                     # arena None → no-op


def test_randomise_obstacles_places_bodies_in_ring():
    torch.manual_seed(0)
    w = _world(scene_kind="arena")
    w.randomise_obstacles()
    walls, obstacles = w.scene.bodies[:4], w.scene.bodies[4:]
    assert all(b.pos is None for b in walls)
    for b in obstacles:
        assert b.pos.shape == (2, 3)
        r = b.pos[:, :2].norm(dim=1)
        # Chair legs sit up to `chair_leg_spread*sqrt2` off the chair centre.
        slack = w.cfg.arena.chair_leg_spread * 1.5
        assert (r >= w.cfg.arena.ring_min - slack).all()
        assert (r <= w.cfg.arena.ring_max + slack).all()
    w.randomise_obstacles(torch.tensor([1]))    # subset call also fine


def test_reset_robot_restores_spawn_and_clears_lidar():
    w = _world(scene_kind="arena")
    art = w.scene.articulation
    art.pos[:] = 5.0
    for _ in range(w.lidar.update_interval):
        w.lidar.tick()
    assert (w.lidar.read() == 2.0).all()
    w.reset_robot()
    assert torch.allclose(w.robot.state.base_pos, torch.tensor([[0.0, 0.0, 0.35]] * 2))
    assert (w.lidar.read() == w.lidar.max_range).all()


def test_unknown_scene_kind_raises():
    with pytest.raises(ValueError, match="unknown scene_kind"):
        _world(scene_kind="moon")


def test_make_loop_wires_control_loop():
    from domo.control import SimControlLoop

    class Hold:
        def setup(self, robot): ...

        def update(self, state, dt):
            return torch.zeros(state.dof_pos.shape)

    w = _world(scene_kind="flat")
    loop = w.make_loop(Hold())
    assert isinstance(loop, SimControlLoop)
