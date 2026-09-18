"""
Engine-free checks for domo.scenes (obstacle arena geometry, ReplicaCAD
resolution/spawning with a synthetic dataset) and the domo.policies registry.
"""

import json
import math
import os

import pytest
import torch

from domo.scenes import ObstacleArena, ObstacleArenaConfig, ReplicaSpawn
from domo.scenes.replica import _habitat_to_zup, load_replica_scene, resolve_replica_assets
from domo.sim import Scene


class RecordingScene(Scene):
    """Records every add_* call; returned bodies remember their last position."""

    class Body:
        def __init__(self, kind, **geom):
            self.kind, self.geom, self.pos = kind, geom, None
            for name, value in geom.items():      # b.size, b.fixed, b.path, ...
                if name != "pos":                  # .pos is the runtime placement
                    setattr(self, name, value)

        def set_position(self, pos, envs_idx=None):
            self.pos = pos.clone()

    def __init__(self, fail_on: str = ""):
        self.n_envs = 0
        self.bodies = []
        self.fail_on = fail_on

    def _add(self, kind, **geom):
        if self.fail_on and self.fail_on in str(geom.get("path", "")):
            raise RuntimeError("engine import failed")
        b = self.Body(kind, **geom)
        self.bodies.append(b)
        return b

    def add_ground(self, height=0.0): ...
    def add_terrain(self, cfg): ...

    def add_mesh(self, file_path, pos, quat_wxyz, fixed=True, scale=1.0):
        return self._add("mesh", path=file_path, pos=pos, quat=quat_wxyz, fixed=fixed, scale=scale)

    def add_urdf_prop(self, file_path, pos, quat_wxyz, fixed=True):
        return self._add("urdf", path=file_path, pos=pos, quat=quat_wxyz, fixed=fixed)

    def add_box(self, size, pos, fixed=True):
        return self._add("box", size=size, pos=pos, fixed=fixed)

    def add_cylinder(self, radius, height, pos, fixed=True):
        return self._add("cylinder", radius=radius, height=height, pos=pos, fixed=fixed)

    def add_sphere(self, radius, pos, fixed=True):
        return self._add("sphere", radius=radius, pos=pos, fixed=fixed)

    def add_articulation(self, *a, **k): ...
    def add_lidar(self, *a, **k): ...

    def build(self, n_envs):
        self.n_envs = n_envs

    def step(self): ...


# ---------------------------------------------------------------------------
# ObstacleArena
# ---------------------------------------------------------------------------

def test_arena_builds_walls_and_parked_obstacles():
    cfg = ObstacleArenaConfig(n_chairs=2, n_sofas=1, n_pillars=1, n_steps=1, n_balls=1)
    scene = RecordingScene()
    arena = ObstacleArena(scene, cfg, spawn_xy=(0.5, 0.0))
    assert arena.termination_distance == cfg.half_size + 0.5
    walls = scene.bodies[:4]
    assert all(b.kind == "box" and b.fixed for b in walls)
    assert {b.size[2] for b in walls} == {cfg.wall_height}
    assert len(scene.bodies) == 4 + 2 * 4 + 1 + 1 + 1 + 1
    # Obstacles start parked far outside the arena until randomise().
    for b in scene.bodies[4:]:
        assert b.pos is None and b.geom["pos"][0] == 99.0
    # Sizes span their ranges evenly: a single sofa takes the range start.
    sofa = next(b for b in scene.bodies[4:] if b.kind == "box")
    assert sofa.size[0] == cfg.sofa_width_range[0]


def test_arena_randomise_places_everything_in_ring_and_keeps_chair_shape():
    torch.manual_seed(1)
    cfg = ObstacleArenaConfig(n_chairs=1, n_sofas=1, n_pillars=0, n_steps=0, n_balls=1)
    scene = RecordingScene()
    arena = ObstacleArena(scene, cfg, spawn_xy=(1.0, -1.0))
    envs = torch.tensor([0, 1, 2])
    arena.randomise(envs, torch.device("cpu"))
    legs = [b for b in scene.bodies if b.kind == "cylinder"]
    sofa = next(b for b in scene.bodies[4:] if b.kind == "box")
    ball = next(b for b in scene.bodies if b.kind == "sphere")
    for b in (sofa, ball):
        d = (b.pos[:, :2] - torch.tensor([1.0, -1.0])).norm(dim=1)
        assert (d >= cfg.ring_min).all() and (d <= cfg.ring_max).all()
    assert (sofa.pos[:, 2] == cfg.sofa_height_range[0] / 2).all()
    assert (ball.pos[:, 2] == cfg.ball_radius_range[0]).all()
    # The 4 legs stay a rigid square of side 2*spread after the random yaw.
    centre = torch.stack([leg.pos for leg in legs]).mean(dim=0)
    for leg in legs:
        r = (leg.pos[:, :2] - centre[:, :2]).norm(dim=1)
        assert torch.allclose(r, torch.full((3,), cfg.chair_leg_spread * math.sqrt(2)), atol=1e-5)
    arena.randomise(torch.tensor([], dtype=torch.long), torch.device("cpu"))   # no-op


# ---------------------------------------------------------------------------
# ReplicaCAD
# ---------------------------------------------------------------------------

def test_habitat_to_zup_swizzle():
    pos, quat = _habitat_to_zup((1.0, 2.0, 3.0), (0.5, 0.1, 0.2, 0.3))
    assert pos == (1.0, -3.0, 2.0)
    assert quat == (0.5, 0.1, -0.3, 0.2)


@pytest.fixture
def replica_dataset(tmp_path):
    """Minimal Habitat layout: configs/{stages,objects}/*.json + assets."""
    root = tmp_path / "replica_cad"
    (root / "configs" / "stages").mkdir(parents=True)
    (root / "configs" / "objects").mkdir(parents=True)
    (root / "stages").mkdir()
    (root / "objects").mkdir()
    (root / "stages" / "room.glb").write_bytes(b"glb")
    (root / "objects" / "chair.glb").write_bytes(b"glb")
    (root / "objects" / "fridge.urdf").write_text("<robot/>")
    (root / "configs" / "stages" / "room.stage_config.json").write_text(
        json.dumps({"render_asset": "../../stages/room.glb"}))
    (root / "configs" / "objects" / "chair.object_config.json").write_text(
        json.dumps({"render_asset": "../../objects/chair.glb"}))
    (root / "configs" / "objects" / "fridge.ao_config.json").write_text(
        json.dumps({"urdf_filepath": "../../objects/fridge.urdf"}))
    (root / "configs" / "objects" / "broken.object_config.json").write_text("{not json")
    scene = {
        "stage_instance": {"template_name": "stages/room",
                           "translation": [0, 0.2, 0]},
        "object_instances": [
            {"template_name": "objects/chair", "translation": [1, 0, 2],
             "rotation": [1, 0, 0, 0], "motion_type": "STATIC", "uniform_scale": 2.0},
            {"template_name": "objects/missing", "translation": [0, 0, 0],
             "rotation": [1, 0, 0, 0]},
            {"template_name": "objects/broken", "translation": [0, 0, 0],
             "rotation": [1, 0, 0, 0]},
        ],
        "articulated_object_instances": [
            {"template_name": "objects/fridge", "translation": [0, 0, 0],
             "rotation": [1, 0, 0, 0], "fixed_base": True},
        ],
    }
    scene_json = root / "apt.scene_instance.json"
    scene_json.write_text(json.dumps(scene))
    return str(scene_json), str(root)


def test_resolve_replica_assets(replica_dataset):
    scene_json, root = replica_dataset
    spawns = resolve_replica_assets(scene_json, root)
    assert [os.path.basename(s.asset_path) for s in spawns] == [
        "room.glb", "chair.glb", "fridge.urdf"]          # missing/broken skipped
    stage, chair, fridge = spawns
    assert isinstance(stage, ReplicaSpawn) and stage.fixed
    assert stage.pos == pytest.approx((0.0, 0.0, 0.2 - 0.05))   # y-up → z-up, nudged
    assert chair.pos == (1, -2, 0) and chair.fixed and chair.scale == 2.0
    assert fridge.fixed and fridge.scale == 1.0
    assert resolve_replica_assets(scene_json, root, stage_z_nudge=0.0)[0].pos[2] == 0.2


def test_load_replica_scene_spawns_and_tolerates_failures(replica_dataset, capsys):
    scene_json, root = replica_dataset
    scene = RecordingScene()
    assert load_replica_scene(scene, scene_json, root, verbose=False) == 3
    assert [b.kind for b in scene.bodies] == ["mesh", "mesh", "urdf"]
    assert scene.bodies[1].geom["scale"] == 2.0
    failing = RecordingScene(fail_on="chair")
    assert load_replica_scene(failing, scene_json, root, verbose=True) == 2
    out = capsys.readouterr().out
    assert "FAIL chair" in out and "ok  room" in out


# ---------------------------------------------------------------------------
# Stable-policy registry
# ---------------------------------------------------------------------------

def test_stable_policy_errors(monkeypatch, tmp_path):
    import domo.policies as pol
    with pytest.raises(KeyError):
        pol.stable_policy("fly")
    monkeypatch.setattr(pol, "POLICY_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="walk"):
        pol.stable_policy("walk")
    (tmp_path / "walk.pt").write_bytes(b"")
    assert pol.stable_policy("walk") == str(tmp_path / "walk.pt")
