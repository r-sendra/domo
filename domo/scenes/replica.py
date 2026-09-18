"""
ReplicaCAD / Habitat scene loading.

Parses a Habitat `*.scene_instance.json`, resolves each template to an asset
file on disk, converts Habitat's Y-up frame to Z-up, and spawns everything
through the engine-agnostic `domo.sim.Scene` API (add_mesh / add_urdf_prop).

Ported from scripts/house_scene/go2_cpg_rl_avoid_house.py. Split into a pure
resolution step (`resolve_replica_assets` — testable without a physics
engine) and a spawn step (`load_replica_scene`).

Asset resolution: for a template name `<dir>/<name>` the matching Habitat
config is any `<name>*.json` under `<asset_root>/configs` (or `<asset_root>`
if there is no `configs/`); its `urdf_filepath` or `render_asset` entry,
relative to that JSON, is the spawnable file. Unresolvable templates are
skipped, not fatal — a partially furnished house is still a usable scene.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from domo.sim.base import Scene

__all__ = ["ReplicaSpawn", "load_replica_scene", "resolve_replica_assets"]

_log = logging.getLogger(__name__)

# Habitat identity rotation (wxyz) used when a stage carries no rotation.
_HABITAT_IDENTITY_QUAT = [1, 0, 0, 0]

# Lifted below the floor plane by default to prevent z-fighting.
_DEFAULT_STAGE_Z_NUDGE = -0.05


@dataclass
class ReplicaSpawn:
    """One resolved entity, ready for `Scene.add_mesh` / `add_urdf_prop`."""
    template_name: str
    asset_path: str
    pos: tuple[float, float, float]          # Genesis frame (Z-up)
    quat_wxyz: tuple[float, float, float, float]
    fixed: bool
    scale: float = 1.0


def _habitat_to_zup(pos, quat_wxyz):
    """
    Swizzle Habitat Y-up → Z-up (position and wxyz quaternion).

    Habitat: x right, y up, z towards the viewer. Z-up: (x, y, z) →
    (x, −z, y); the same axis permutation applies to the quaternion's
    vector part.
    """
    x, y, z = pos
    qw, qx, qy, qz = quat_wxyz
    return (x, -z, y), (qw, qx, -qz, qy)


def _resolve_asset_path(template_path: str, root_dir: str) -> str | None:
    """Hunt for the matching Habitat config JSON and return the asset path."""
    base_name = os.path.basename(template_path)
    search_dir = os.path.join(root_dir, "configs")
    if not os.path.exists(search_dir):
        search_dir = root_dir

    for subdir, _dirs, files in os.walk(search_dir):
        for file in files:
            if not (file.startswith(base_name) and file.endswith(".json")):
                continue
            json_path = os.path.join(subdir, file)
            try:
                with open(json_path) as f:
                    obj_config = json.load(f)
                asset_rel = (obj_config.get("urdf_filepath")
                             or obj_config.get("render_asset"))
            except (OSError, ValueError, AttributeError) as exc:
                # Unreadable / malformed / non-object JSON: not the config we
                # want; keep looking for another candidate file.
                _log.debug("skipping %s: %s", json_path, exc)
                continue
            if asset_rel:
                abs_path = os.path.abspath(os.path.join(
                    os.path.dirname(json_path), asset_rel))
                if os.path.exists(abs_path):
                    return abs_path
    return None


def resolve_replica_assets(scene_json: str, asset_root: str,
                           stage_z_nudge: float = _DEFAULT_STAGE_Z_NUDGE) -> list[ReplicaSpawn]:
    """
    Parse the scene instance file and resolve every entity to a spawnable
    asset. Pure I/O — no physics engine involved.

    Args:
        scene_json: path to the `*.scene_instance.json`.
        asset_root: ReplicaCAD dataset root containing `configs/` and assets.
        stage_z_nudge: vertical offset (m) applied to the stage (room shell).

    Returns:
        Spawns in file order: stage first (if resolved), then rigid objects
        (`fixed` iff motion_type == "STATIC"), then articulated objects
        (spawned as fixed props for now — their joints are not simulated).
    """
    with open(scene_json) as f:
        config = json.load(f)

    spawns: list[ReplicaSpawn] = []

    # Stage (room shell)
    stage_info = config.get("stage_instance", {})
    stage_tmpl = stage_info.get("template_name")
    if stage_tmpl:
        asset = _resolve_asset_path(stage_tmpl, asset_root)
        if asset:
            pos, quat = _habitat_to_zup(
                stage_info.get("translation", [0, 0, 0]),
                stage_info.get("rotation", _HABITAT_IDENTITY_QUAT))
            pos = (pos[0], pos[1], pos[2] + stage_z_nudge)
            spawns.append(ReplicaSpawn(stage_tmpl, asset, pos, quat, fixed=True))
        else:
            print(f"  [replica] could not resolve stage: {stage_tmpl}")

    # Rigid objects
    for obj in config.get("object_instances", []):
        tmpl = obj["template_name"]
        asset = _resolve_asset_path(tmpl, asset_root)
        if asset:
            pos, quat = _habitat_to_zup(obj["translation"], obj["rotation"])
            spawns.append(ReplicaSpawn(
                tmpl, asset, pos, quat,
                fixed=(obj.get("motion_type") == "STATIC"),
                scale=obj.get("uniform_scale", 1.0)))

    # Articulated objects (spawned as fixed props for now)
    for obj in config.get("articulated_object_instances", []):
        tmpl = obj["template_name"]
        asset = _resolve_asset_path(tmpl, asset_root)
        if asset:
            pos, quat = _habitat_to_zup(obj["translation"], obj["rotation"])
            spawns.append(ReplicaSpawn(
                tmpl, asset, pos, quat,
                fixed=obj.get("fixed_base", True),
                scale=obj.get("uniform_scale", 1.0)))

    return spawns


def _spawn(scene: Scene, s: ReplicaSpawn) -> bool:
    """Add one resolved asset; False if its file type is not spawnable."""
    if s.asset_path.endswith(".urdf"):
        scene.add_urdf_prop(s.asset_path, s.pos, s.quat_wxyz, fixed=s.fixed)
    elif s.asset_path.endswith((".glb", ".obj")):
        scene.add_mesh(s.asset_path, s.pos, s.quat_wxyz,
                       fixed=s.fixed, scale=s.scale)
    else:
        return False
    return True


def load_replica_scene(scene: Scene, scene_json: str, asset_root: str,
                       verbose: bool = True) -> int:
    """
    Spawn a ReplicaCAD scene into `scene` (before build). Returns the number
    of entities added. Assets the engine fails to import are reported
    (when `verbose`) and skipped rather than aborting the whole house.
    """
    spawns = resolve_replica_assets(scene_json, asset_root)
    n_ok = 0
    for s in spawns:
        try:
            if not _spawn(scene, s):
                continue
        except Exception as e:  # engine-specific import errors; keep loading
            if verbose:
                print(f"  [replica] FAIL {os.path.basename(s.template_name)}: {e}")
            continue
        n_ok += 1
        if verbose:
            print(f"  [replica] ok  {os.path.basename(s.template_name)}")
    return n_ok
