"""
Engine-agnostic scene builders.

Every builder here only uses the `domo.sim.Scene` contract (add_box,
add_mesh, ...), so the same environment is reproduced identically by the
World, by vectorised training tasks and by future LLM-generated layouts.
Builders follow the two-phase rule of the sim layer: construct (add
entities) BEFORE `scene.build()`, randomise / move things AFTER it.

Adding a scene kind: write a builder taking a `Scene` (plus a config
dataclass), then teach `domo.world.WorldConfig.scene_kind` about it.
"""

from .arena import ObstacleArena, ObstacleArenaConfig
from .replica import ReplicaSpawn, load_replica_scene, resolve_replica_assets

__all__ = ["ObstacleArena", "ObstacleArenaConfig", "ReplicaSpawn",
           "load_replica_scene", "resolve_replica_assets"]
