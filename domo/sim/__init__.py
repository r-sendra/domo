"""
Simulation layer: engine-agnostic interfaces + backend factory.

Usage:
    from domo.sim import create_engine, SimConfig
    engine = create_engine("genesis", device="cuda")
    scene  = engine.create_scene(SimConfig(dt=0.02, headless=True))

The ABCs in `domo.sim.base` are the contract every other layer codes
against; a backend implements them in its own module and registers a
factory here. Backends are imported lazily (inside the factory) so that
e.g. RL utilities or the control layer can be used on machines without any
physics engine installed — only `create_engine(...)` triggers the import.

Adding a backend:
    1. implement PhysicsEngine / Scene / Articulation / LidarSensorHandle
       in `domo/sim/<name>_backend.py` (the only module allowed to import
       the physics package),
    2. `register_backend("<name>", factory)` where `factory(**kwargs)`
       returns the PhysicsEngine (kwargs come from `create_engine`).
"""

from __future__ import annotations

from collections.abc import Callable

from .base import (
    Articulation,
    CameraHandle,
    LidarConfig,
    LidarSensorHandle,
    PhysicsEngine,
    RigidObject,
    Scene,
    SimConfig,
    TerrainConfig,
    ViewerConfig,
)

__all__ = [
    "Articulation", "CameraHandle", "LidarConfig", "LidarSensorHandle",
    "PhysicsEngine", "RigidObject", "Scene", "SimConfig", "TerrainConfig",
    "ViewerConfig", "create_engine", "register_backend",
]

BackendFactory = Callable[..., PhysicsEngine]

_BACKEND_FACTORIES: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register (or replace) a PhysicsEngine factory: factory(**kwargs) -> PhysicsEngine."""
    _BACKEND_FACTORIES[name] = factory


def _genesis_factory(**kwargs) -> PhysicsEngine:
    # Deferred import: `genesis` is heavy and optional.
    from .genesis_backend import GenesisEngine
    return GenesisEngine(**kwargs)


register_backend("genesis", _genesis_factory)


def create_engine(name: str = "genesis", **kwargs) -> PhysicsEngine:
    """
    Instantiate a registered physics backend.

    Args:
        name: backend name as passed to `register_backend` ("genesis" by default).
        **kwargs: forwarded to the backend factory (e.g. `device="cuda"`).

    Raises:
        ValueError: if no backend is registered under `name`.
    """
    if name not in _BACKEND_FACTORIES:
        raise ValueError(
            f"Unknown physics backend '{name}'. "
            f"Available: {sorted(_BACKEND_FACTORIES)}")
    return _BACKEND_FACTORIES[name](**kwargs)
