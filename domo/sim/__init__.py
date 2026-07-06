"""
Simulation layer: engine-agnostic interfaces + backend factory.

Usage:
    from domo.sim import create_engine, SimConfig
    engine = create_engine("genesis", device="cuda")
    scene  = engine.create_scene(SimConfig(dt=0.02, headless=True))

Backends are imported lazily so that e.g. RL utilities or the control layer
can be used on machines without any physics engine installed.
"""

from .base import (
    Articulation,
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
    "Articulation", "LidarConfig", "LidarSensorHandle", "PhysicsEngine",
    "RigidObject", "Scene", "SimConfig", "TerrainConfig", "ViewerConfig",
    "create_engine", "register_backend",
]

_BACKEND_FACTORIES = {}


def register_backend(name: str, factory) -> None:
    """Register a PhysicsEngine factory: factory(**kwargs) -> PhysicsEngine."""
    _BACKEND_FACTORIES[name] = factory


def _genesis_factory(**kwargs):
    from .genesis_backend import GenesisEngine
    return GenesisEngine(**kwargs)


register_backend("genesis", _genesis_factory)


def create_engine(name: str = "genesis", **kwargs) -> PhysicsEngine:
    if name not in _BACKEND_FACTORIES:
        raise ValueError(
            f"Unknown physics backend '{name}'. "
            f"Available: {sorted(_BACKEND_FACTORIES)}")
    return _BACKEND_FACTORIES[name](**kwargs)
