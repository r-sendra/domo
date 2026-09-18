"""
DOMO — Developmental Orchestration of Motor-skill Onset.

Layered library (lower layers never import higher ones):
    domo.sim         — physics-engine abstraction (ABCs + backend registry;
                       Genesis backend today, swappable via register_backend)
    domo.robot       — robot specs, sensors, actuators, state snapshot,
                       lidar device models, domain randomization
    domo.utils       — pure-math utilities (rotations, ...)
    domo.scenes      — engine-agnostic scene builders (obstacle arena, ReplicaCAD)
    domo.control     — engine-free controllers (CPG, analytic leg IK/FK,
                       control loops, navigation, SLAM)
    domo.world       — the goal-free digital twin (engine + scene + robot + sensors)
    domo.skills      — skill library / grammar composed on top of the World
    domo.tasks       — vectorised training task environments (RL-agnostic API)
    domo.rl          — optional PPO training layer
    domo.checkpoints — checkpoint I/O and config reconstruction (both formats)
    domo.policies    — registry of blessed ("stable") policy checkpoints
    domo.llm / domo.eureka — LLM supervisor (Eureka / DrEureka)
    domo.hri         — human-robot interaction (voice)
    domo.dashboard   — non-blocking live web dashboard (twin/eval only)

Subpackages are imported lazily where they carry heavy deps; importing
`domo` itself requires only the standard library.
"""

__version__ = "0.1.0"
