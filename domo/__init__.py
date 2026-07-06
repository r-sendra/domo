"""
DOMO — Developmental Orchestration of Motor-skill Onset.

Layered library:
    domo.sim     — physics-engine abstraction (Genesis backend today)
    domo.robot   — robot specs, sensors, actuators, state
    domo.control — engine-free controllers (CPG, analytic leg IK/FK)
    domo.tasks   — vectorised task environments (RL-agnostic API)
    domo.rl      — optional PPO training layer
    domo.llm     — developmental supervisor (skill-gap detection; upcoming)
    domo.utils   — pure-math utilities (rotations, ...)

Subpackages are imported lazily where they carry heavy deps; importing
`domo` itself requires only the standard library.
"""

__version__ = "0.1.0"
