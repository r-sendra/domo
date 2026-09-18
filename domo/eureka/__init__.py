"""
Eureka / DrEureka skill-learning routine (M2 + M3 + robustness).

    from domo.eureka import SkillLearningRequest, learn_skill
    result = learn_skill(SkillLearningRequest(
        skill_name="getup",
        description="Stand up from any fallen posture and hold the stance.",
        run_dr=True))

Callable from the digital twin, a CLI, or the real-robot supervisor —
training always happens in worker subprocesses over simulation tasks.

Module map:
    spec      — request/config/result dataclasses, TASK_REGISTRY
    routine   — learn_skill entry point, shared stage functions, worker RPC
    graph     — LangGraph driver over the same stage functions
    prompts   — reward / reflection / DR prompt text
    rewards   — LLM code extraction, validation, exec-loading
    dr        — DrEureka: physics prior, DR proposal + clamping, retraining
    worker    — subprocess entry point (the only module that touches physics)
"""

from .routine import learn_skill, make_client, run_worker
from .spec import (
    TASK_REGISTRY,
    CandidateResult,
    DrEurekaConfig,
    EurekaConfig,
    IterationResult,
    LearnedSkill,
    SkillLearningRequest,
    TaskSpec,
)

__all__ = [
    "TASK_REGISTRY",
    "CandidateResult",
    "DrEurekaConfig",
    "EurekaConfig",
    "IterationResult",
    "LearnedSkill",
    "SkillLearningRequest",
    "TaskSpec",
    "learn_skill",
    "make_client",
    "run_worker",
]
