"""
Eureka / DrEureka skill-learning routine (M2 + M3 + robustness).

    from domo.eureka import SkillLearningRequest, learn_skill
    result = learn_skill(SkillLearningRequest(
        skill_name="getup",
        description="Stand up from any fallen posture and hold the stance.",
        run_dr=True))

Callable from the digital twin, a CLI, or the real-robot supervisor —
training always happens in worker subprocesses over simulation tasks.
"""

from .routine import learn_skill, make_client, run_worker
from .spec import (TASK_REGISTRY, CandidateResult, DrEurekaConfig,
                   EurekaConfig, IterationResult, LearnedSkill,
                   SkillLearningRequest, TaskSpec)

__all__ = [
    "learn_skill", "make_client", "run_worker",
    "SkillLearningRequest", "EurekaConfig", "DrEurekaConfig",
    "LearnedSkill", "CandidateResult", "IterationResult",
    "TaskSpec", "TASK_REGISTRY",
]
