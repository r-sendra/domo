"""
Specs and result types for the Eureka / DrEureka skill-learning routine.

`SkillLearningRequest` is the routine's entire input — deliberately plain
data so the SAME request can be issued from anywhere: the digital twin's
PlanningController, a CLI, or (later) the supervisor process running beside
the real robot. Training itself always happens in simulation, in worker
subprocesses; the caller's world is never touched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

__all__ = ["TaskSpec", "TASK_REGISTRY", "EurekaConfig", "DrEurekaConfig",
           "SkillLearningRequest", "CandidateResult", "IterationResult",
           "LearnedSkill"]


@dataclass(frozen=True)
class TaskSpec:
    """How the routine instantiates and describes a training task."""
    module: str
    task_class: str
    config_class: str
    success_description: str
    env_interface: str          # prompt block: what generated code may use


_GETUP_INTERFACE = """\
The reward function receives `task` (the running environment). Available:
  task.robot.state.base_pos            [N,3]  world position (z = height, m)
  task.robot.state.base_quat           [N,4]  orientation, wxyz
  task.robot.state.base_euler          [N,3]  roll, pitch, yaw (rad)
  task.robot.state.base_lin_vel        [N,3]  body-frame linear velocity (m/s)
  task.robot.state.base_ang_vel        [N,3]  body-frame angular velocity (rad/s)
  task.robot.state.projected_gravity   [N,3]  gravity dir in body frame
                                              ([0,0,-1] when perfectly upright)
  task.robot.state.dof_pos             [N,12] joint positions (rad)
  task.robot.state.dof_vel             [N,12] joint velocities (rad/s)
  task.robot.default_dof_pos           [12]   nominal standing joint angles
  task.actions, task.last_actions      [N,12] current / previous action
  task.episode_length_buf              [N]    steps since reset
  task.dt                              float  control period (0.02 s)
All tensors are torch tensors on the same device. N = number of envs."""


TASK_REGISTRY: Dict[str, TaskSpec] = {
    "go2_getup": TaskSpec(
        module="domo.tasks.go2_getup",
        task_class="Go2GetUpTask",
        config_class="Go2GetUpConfig",
        success_description=(
            "The robot spawns FALLEN on its side or back with scrambled "
            "joints. Success (fixed metric, not modifiable): base height "
            "> 0.26 m AND |roll| < 0.4 rad AND |pitch| < 0.4 rad, held for "
            "50 consecutive control steps (1 s), within a 5 s episode."),
        env_interface=_GETUP_INTERFACE,
    ),
}


@dataclass
class EurekaConfig:
    iterations: int = 3            # evolutionary rounds
    samples: int = 4               # reward candidates per round
    temperature: float = 1.0       # sampling diversity
    safety_reward: bool = True     # DrEureka Stage 1: safety-regularized prompt

    # Per-candidate training budget
    n_envs: int = 2048
    train_steps: int = 4_000_000
    rollout_steps: int = 24
    hidden_size: int = 256
    eval_episodes: int = 32
    device: str = "cuda"
    headless: bool = True
    worker_timeout_s: float = 3600.0
    snapshots: int = 4             # reward-reflection sample points


@dataclass
class DrEurekaConfig:
    """Reward-aware physics prior sweep + LLM-proposed randomization."""
    # RAPP: values swept per parameter to find the feasible bounds. One
    # frozen-policy evaluation per value, one parameter perturbed at a time.
    friction_values: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 1.5, 2.0, 4.0])
    base_mass_values: List[float] = field(default_factory=lambda: [-1.0, 0.0, 1.0, 2.0, 3.0, 5.0])
    com_shift_values: List[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.1, 0.15])
    kp_scale_values: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5])
    obs_noise_values: List[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.1])
    # A setting is feasible if success ≥ max(floor, ratio × nominal success)
    feasible_ratio: float = 0.5
    feasible_floor: float = 0.1
    eval_episodes: int = 32
    # DrEureka Stage 3: the LLM proposes `samples` INDEPENDENT DR configs;
    # all are trained and the best is kept (paper uses m=16).
    samples: int = 4
    retrain_steps: int = 4_000_000


@dataclass
class SkillLearningRequest:
    skill_name: str
    description: str               # natural language: what the skill must do
    task: str = "go2_getup"        # key into TASK_REGISTRY
    eureka: EurekaConfig = field(default_factory=EurekaConfig)
    run_dr: bool = False           # append the DrEureka robustness stage
    dr: DrEurekaConfig = field(default_factory=DrEurekaConfig)
    run_root: str = "runs/eureka"
    llm: str = "gemini"            # gemini | vllm | openai | gemini-lc | scripted
    llm_kwargs: Optional[dict] = None   # forwarded to the client (model, base_url, ...)
    use_graph: bool = True         # orchestrate via LangGraph when available


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    index: int
    code: str
    success_rate: float = -1.0
    mean_ep_len: float = 0.0
    snapshots: List[dict] = field(default_factory=list)
    checkpoint: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class IterationResult:
    index: int
    candidates: List[CandidateResult]

    @property
    def best(self) -> Optional[CandidateResult]:
        ranked = sorted((c for c in self.candidates if c.ok),
                        key=lambda c: (-c.success_rate, c.mean_ep_len))
        return ranked[0] if ranked else None


@dataclass
class LearnedSkill:
    name: str
    task: str
    checkpoint: str
    reward_code: str
    success_rate: float
    dr_config: Optional[dict] = None       # DrEureka output, if run
    dr_success_rate: Optional[float] = None
    history: List[IterationResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"LearnedSkill '{self.name}' ({self.task})",
                 f"  checkpoint : {self.checkpoint}",
                 f"  success    : {self.success_rate:.0%}"]
        if self.dr_config is not None:
            lines.append(f"  DR         : {self.dr_config} "
                         f"(robust success {self.dr_success_rate:.0%})")
        for it in self.history:
            rates = ", ".join(
                (f"{c.success_rate:.0%}" if c.ok else "ERR")
                for c in it.candidates)
            lines.append(f"  iter {it.index + 1}: [{rates}]")
        return "\n".join(lines)


def request_to_dict(request: SkillLearningRequest) -> dict:
    return asdict(request)
