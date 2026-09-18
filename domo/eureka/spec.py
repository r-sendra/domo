"""
Specs and result types for the Eureka / DrEureka skill-learning routine.

Place in DOMO: this is the *contract* of the M2 (LLM reward generation) and
M3 (autonomous skill training) stages. Everything here is plain dataclasses
with no engine, LLM, or subprocess dependency, so the SAME request can be
issued from anywhere: the digital twin's PlanningController, a CLI, or
(later) the supervisor process running beside the real robot. Training
itself always happens in simulation, in worker subprocesses; the caller's
world is never touched.

Contents
--------
* ``TaskSpec`` / ``TASK_REGISTRY`` — which simulation task a skill trains on
  and how it is described to the LLM (add a new entry to teach the routine a
  new skill; see ``TaskSpec``).
* ``EurekaConfig`` / ``DrEurekaConfig`` — search and training budgets.
* ``SkillLearningRequest`` — the routine's entire input.
* ``CandidateResult`` / ``IterationResult`` / ``LearnedSkill`` — outputs, in
  increasing granularity. ``CandidateResult.rank_key`` is the single place
  that defines how candidates are ordered.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

__all__ = [
    "TASK_REGISTRY",
    "CandidateResult",
    "DrEurekaConfig",
    "EurekaConfig",
    "IterationResult",
    "LearnedSkill",
    "SkillLearningRequest",
    "TaskSpec",
]


@dataclass(frozen=True)
class TaskSpec:
    """How the routine instantiates and describes a training task.

    The worker imports ``module`` and builds ``task_class(config_class(**overrides))``;
    the prompts embed ``success_description`` (the FIXED metric the LLM must
    improve but cannot modify) and ``env_interface`` (the ``task`` fields the
    generated reward is allowed to read). To register a new skill, add an entry
    to ``TASK_REGISTRY``: the task class must be a ``VecTask`` implementing
    ``compute_success``/``compute_fitness`` and recording ``episode_outcomes``
    (see ``domo.tasks.go2_getup`` for the reference implementation).
    """
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


TASK_REGISTRY: dict[str, TaskSpec] = {
    "go2_getup": TaskSpec(
        module="domo.tasks.go2_getup",
        task_class="Go2GetUpTask",
        config_class="Go2GetUpConfig",
        success_description=(
            "The robot spawns FALLEN on its side or back with scrambled "
            "joints. Success (fixed metric, not modifiable): base height "
            "> 0.26 m AND |roll| < 0.4 rad AND |pitch| < 0.4 rad, held for "
            "25 consecutive control steps (0.5 s), within an 8 s episode."),
        env_interface=_GETUP_INTERFACE,
    ),
}


@dataclass
class EurekaConfig:
    """Budget of the Eureka evolutionary search (per ``learn_skill`` call).

    The search costs ``iterations × samples`` candidate trainings of
    ``train_steps`` env-steps each; every training runs in its own worker
    subprocess with ``n_envs`` parallel environments.
    """
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
    """Reward-aware physics prior sweep + LLM-proposed randomization.

    The ``*_values`` lists define the RAPP sweep grid: one frozen-policy
    evaluation per value with that parameter alone perturbed. A value is
    *feasible* when success stays ≥ ``max(feasible_floor, feasible_ratio ×
    nominal)``; the feasible values bound what the LLM may randomise over.
    """
    # RAPP: values swept per parameter to find the feasible bounds. One
    # frozen-policy evaluation per value, one parameter perturbed at a time.
    friction_values: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 1.5, 2.0, 4.0])
    base_mass_values: list[float] = field(default_factory=lambda: [-1.0, 0.0, 1.0, 2.0, 3.0, 5.0])
    com_shift_values: list[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.1, 0.15])
    kp_scale_values: list[float] = field(default_factory=lambda: [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5])
    obs_noise_values: list[float] = field(default_factory=lambda: [0.0, 0.02, 0.05, 0.1])
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
    """The routine's entire input: what to learn, on which task, with what budget.

    ``llm`` names a provider understood by ``domo.llm.make_llm`` and
    ``llm_kwargs`` is forwarded to that constructor; an explicit client passed
    to ``learn_skill(request, llm=...)`` overrides both. Run artefacts land
    under ``<run_root>/<skill_name>/``.
    """
    skill_name: str
    description: str               # natural language: what the skill must do
    task: str = "go2_getup"        # key into TASK_REGISTRY
    eureka: EurekaConfig = field(default_factory=EurekaConfig)
    run_dr: bool = False           # append the DrEureka robustness stage
    dr: DrEurekaConfig = field(default_factory=DrEurekaConfig)
    run_root: str = "runs/eureka"
    llm: str = "gemini"            # gemini | vllm | openai | gemini-lc | scripted
    llm_kwargs: dict | None = None   # forwarded to the client (model, base_url, ...)
    use_graph: bool = True         # orchestrate via LangGraph when available


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """One LLM-generated reward candidate and (if it ran) its training outcome.

    ``success_rate`` is the task's FIXED metric; ``fitness`` is the task's dense
    progress proxy (Eureka's F). ``error`` is set when extraction, validation,
    or the worker failed — such a candidate has ``ok == False`` and is
    excluded from ranking but still reported in the reflection.
    """
    index: int
    code: str
    success_rate: float = -1.0
    fitness: float = 0.0            # dense progress (Eureka F) — ranks candidates
    peak_height: float = 0.0        # diagnostic: mean peak base height
    ever_upright_rate: float = 0.0  # diagnostic: fraction that ever stood ≥1 step
    max_hold: float = 0.0           # diagnostic: mean longest upright streak (steps)
    mean_ep_len: float = 0.0
    snapshots: list[dict] = field(default_factory=list)
    checkpoint: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def rank_key(self) -> tuple[float, float, float]:
        """Sort key: lower is better (use with ``min``/``sorted``)."""
        # Rank by binary success first, then the dense fitness proxy — so the
        # search keeps a gradient even when no candidate has succeeded yet.
        return (-self.success_rate, -self.fitness, self.mean_ep_len)


@dataclass
class IterationResult:
    """All candidates of one evolutionary round."""
    index: int
    candidates: list[CandidateResult]

    @property
    def best(self) -> CandidateResult | None:
        """Best runnable candidate by ``rank_key``, or None if all failed."""
        ranked = sorted((c for c in self.candidates if c.ok),
                        key=lambda c: c.rank_key)
        return ranked[0] if ranked else None


@dataclass
class LearnedSkill:
    """Output of ``learn_skill``: the winning policy plus its provenance.

    ``checkpoint`` points at the policy to deploy. When DrEureka ran and
    produced a robust policy, ``checkpoint`` is the DR-retrained one and
    ``dr_config``/``dr_success_rate`` describe it; ``success_rate`` always
    refers to the Eureka-stage winner (non-randomised evaluation).
    """
    name: str
    task: str
    checkpoint: str
    reward_code: str
    success_rate: float
    dr_config: dict | None = None       # DrEureka output, if run
    dr_success_rate: float | None = None
    history: list[IterationResult] = field(default_factory=list)

    def summary(self) -> str:
        """Multi-line human-readable report (printed at the end of a run)."""
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
    """JSON-friendly view of a request (nested dataclasses become dicts)."""
    return asdict(request)
