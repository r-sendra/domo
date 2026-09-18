"""
LangGraph orchestration of the Eureka + DrEureka pipeline.

The routine is expressed as an explicit state machine — each stage is a node,
the evolutionary loop is a conditional edge, and the DrEureka branch is a
conditional edge off the best-selection node:

    START ─▶ iterate ─▶ (more iters? ─▶ iterate)
                          │ done
                          ▼
                        select ─▶ (run_dr? ─▶ dr) ─▶ finish ─▶ END

The nodes call the same stage functions the imperative path uses
(routine.run_iteration, routine.build_learned_skill, dr.run_dr_eureka,
routine.finalize_skill), so the two drivers stay in lock step — LangGraph
adds inspectable structure, checkpointing hooks, and a place for future
human-in-the-loop interrupts, not a second implementation. Any change to
what a stage does belongs in those shared functions, never in a node.

LangGraph is an optional dependency; `learn_skill` falls back to the
imperative path if it is not installed. Node functions return partial state
updates (LangGraph merges them into ``EurekaState``).
"""

from __future__ import annotations

import os
from typing import TypedDict

from domo.llm.client import LLMClient

from . import prompts
from .spec import TASK_REGISTRY, IterationResult, LearnedSkill, SkillLearningRequest

__all__ = ["build_graph", "run_graph"]

# LangGraph's recursion limit counts node visits: one "iterate" per round
# plus select/dr/finish. Lifted per run so long searches never trip it.
_RECURSION_STEPS_PER_ITER = 4
_RECURSION_SLACK = 20


class EurekaState(TypedDict, total=False):
    """Graph state; ``iteration`` counts completed rounds."""
    request: SkillLearningRequest
    llm: LLMClient
    reflection: str
    iteration: int
    history: list[IterationResult]
    skill: LearnedSkill | None


def _node_iterate(state: EurekaState) -> EurekaState:
    """Run one Eureka round (prompt → sample → train → reflect)."""
    from .routine import run_iteration
    request = state["request"]
    task_spec = TASK_REGISTRY[request.task]
    it = state.get("iteration", 0)
    root = os.path.join(request.run_root, request.skill_name)
    iteration = run_iteration(request, task_spec, state["llm"],
                              state.get("reflection", ""),
                              os.path.join(root, f"iter_{it}"), it)
    return {
        "history": [*state.get("history", []), iteration],
        "iteration": it + 1,
        "reflection": prompts.reflection_block(iteration.candidates),
    }


def _should_continue(state: EurekaState) -> str:
    """Loop back for another evolutionary round, or move to selection."""
    if state["iteration"] < state["request"].eureka.iterations:
        return "iterate"
    return "select"


def _node_select(state: EurekaState) -> EurekaState:
    """Pick the global best candidate and wrap it as a ``LearnedSkill``."""
    from .routine import build_learned_skill
    return {"skill": build_learned_skill(state["request"], state["history"])}


def _route_dr(state: EurekaState) -> str:
    """Branch into the DrEureka robustness stage when requested."""
    return "dr" if state["request"].run_dr else "finish"


def _node_dr(state: EurekaState) -> EurekaState:
    """DrEureka: RAPP → LLM DR proposals → train all → keep best."""
    from .dr import run_dr_eureka
    skill = run_dr_eureka(state["request"], state["skill"], state["llm"])
    return {"skill": skill}


def _node_finish(state: EurekaState) -> EurekaState:
    """Write result.json and print the summary."""
    from .routine import finalize_skill
    return {"skill": finalize_skill(state["request"], state["skill"])}


def build_graph():
    """Compile the Eureka/DrEureka StateGraph (raises if langgraph absent)."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(EurekaState)
    g.add_node("iterate", _node_iterate)
    g.add_node("select", _node_select)
    g.add_node("dr", _node_dr)
    g.add_node("finish", _node_finish)

    g.add_edge(START, "iterate")
    g.add_conditional_edges("iterate", _should_continue,
                            {"iterate": "iterate", "select": "select"})
    g.add_conditional_edges("select", _route_dr,
                            {"dr": "dr", "finish": "finish"})
    g.add_edge("dr", "finish")
    g.add_edge("finish", END)
    return g.compile()


def run_graph(request: SkillLearningRequest, llm: LLMClient) -> LearnedSkill:
    """Graph driver; same inputs/outputs/side effects as the imperative loop."""
    os.makedirs(os.path.join(request.run_root, request.skill_name), exist_ok=True)
    cfg = request.eureka
    print(f"\n{'=' * 60}\n  EUREKA (LangGraph) — learning "
          f"'{request.skill_name}' ({cfg.iterations} iters × {cfg.samples} "
          f"candidates)\n{'=' * 60}")
    graph = build_graph()
    # Iteration count can exceed LangGraph's default recursion cap; lift it.
    final = graph.invoke(
        {"request": request, "llm": llm, "reflection": "",
         "iteration": 0, "history": []},
        {"recursion_limit": _RECURSION_STEPS_PER_ITER * cfg.iterations
                            + _RECURSION_SLACK})
    return final["skill"]
