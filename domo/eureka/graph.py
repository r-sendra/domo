"""
LangGraph orchestration of the Eureka + DrEureka pipeline.

The routine is expressed as an explicit state machine — each stage is a node,
the evolutionary loop is a conditional edge, and the DrEureka branch is a
conditional edge off the best-selection node:

    generate_and_train ─▶ reflect ─▶ (more iters? ─▶ generate_and_train)
                                       │ done
                                       ▼
                                  select_best ─▶ (run_dr? ─▶ dr_stage) ─▶ END

The nodes call the same stage functions the imperative path uses
(routine.run_iteration, dr.run_dr_eureka), so the two drivers stay in lock
step — LangGraph adds inspectable structure, checkpointing hooks, and a place
for future human-in-the-loop interrupts, not a second implementation.

LangGraph is an optional dependency; `learn_skill` falls back to the
imperative path if it is not installed.
"""

from __future__ import annotations

import os
from typing import TypedDict

from domo.llm.client import LLMClient

from . import prompts
from .spec import TASK_REGISTRY, IterationResult, LearnedSkill, SkillLearningRequest

__all__ = ["build_graph", "run_graph"]


class EurekaState(TypedDict, total=False):
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
    history = state.get("history", []) + [iteration]
    return {
        "history": history,
        "iteration": it + 1,
        "reflection": prompts.reflection_block(iteration.candidates),
    }


def _should_continue(state: EurekaState) -> str:
    """Loop back for another evolutionary round, or move to selection."""
    if state["iteration"] < state["request"].eureka.iterations:
        return "iterate"
    return "select"


def _node_select(state: EurekaState) -> EurekaState:
    from .routine import select_global_best
    request = state["request"]
    best = select_global_best(state["history"])
    if best is None:
        raise RuntimeError(
            "Eureka produced no runnable candidate — see iter_*/reflection.txt")
    skill = LearnedSkill(
        name=request.skill_name, task=request.task,
        checkpoint=best.checkpoint, reward_code=best.code,
        success_rate=best.success_rate, history=state["history"])
    return {"skill": skill}


def _route_dr(state: EurekaState) -> str:
    return "dr" if state["request"].run_dr else "finish"


def _node_dr(state: EurekaState) -> EurekaState:
    from .dr import run_dr_eureka
    skill = run_dr_eureka(state["request"], state["skill"], state["llm"])
    return {"skill": skill}


def _node_finish(state: EurekaState) -> EurekaState:
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
        {"recursion_limit": 4 * cfg.iterations + 20})
    return final["skill"]
