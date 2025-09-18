"""Planner orchestrating specialist agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PlanStep:
    name: str
    params: Dict[str, object]


class Planner:
    COMPARE_KEYWORDS = {"compare", "versus", "vs", "evolution", "trend", "global"}

    def __init__(self) -> None:
        pass

    def classify(self, query: str) -> str:
        lowered = query.lower()
        if any(keyword in lowered for keyword in {"how", "why", "explain"}):
            return "analysis"
        if any(keyword in lowered for keyword in self.COMPARE_KEYWORDS):
            return "compare"
        return "lookup"

    def plan(self, query: str) -> List[PlanStep]:
        query_type = self.classify(query)
        steps = [PlanStep("retrieve", {}), PlanStep("rerank", {})]
        if query_type == "compare":
            steps.append(PlanStep("graph_retrieve", {}))
        steps.extend(
            [
                PlanStep("pack", {}),
                PlanStep("summarize", {}),
                PlanStep("budget", {}),
                PlanStep("generate", {}),
                PlanStep("critic", {}),
            ]
        )
        return steps


__all__ = ["Planner", "PlanStep"]
