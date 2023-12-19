from typing import Dict, Type

from .agent import Agent
from .attackers.attackers import (
    InformedAttacker,
    RandomAttacker,
    RoundRobinAttacker,
    WellInformedAttacker,
)
from .attackers.pathfinder import PathFinderAttacker
from .attackers.searchers import BreadthFirstAttacker, DepthFirstAttacker

ATTACKERS: Dict[str, Type[Agent]] = {
    # "well-informed": WellInformedAttacker,
    # "informed": InformedAttacker,
    # "round-robin-no-action": RoundRobinNoActionAttacker,
    # "round-robin": RoundRobinAttacker,
    # "random-no-action": RandomNoActionAttacker,
    "random": RandomAttacker,
    "pathplanner": PathFinderAttacker,
    "depth-first": DepthFirstAttacker,
    "breadth-first": BreadthFirstAttacker,
}

__all__ = [
    "Agent",
    "RandomAttacker",
    "RoundRobinAttacker",
    "InformedAttacker",
    "WellInformedAttacker",
    "RandomAgent",
    "SkipAgent",
    "DisableProbabilityAgent",
    "RuleBasedAgent",
    "NewRuleBasedAgent",
    "RiskAwareAgent",
]
