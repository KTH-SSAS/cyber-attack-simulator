from typing import Dict, Type

from .searchers import BreadthFirstAttacker, DepthFirstAttacker

from .agent import Agent
from .attackers import InformedAttacker, RandomAttacker, RoundRobinAttacker, WellInformedAttacker
from .baseline_defender import (
    DisableProbabilityAgent,
    NewRuleBasedAgent,
    RandomAgent,
    RiskAwareAgent,
    RuleBasedAgent,
    SkipAgent,
)
from .pathfinder import PathFinderAttacker

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

DEFENDERS = {
    "no-action": SkipAgent,
    "random": RandomAgent,
    "disable-probability": DisableProbabilityAgent,
    "rule-based": RuleBasedAgent,
    "new-rule-based": NewRuleBasedAgent,
    "risk-aware": RiskAwareAgent,
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
