from typing import Dict, Type

from .agent import Agent
from .attackers import (
    InformedAttacker,
    PathFinderAttacker,
    RandomAttacker,
    RandomNoActionAttacker,
    RoundRobinAttacker,
    RoundRobinNoActionAttacker,
    WellInformedAttacker,
)
from .baseline_agents import (
    DisableProbabilityAgent,
    NewRuleBasedAgent,
    RandomAgent,
    RiskAwareAgent,
    RuleBasedAgent,
    SkipAgent,
)

ATTACKERS: Dict[str, Type[Agent]] = {
    "well-informed": WellInformedAttacker,
    "informed": InformedAttacker,
    "round-robin-no-action": RoundRobinNoActionAttacker,
    "round-robin": RoundRobinAttacker,
    "random-no-action": RandomNoActionAttacker,
    "random": RandomAttacker,
    "pathplanner": PathFinderAttacker,
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
    "BanditAgent",
    "QLearningAgent",
    "ReinforceAgent",
]
