from .agent import Agent
from .attackers import (
    InformedAttacker,
    RandomAttacker,
    RandomNoActionAttacker,
    RoundRobinAttacker,
    RoundRobinNoActionAttacker,
    WellInformedAttacker,
)
from .baseline_agents import DisableProbabilityAgent, RandomAgent, RuleBasedAgent, SkipAgent
from .policy_agents import ReinforceAgent
from .tabular_agents import BanditAgent, QLearningAgent

ATTACKERS = {
    "well-informed": WellInformedAttacker,
    "informed": InformedAttacker,
    "round-robin-no-action": RoundRobinNoActionAttacker,
    "round-robin": RoundRobinAttacker,
    "random-no-action": RandomNoActionAttacker,
    "random": RandomAttacker,
}

DEFENDERS = {
    "no-action": SkipAgent,
    "random": RandomAgent,
    "disable-probability": DisableProbabilityAgent,
    "rule-based": RuleBasedAgent,
    "bandit": BanditAgent,
    "q-learning": QLearningAgent,
    "reinforce": ReinforceAgent,
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
    "BanditAgent",
    "QLearningAgent",
    "ReinforceAgent",
]
