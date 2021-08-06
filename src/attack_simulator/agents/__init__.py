from .agent import Agent
from .attackers import InformedAttacker, RandomAttacker, RoundRobinAttacker, WellInformedAttacker
from .baseline_agents import DisableProbabilityAgent, RandomAgent, RuleBasedAgent, SkipAgent
from .policy_agents import ReinforceAgent
from .tabular_agents import BanditAgent, QLearningAgent

ATTACKERS = {
    "well-informed": WellInformedAttacker,
    "informed": InformedAttacker,
    "round-robin": RoundRobinAttacker,
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
