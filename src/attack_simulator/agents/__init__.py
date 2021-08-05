from .agent import Agent
from .baseline_agents import DisableProbabilityAgent, RandomAgent, RuleBasedAgent, SkipAgent
from .policy_agents import ReinforceAgent
from .tabular_agents import BanditAgent, QLearningAgent

DEFENDERS = {
    "random": RandomAgent,
    "no-action": SkipAgent,
    "disable-probability": DisableProbabilityAgent,
    "rule-based": RuleBasedAgent,
    "bandit": BanditAgent,
    "q-learning": QLearningAgent,
    "reinforce": ReinforceAgent,
}

__all__ = [
    "Agent",
    "RandomAgent",
    "SkipAgent",
    "DisableProbabilityAgent",
    "RuleBasedAgent",
    "BanditAgent",
    "QLearningAgent",
    "ReinforceAgent",
]
