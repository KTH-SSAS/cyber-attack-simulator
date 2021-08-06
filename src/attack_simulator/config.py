from dataclasses import dataclass
from typing import Optional

from .graph import AttackGraph


@dataclass
class EnvConfig:
    deterministic: bool
    early_flag_reward: int
    late_flag_reward: int
    final_flag_reward: int
    easy_ttc: int
    hard_ttc: int
    graph_size: str
    attacker_strategy: str
    true_positive: float
    false_positive: float


@dataclass
class AgentConfig:
    agent_type: str
    random_seed: Optional[int]
    input_dim: int
    hidden_dim: int
    num_actions: int
    learning_rate: float
    use_cuda: bool
    attack_graph: AttackGraph
