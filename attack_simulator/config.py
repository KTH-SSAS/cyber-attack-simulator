from dataclasses import dataclass


@dataclass
class EnvironmentConfig():
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

    @property
    def attack_steps(self):
        if self.graph_size == 'small':
            attack_steps = 7
        elif self.graph_size == 'medium':
            attack_steps = 29
        elif self.graph_size == 'large':
            attack_steps = 78
        return attack_steps


@dataclass
class AgentConfig():
    agent_type: str
    hidden_dim: int
    learning_rate: float
    input_dim: int
    num_actions: int
    allow_skip: bool
