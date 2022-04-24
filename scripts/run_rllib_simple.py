from dataclasses import asdict

from ray.rllib.agents import dqn

from attack_simulator.config import EnvConfig
from attack_simulator.env import AttackSimulationEnv
from attack_simulator.rng import set_seeds

config = {
    "attacker": "pathplanner",
    "false_positive": 0.0,
    "save_graphs": True,
    "save_logs": False,
    "false_negative": 0.0,
    "attack_start_time": 0,
    "seed": 42,
    "reward_mode": "simple",
    "graph_config": {
        "easy_ttc": 1,
        "hard_ttc": 10,
        "high_flag_reward": 500,
        "medium_flag_reward": 50,
        "low_flag_reward": 5,
        "prune": [],
        "root": "Attacker:-8227084409955727818:firstSteps",
        # "root": "internet.connect",
        # "unmalleable_assets": ["internet", "office_network", "hidden_network"],
        # "filename": "graphs/big.yaml",
        "filename": "graphs/model_small.json",
    },
}

set_seeds(5)

env_config: EnvConfig = EnvConfig(**config)  # type: ignore

env_config = env_config.replace(save_graphs=True, save_logs=True, seed=6)
model_config = {"use_lstm": False, "lstm_cell_size": 256}

config = {
    "seed": env_config.seed,
    "framework": "torch",
    "env": AttackSimulationEnv,
    "env_config": asdict(env_config),
    "num_workers": 0,
    # "model": model_config,
    "render_env": True,
}

# trainer = ppo.PPOTrainer(config=config)
trainer = dqn.DQNTrainer(config=config)


for i in range(1):
    result = trainer.train()
