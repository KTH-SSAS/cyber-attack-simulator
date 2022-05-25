import shutil
from dataclasses import asdict
from pathlib import Path

from ray.rllib.agents import dqn

from attack_simulator.config import EnvConfig
from attack_simulator.env import AttackSimulationEnv
from attack_simulator.rng import set_seeds

config = {
    "attacker": "pathplanner",
    "false_positive": 0.0,
    "save_graphs": True,
    "save_logs": True,
    "false_negative": 0.0,
    "attack_start_time": 0,
    "seed": 22,
    "reward_mode": "simple",
    "run_id": "simple",
    "graph_config": {
        "ttc": {"easy_ttc": 5, "hard_ttc": 10, "default_ttc": 1},
        "rewards": {
            "high_flag_reward": 10,
            "medium_flag_reward": 10,
            "low_flag_reward": 1,
            "default_reward": 0.0,
        },
        # "root": "internet.connect",
        # "unmalleable_assets": ["internet", "office_network", "hidden_network"],
        # "filename": "graphs/big.yaml",
        "filename": "graphs/test_graph.yaml",
    },
}

set_seeds(5)

env_config: EnvConfig = EnvConfig(**config)  # type: ignore

render_path = Path("render/simple")
if render_path.is_dir():
    shutil.rmtree(render_path)

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
    "disable_env_checking": True
}

# trainer = ppo.PPOTrainer(config=config)
trainer = dqn.DQNTrainer(config=config)


for i in range(1):
    result = trainer.train()
