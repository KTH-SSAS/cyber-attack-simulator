import shutil
from pathlib import Path

from ray.rllib.agents import dqn

from attack_simulator.env import register_rllib_env
from attack_simulator.ids_model import register_rllib_model
from attack_simulator.rng import set_seeds

if __name__ == "__main__":

    env_name = register_rllib_env()
    # Register the model with the registry.
    model_name = register_rllib_model()

    seed = 22
    set_seeds(seed)

    env_config = {
        "attacker": "pathplanner",
        "false_positive": 0.0,
        "save_graphs": False,
        "save_logs": False,
        "false_negative": 0.0,
        "attack_start_time": 0,
        "seed": seed,
        "reward_mode": "simple",
        "run_id": "simple",
        "graph_config": {
            "ttc": {"easy": 5, "hard": 10, "default": 1},
            "rewards": {
                "high_flag": 10,
                "medium_flag": 10,
                "low_flag": 1,
                "default": 0.0,
            },
            # "root": "asset:0:0",
            "root": "a.attack",
            # "root": "internet.connect",
            # "filename": "graphs/big.yaml",
            "filename": "graphs/test_graph.yaml",
        },
    }

    render_path = Path("render/simple")
    if render_path.is_dir():
        shutil.rmtree(render_path)

    model_config = {"use_lstm": False, "lstm_cell_size": 256}

    config = {
        "seed": env_config["seed"],
        "framework": "torch",
        "env": env_name,
        "env_config": env_config,
        "num_workers": 0,
        # "model": model_config,
        "render_env": False,
        "disable_env_checking": True,
        "evaluation_interval": 1,
        "evaluation_num_workers": 0,
        "evaluation_config": {
            "render_env": True,
            "num_envs_per_worker": 1,
            "env_config": env_config | {"save_graphs": True, "save_logs": True},
        },
    }

    # trainer = ppo.PPOTrainer(config=config)
    trainer = dqn.DQNTrainer(config=config)

    for i in range(1):
        result = trainer.train()
