import shutil
from pathlib import Path

from ray.rllib.agents import ppo
from attack_simulator.config import EnvConfig
from attack_simulator.custom_callback import AttackSimCallback

from attack_simulator.env import AttackSimulationEnv, register_rllib_env
import attack_simulator.ids_model as ids_model
import attack_simulator.optimal_defender as optimal_defender
import attack_simulator.random_defender as random_defender
from attack_simulator.rng import set_seeds

if __name__ == "__main__":

    env_name = register_rllib_env()
    # Register the model with the registry.
    ids_model.register_rllib_model()
    optimal_defender.register_rllib_model()
    random_defender.register_rllib_model()

    seed = 22
    set_seeds(seed)

    env_config = {
        "attacker": "depth-first",
        "false_positive": 0.0,
        "save_graphs": False,
        "save_logs": False,
        "false_negative": 0.0,
        "attack_start_time": 5,
        "seed": seed,
        "reward_mode": "downtime-penalty",
        "run_id": "simple",
        "graph_config": {
            "ttc": {"easy": 5, "hard": 10, "default": 1},
            "rewards": {
                "high_flag": 10,
                "medium_flag": 10,
                "low_flag": 1,
                "default": 0.0,
                "defense_default": 10,
            },
            # "root": "asset:0:0",
            "root": "attacker:13:enter:13",
            # "root": "internet.connect",
            # "filename": "graphs/big.yaml",
            "filename": "graphs/four_ways.yaml",
        },
    }

    dummy_env = AttackSimulationEnv(EnvConfig(**env_config))

    model_config = {
        "custom_model": "DefenderModel",
        "custom_model_config": {"defense_steps": dummy_env.sim.g.attack_steps_by_defense_step, "seed": seed},
    }

    render_path = Path("render/simple")
    if render_path.is_dir():
        shutil.rmtree(render_path)

    config = {
        "callbacks": AttackSimCallback,
        "seed": env_config["seed"],
        "framework": "torch",
        "env": env_name,
        "env_config": env_config,
        "num_workers": 0,
        "model": model_config,
        "render_env": False,
        "explore": False,
        "disable_env_checking": True,
        "evaluation_interval": 1,
        "evaluation_num_workers": 0,
        "evaluation_config": {
            "render_env": True,
            "num_envs_per_worker": 1,
            "env_config": env_config | {"save_graphs": True, "save_logs": True},
        },
    }

    trainer = ppo.PPOTrainer(config=config)
    # trainer = dqn.DQNTrainer(config=config)
    # trainer = RandomDefender(config=config)
    trainer = optimal_defender.TripwireDefender(config=config | {"simple_optimizer": True, "defense_steps": dummy_env.sim.g.attack_steps_by_defense_step})

    for i in range(1):
        result = trainer.train()
