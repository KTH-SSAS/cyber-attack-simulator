import dataclasses
import shelve
from itertools import product
from pathlib import Path

import numpy as np
import ray
from ray.rllib.algorithms import Algorithm

from attack_simulator.utils.config import EnvConfig
from attack_simulator.rllib.custom_callback import AttackSimCallback
from attack_simulator.env.env import AttackSimulationEnv, register_rllib_env
from attack_simulator.rllib.optimal_defender import TripwireDefender
from attack_simulator.rllib.random_defender import RandomDefender
from attack_simulator.utils.rng import set_seeds

config_file = "config/maze_env_config.yaml"
env_config = EnvConfig.from_yaml(config_file)
sweep_id = "modgraph"
trainers = {"Tripwire": TripwireDefender, "Random": RandomDefender}


@ray.remote
def run_eval(
    config: dict, trainer_name: str, fp: float, fn: float, seed: int, num: int, filename: str = None
):
    env_config_dict = dataclasses.asdict(env_config)
    env_config_dict["false_positive"] = fp
    env_config_dict["false_negative"] = fn
    if filename:
        env_config_dict["graph_config"]["filename"] = filename
    config["env_config"] = env_config_dict
    config["seed"] = seed
    dummy_env = AttackSimulationEnv(EnvConfig(**env_config_dict))
    config["defense_steps"] = dummy_env.sim.g.attack_steps_by_defense_step
    trainer: Algorithm = trainers[trainer_name](config)
    results = trainer.evaluate()
    return (results | {"trial_id": f"{trainer_name}_{sweep_id}_{num:05}"}, config.copy())


def main():
    # Set global seeds
    global_seed = 22
    set_seeds(global_seed)

    config = {
        "callbacks": AttackSimCallback,
        "framework": "torch",
        "env": register_rllib_env(),
        "num_workers": 0,
        "disable_env_checking": True,
        "evaluation_interval": 1,
        "evaluation_num_workers": 0,
        "simple_optimizer": True,
        "evaluation_duration": 500,
        "log_level": "ERROR",
        "gamma": 1,
    }

    ray.init(local_mode=False, num_cpus=8, num_gpus=0)

    # fp_range = np.geomspace(0.5, 1, 7) - 0.5
    # fn_range = np.geomspace(0.5, 1, 7) - 0.5
    # fp_range = [0.1]
    # fn_range = [0.1]
    fp_range = np.linspace(0, 1, 5)
    fn_range = np.linspace(0, 1, 5)
    seeds = range(1, 4)
    filenames = [None]

    #  [
    #       f"graphs/second_graph_sweep/model_graph_{size}.yaml"
    #       for size in [80, 20, 60, 40]
    # ]

    ray.util.inspect_serializability(run_eval, name="test")

    futures = [
        run_eval.remote(config.copy(), trainer, fp, fn, seed, i, filename)
        for i, (trainer, fp, fn, seed, filename) in enumerate(
            product(trainers, fp_range, fn_range, seeds, filenames)
        )
        if fn <= 1 - fp
    ]

    results = ray.get(futures)

    out_path = Path("baselines")

    with shelve.open(f"baseline_{sweep_id}", flag="n") as db:
        for result, config in results:
            db[result["trial_id"]] = result | {"config": config}


if __name__ == "__main__":
    main()
