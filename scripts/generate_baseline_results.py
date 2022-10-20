from itertools import product
import dataclasses
from attack_simulator.config import EnvConfig
from attack_simulator.optimal_defender import TripwireDefender
from attack_simulator.random_defender import RandomDefender
from attack_simulator.custom_callback import AttackSimCallback
from attack_simulator.env import AttackSimulationEnv, register_rllib_env
from ray.rllib.algorithms import Algorithm
from tqdm import tqdm
import ray
import shelve
import numpy as np

config_file = "config/maze_env_config.yaml"
env_config = EnvConfig.from_yaml(config_file)
sweep_id = "fffff"
trainers = {"Tripwire": TripwireDefender, "Random": RandomDefender}



@ray.remote
def run_eval(config: dict, trainer_name: str, fp: float, fn: float, seed: int, num: int):
    env_config_dict = dataclasses.asdict(env_config)
    env_config_dict["false_positive"] = fp
    env_config_dict["false_negative"] = fn
    config["env_config"] = env_config_dict
    config["seed"] = seed
    dummy_env = AttackSimulationEnv(EnvConfig(**env_config_dict))
    config["defense_steps"] = dummy_env.sim.g.attack_steps_by_defense_step
    trainer: Algorithm = trainers[trainer_name](config)
    results = trainer.evaluate()
    return (results | {"trial_id": f"{trainer_name}_{sweep_id}_{num}"}, config.copy())


def main():

    config = {
        "callbacks": AttackSimCallback,
        "seed": 0,
        "framework": "torch",
        "env": register_rllib_env(),
        "num_workers": 0,
        "disable_env_checking": True,
        "evaluation_interval": 1,
        "evaluation_num_workers": 0,
        "simple_optimizer": True,
        "evaluation_duration": 500,
        "log_level": "ERROR",
    }

    ray.init()

    fp_range = np.geomspace(0.5, 1, 7) - 0.5
    fn_range = np.geomspace(0.5, 1, 7) - 0.5
    seeds = range(1, 4)

    ray.util.inspect_serializability(run_eval, name="test")

    futures = [
        run_eval.remote(config.copy(), trainer, fp, fn, seed, i)
        for i, (trainer, fp, fn, seed) in enumerate(product(trainers, fp_range, fn_range, seeds))
    ]

    results = ray.get(futures)

    with shelve.open("baseline_results_fine", flag="n") as db:
        for result, config in results:
            db[result["trial_id"]] = result | {"config": config}


if __name__ == "__main__":
    main()
