import shelve
from pathlib import Path

import cloudpickle
import ray
from tqdm import tqdm

import attack_simulator.ids_model as ids_model
from attack_simulator.custom_callback import AttackSimCallback
from attack_simulator.env import register_rllib_env

trainers = {"PPO": ids_model.Defender}


def get_checkpoint(folder):
    checkpoint = next(reversed(sorted((folder.glob("checkpoint_*")))))
    if checkpoint:
        with open(folder / "params.pkl", "rb") as f:
            run_config = cloudpickle.load(f)
    return checkpoint, run_config


# @ray.remote
def run_eval(config: dict, checkpoint, trainer_name: str, sweep_id, num: int):
    #config["env_config"]["sim_config"]["false_positive_rate"] = fp
    #config["env_config"]["sim_config"]["false_negative_rate"] = fn
    #if filename:
    #    config["env_config"]["graph_config"]["filename"] = filename
    # config["seed"] = seed

    trainer = trainers[trainer_name](config)
    trainer.restore(checkpoint)
    results = trainer.evaluate()

    return results | {"trial_id": f"{trainer_name}_{sweep_id}_{num:05}"} | {"config": config}


def main():
    # Config options to ovveride
    config = {
        "callbacks": AttackSimCallback,
        "framework": "torch",
        "env": register_rllib_env(),
        "num_workers": 0,
        "disable_env_checking": True,
        "evaluation_interval": 1,
        "evaluation_num_workers": 0,
        "evaluation_duration": 500,
        "log_level": "ERROR",
        "num_gpus": 0,
        "num_gpus_per_worker": 0,
    }

    # fp_range = np.linspace(0, 1, 5)
    # fn_range = np.linspace(0, 1, 5)
    # seeds = range(1, 4)
    # filenames = [None]

    ray_results = Path.home() / "sentience/data/ray_results/"
    sweeps = ["09041"]
    local_mode = False

    ray.init(local_mode=local_mode)

    checkpoint_configs = [
        (algorithm, sweep_id, get_checkpoint(folder))
        for sweep_id in sweeps
        for algorithm in ray_results.iterdir()
        for folder in sorted((ray_results / algorithm).glob(f"*_{sweep_id}_*"))
    ]

    merged_configs = (
        (algorithm, sweep_id, checkpoint, run_config | config)
        for algorithm, sweep_id, (checkpoint, run_config) in checkpoint_configs
    )

    trials = (
        run_eval(run_config, checkpoint, algorithm.name, sweep_id + "eval", i)
        for i, (algorithm, sweep_id, checkpoint, run_config) in enumerate(merged_configs)
    )

    for result in tqdm(trials, total=len(checkpoint_configs)):
        trial_id = result["trial_id"]
        sweep = trial_id.split("_")[1]
        with shelve.open(f"eval_{sweep}_hipenalty", flag="c") as db:
            db[result["trial_id"]] = result


main()
