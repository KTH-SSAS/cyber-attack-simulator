import json
import re
from argparse import ArgumentParser
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import ray
import torch
from numpy.typing import NDArray
from ray.rllib.agents import ppo

from attack_simulator.env.env import AttackSimulationEnv, register_rllib_env
from attack_simulator.rllib.ids_model import register_rllib_model
from attack_simulator.utils.config import EnvConfig
from attack_simulator.utils.rng import set_seeds


class RLLibEvaluator:
    """Run RLLib in evaluation."""

    def __init__(
        self, checkpoint_path: Path, parameter_dict: Optional[Dict[str, Any]] = None
    ) -> None:

        ray.init(local_mode=True)

        if isinstance(parameter_dict, Dict):
            model_config = parameter_dict.get("model")
            del parameter_dict["callbacks"]  # remove callbacks
            config = parameter_dict
        else:
            env_config: EnvConfig = EnvConfig.from_yaml("config/generated_env_config.yaml")
            env_config = replace(
                env_config,
                save_graphs=True,
                save_logs=True,
                seed=2,
                false_positive_rate=0.0,
                false_negative_rate=0.0,
            )
            model_config = {"use_lstm": False, "lstm_cell_size": 256}
            config = {
                "seed": env_config.seed,
                "framework": "torch",
                "env_config": asdict(env_config),
                "num_workers": 0,
                "model": model_config,
                "in_evaluation": True,
                "evaluation_num_workers": 0,
                "evaluation_interval": 1,
                # Run 1 episode each time evaluation runs.
                "evaluation_num_episodes": 1,
                "evaluation_config": {"explore": False},
            }

        self.model_config = model_config if model_config else {}
        self.config = config
        self.env = AttackSimulationEnv(EnvConfig(**self.config["env_config"]))

        set_seeds(self.config["seed"])

        self.config["disable_env_checking"] = True

        self.agent = ppo.PPOTrainer(config=self.config)
        # self.agent.restore(checkpoint_path)
        self.run_id = checkpoint_path.parts[-3]

    def test(self, episodes: int, render: bool = False) -> NDArray[np.int64]:
        """Test trained agent for a number of episodes.

        Return the episode reward
        """
        # instantiate env class

        # run until episode ends
        episode_rewards = np.zeros(episodes, dtype=np.int64)
        done = False
        obs = self.env.reset()

        def get_initial_lstm_state() -> Union[List[torch.Tensor], None]:
            return (
                [
                    torch.zeros(self.config["model"]["lstm_cell_size"]),
                    torch.zeros(self.config["model"]["lstm_cell_size"]),
                ]
                if "lstm_cell_size" in self.config["model"]
                else None
            )

        lstm_state = get_initial_lstm_state()

        if render:
            self.env.render()

        for e in range(0, episodes):
            episode_reward = 0
            while not done:
                action, lstm_state, agent_info = self.agent.compute_single_action(
                    obs, state=lstm_state, full_fetch=True
                )
                obs, reward, done, env_info = self.env.step(action)
                episode_reward += reward
                if render:
                    self.env.render()

            episode_rewards[e] = episode_reward
            obs = self.env.reset()
            lstm_state = get_initial_lstm_state()

        return episode_rewards


def main() -> None:
    """Main function."""

    register_rllib_env()
    register_rllib_model()
    parser = ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to RLLib run to load model from.")

    results_dir = Path("/home/jakob/sentience/ray_results/PPO")
    run_id = "PPO_AttackSimulationEnv_27e60_00000_0_seed=1_2022-06-28_13-28-11/"

    run_dir = results_dir / run_id

    assert run_dir.is_dir(), "Run directory does not exist."

    # args = parser.parse_args()

    checkpoint_folder = reversed(sorted((run_dir.glob("checkpoint_*")))).__next__()
    for f in checkpoint_folder.glob("checkpoint-*"):
        if re.match(r"checkpoint-\d\d?\d?$", f.name):
            checkpoint = Path(f)
            break

    params_file = run_dir / "params.json"
    with open(str(params_file), encoding="utf8") as p:
        params_dict = json.load(p)

    evaluator = RLLibEvaluator(checkpoint, parameter_dict=params_dict)

    reward = evaluator.test(1, render=True)
    print(np.mean(reward))


if __name__ == "__main__":
    main()
