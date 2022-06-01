import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser

import numpy as np
import ray
import torch
from ray.rllib.agents import ppo

from attack_simulator.config import EnvConfig
from attack_simulator.env import AttackSimulationEnv
from attack_simulator.rng import set_seeds
from numpy.typing import NDArray

class RLLibEvaluator:
    """Run RLLib in evaluation."""

    def __init__(self, checkpoint_path: Path, parameter_dict: Optional[dict] = None) -> None:

        ray.init(local_mode=True)
        self.config = {}
        if parameter_dict is None:
            env_config: EnvConfig = EnvConfig.from_yaml("config/generated_env_config.yaml")
            env_config = env_config.replace(
                save_graphs=True, save_logs=True, seed=2, false_positive=0.0, false_negative=0.0
            )
            self.model_config = {"use_lstm": False, "lstm_cell_size": 256}
            self.config = {
                "seed": env_config.seed,
                "framework": "torch",
                "env": AttackSimulationEnv,
                "env_config": asdict(env_config),
                "num_workers": 0,
                "model": self.model_config,
                "in_evaluation": True,
                "evaluation_num_workers": 0,
                "evaluation_interval": 1,
                # Run 1 episode each time evaluation runs.
                "evaluation_num_episodes": 1,
                "evaluation_config": {"explore": False},
            }
        else:
            self.model_config = parameter_dict.get("model")
            parameter_dict["env"] = AttackSimulationEnv
            del parameter_dict["callbacks"]
            self.config = parameter_dict

        self.env = AttackSimulationEnv(self.config["env_config"])

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

        lstm_state = [
                torch.zeros(self.config["model"]["lstm_cell_size"]),
                torch.zeros(self.config["model"]["lstm_cell_size"]),
            ] if self.model_config else None

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
            lstm_state = [
                torch.zeros(self.config["model"]["lstm_cell_size"]),
                torch.zeros(self.config["model"]["lstm_cell_size"]),
            ] if self.model_config else None

        return episode_rewards


def main() -> None:
    """Main function."""
    parser = ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to RLLib run to load model from.")
    
    results_dir = Path("/home/jakob/sentience/data/ray_results_from_vm/PPO")
    run_id = "PPO_AttackSimulationEnv_629f8_00000_0_seed=1_2022-05-30_13-35-35"

    run_dir = results_dir / run_id

    assert run_dir.is_dir()
    
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
