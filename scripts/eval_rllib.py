import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import ray
import torch
from ray.rllib.agents import ppo

from attack_simulator.config import EnvConfig
from attack_simulator.env import AttackSimulationEnv
from attack_simulator.rng import set_seeds


class RLLibEvaluator:
    """Run RLLib in evaluation."""

    def __init__(self, checkpoint_path) -> None:

        ray.init(local_mode=True)

        model_config = {"use_lstm": True, "lstm_cell_size": 256}

        self.env_config: EnvConfig = EnvConfig.from_yaml("config/generated_env_config.yaml")

        set_seeds(5)

        self.env_config = self.env_config.replace(
            save_graphs=True, save_logs=True, seed=2, false_positive=0.0, false_negative=0.0
        )

        self.config = {
            "seed": self.env_config.seed,
            "framework": "torch",
            "env": AttackSimulationEnv,
            "env_config": asdict(self.env_config),
            "num_workers": 0,
            "model": model_config,
            "in_evaluation": True,
            "evaluation_num_workers": 0,
            "evaluation_interval": 1,
            # Run 1 episode each time evaluation runs.
            "evaluation_num_episodes": 1,
            "evaluation_config": {"explore": False},
        }

        self.agent = ppo.PPOTrainer(config=self.config)
        self.agent.restore(checkpoint_path)
        self.run_id = checkpoint_path.split("/")[-3]

    def test(self, episodes, render=False):
        """Test trained agent for a number of episodes.

        Return the episode reward
        """
        # instantiate env class
        env = AttackSimulationEnv(asdict(self.env_config))

        # run until episode ends
        episode_rewards = np.zeros(episodes, dtype=np.int64)
        done = False
        obs = env.reset()

        lstm_state = [
            torch.zeros(self.config["model"]["lstm_cell_size"]),
            torch.zeros(self.config["model"]["lstm_cell_size"]),
        ]

        if render:
            env.render(subdir=self.run_id)

        for e in range(0, episodes):
            episode_reward = 0
            while not done:
                action, lstm_state, agent_info = self.agent.compute_single_action(
                    obs, state=lstm_state
                )
                obs, reward, done, env_info = env.step(action)
                episode_reward += reward
                if render:
                    env.render()

            episode_rewards[e] = episode_reward
            obs = env.reset()
            lstm_state = [
                torch.zeros(self.config["model"]["lstm_cell_size"]),
                torch.zeros(self.config["model"]["lstm_cell_size"]),
            ]

        return episode_rewards


def main():
    """Main function."""
    # parser = ArgumentParser()
    results_dir = Path("/home/jakob/ray_results_from_vm/ray_results/PPO")
    run_dir = results_dir / "PPO_AttackSimulationEnv_23b1f_00002_2_seed=3_2022-02-07_15-25-42"
    # parser.add_argument("checkpoint", type=str, help="Path to RLLib run to load model from.")

    # args = parser.parse_args()
    checkpoint = 0

    checkpoint_folder = reversed(sorted((run_dir.glob("checkpoint_*")))).__next__()
    for f in checkpoint_folder.glob("checkpoint-*"):
        if re.match(r"checkpoint-\d\d?\d?$", f.name):
            checkpoint = str(f)
            break

    evaluator = RLLibEvaluator(checkpoint)

    reward = evaluator.test(1, render=True)
    print(np.mean(reward))


if __name__ == "__main__":
    main()
