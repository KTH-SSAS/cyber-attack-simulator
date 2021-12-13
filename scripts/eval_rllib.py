from argparse import ArgumentParser
from dataclasses import asdict

import torch
import yaml
from ray.rllib.agents import ppo
from attack_simulator.config import EnvConfig

from attack_simulator.config_util import config_from_dicts
from attack_simulator.env import AttackSimulationEnv
import ray
import numpy as np
class RLLibEvaluator:
    def __init__(self, checkpoint_path) -> None:

        ray.init(local_mode=True)

        model_config = {"use_lstm": True, "lstm_cell_size": 256}

        self.env_config: EnvConfig = EnvConfig.from_yaml("config/default_env_config.yaml")

        self.env_config = self.env_config.replace(save_graphs=True, save_logs=True, seed=5, false_positive=0.0)

        self.config = {
            "seed": self.env_config.seed,
            "framework": "torch",
            "env": AttackSimulationEnv,
            "env_config": asdict(self.env_config),
            "num_workers": 0,
            "model": model_config,
            "in_evaluation": True,
            "evaluation_num_workers": 1,
            "evaluation_interval": 1,
            # Run 1 episode each time evaluation runs.
            "evaluation_num_episodes": 1,
            "evaluation_config": {
                "explore": False
            },
        }

        self.agent = ppo.PPOTrainer(config=self.config)
        self.agent.restore(checkpoint_path)
        self.run_id = checkpoint_path.split('/')[-3]

    def test(self, episodes, render=False):
        """Test trained agent for a number of episodes. Return the episode reward"""
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
                action, lstm_state, agent_info = self.agent.compute_single_action(obs, state=lstm_state)
                obs, reward, done, env_info = env.step(action)
                episode_reward += reward
                if render:
                    env.render()

            episode_rewards[e] = episode_reward
            env.reset()
        
        return episode_rewards


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "checkpoint", type=str, help="Path to RLLib checkpoint to load model from."
    )

    args = parser.parse_args()

    evaluator = RLLibEvaluator(args.checkpoint)
    
    reward = evaluator.test(1, render=True)
    print(np.mean(reward))
