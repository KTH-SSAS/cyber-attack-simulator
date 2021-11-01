from argparse import ArgumentParser
from dataclasses import asdict

import torch
import yaml
from ray.rllib.agents import ppo

from attack_simulator.config import config_from_dicts
from attack_simulator.env import AttackSimulationEnv


class RLLibEvaluator:
    def __init__(self, checkpoint_path) -> None:

        model_config = {"use_lstm": True, "lstm_cell_size": 256}

        with open("config/default_env_config.yaml") as f:
            env_config_dict = yaml.safe_load(f)

        with open("config/default_graph_config.yaml") as f:
            graph_config_dict = yaml.safe_load(f)

        self.env_config, _ = config_from_dicts(graph_config_dict, env_config_dict)

        self.env_config.save_graphs = True
        self.env_config.save_text = True

        self.config = {
            "seed": 0,
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
                "explore": False,
                "env_config": {
                    "render_env": True,
                    # workaround for a bug in RLLib
                    # (https://github.com/ray-project/ray/issues/17921)
                    "replay_sequence_length": -1,
                },
            },
        }

        self.agent = ppo.PPOTrainer(config=self.config)
        self.agent.restore(checkpoint_path)

    def test(self):
        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        env = AttackSimulationEnv(asdict(self.env_config))

        # run until episode ends
        episode_reward = 0
        done = False
        obs = env.reset()

        lstm_state = [
            torch.zeros(self.config["model"]["lstm_cell_size"]),
            torch.zeros(self.config["model"]["lstm_cell_size"]),
        ]

        env.render()

        while not done:
            action, lstm_state, agent_info = self.agent.compute_single_action(obs, state=lstm_state)
            obs, reward, done, env_info = env.step(action)
            episode_reward += reward
            env.render()

        return episode_reward


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "-c", "--checkpoint", type=str, help="Path to RLLib checkpoint to load model from."
    )

    args = parser.parse_args()

    evaluator = RLLibEvaluator(args.checkpoint)
    evaluator.test()
