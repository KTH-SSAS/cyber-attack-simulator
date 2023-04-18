from __future__ import annotations

from itertools import repeat
from logging import Logger
from typing import Callable, Dict, Optional

import numpy as np
import torch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
from ray.rllib.utils.typing import PartialAlgorithmConfigDict

from ..utils.reward_utils import get_minimum_rewards, normalize


class DefenderConfig(PPOConfig):
    # _allow_unknown_configs = True
    # _allow_unknown_subkeys = True

    def __init__(self, algo_class=None) -> None:
        super().__init__(algo_class=algo_class or PPO)
        # self._allow_unknown_configs = True
        # self._allow_unknown_subkeys = True

        self.scale_rewards = True
        self.num_special_actions = 0

    def training(
        self,
        *,
        scale_rewards: Optional[bool] = NotProvided,
        num_special_actions: Optional[int] = NotProvided,
        **kwargs,
    ):
        super().training(**kwargs)
        self.scale_rewards = self.scale_rewards if scale_rewards is NotProvided else scale_rewards
        self.num_special_actions = (
            self.num_special_actions if num_special_actions is NotProvided else num_special_actions
        )
        return self

    def update_from_dict(self, config_dict: PartialAlgorithmConfigDict) -> DefenderConfig:
        # These get added in:
        # ray/tune/execution/ray_trial_executor.py:_setup_remote_runner()
        # for whatever reason. These trigger an error as DefenderConfig gets
        # recreated between trials and the updated config (including these keys)
        # is used to instantiate it.
        config_dict.pop("__stdout_file__", None)
        config_dict.pop("__stderr_file__", None)
        config_dict.pop("__trial_info__", None)
        return super().update_from_dict(config_dict)


class Defender(PPO):
    def __init__(
        self,
        config: Optional[AlgorithmConfig] = None,
        env=None,
        logger_creator: Optional[Callable[[], Logger]] = None,
        **kwargs,
    ):
        # ray.train.torch.enable_reproducibility(config['seed'])
        super().__init__(config, env, logger_creator, **kwargs)

    def get_default_policy_class(self, config):
        return DefenderPolicy

    def reset_config(self, new_config: Dict):
        self.config = new_config
        return True

    @classmethod
    def get_default_config(cls) -> DefenderConfig:
        return DefenderConfig()


class DefenderPolicy(PPOTorchPolicy):
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        with torch.no_grad():
            if self.config["scale_rewards"] is True:

                # sample_batch["rewards"] = rewards / len(rewards)

                try:
                    rewards = sample_batch["rewards"]
                    info = sample_batch["infos"][-1]
                    episode_length = len(rewards)
                    num_defenses = len(info["defense_costs"])

                    # Assume the worst case where the defender immediately
                    # enables the most expensive defenses first.
                    sorted_costs = sorted(info["defense_costs"], reverse=True)
                    initial_cost = [sum(sorted_costs[:x]) for x in range(1, len(sorted_costs) + 1)]
                    # Rewards after all defenses are enabled
                    min_reward = [sum(sorted_costs)] * (episode_length - num_defenses)

                    # Defense costs for entire episode
                    min_reward = initial_cost + min_reward

                    # worst case flag cost
                    # worst case for a timestep is if the attacker grabs the most
                    # expensive flag in that
                    min_flag_cost = np.min(info["flag_costs"])

                    # Calculate the minimum reward for each timestep as the
                    # worst case flag cost + the worst case defense cost
                    min_flag_rewards = repeat(min_flag_cost, episode_length)
                    total_min_reward_per_timestep = map(sum, zip(min_reward, min_flag_rewards))

                    # Scale rewards to be between 0 and 1
                    scaled_rewards = map(
                        lambda x: normalize(
                            x[0], min_val=-x[1], max_val=0, low_bound=0, upper_bound=1
                        ),
                        zip(rewards, total_min_reward_per_timestep),
                    )
                    sample_batch["rewards"] = np.array(list(scaled_rewards))

                except IndexError:
                    pass

            return compute_gae_for_sample_batch(self, sample_batch, other_agent_batches, episode)
