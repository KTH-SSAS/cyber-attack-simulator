from typing import Dict
import numpy as np

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class AttackSimCallback(DefaultCallbacks):

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        # Make sure this episode is really done.
        assert episode.batch_builder.policy_collectors["default_policy"].buffers["dones"][-1], (
            "ERROR: `on_episode_end()` should only be called " "after episode is done!"
        )

        info = episode.last_info_for()

        episode.custom_metrics["compromised_steps"] = len(info["compromised_steps"])
        episode.custom_metrics["compromised_flags"] = len(info["compromised_flags"])
