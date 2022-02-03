from typing import Dict, Optional

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


class AttackSimCallback(DefaultCallbacks):
    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs,
    ) -> None:

        info = episode.last_info_for()

        episode.custom_metrics["compromised_steps"] = len(info["compromised_steps"])
        episode.custom_metrics["compromised_flags"] = len(info["compromised_flags"])
