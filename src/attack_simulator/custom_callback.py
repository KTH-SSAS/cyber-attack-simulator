from typing import Any, Dict, Optional

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


class AttackSimCallback(DefaultCallbacks):
    """Custom callback for AttackSim env."""

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

        for key in [
            "attacker_reward",
        ]:
            episode.custom_metrics[key] = info[key]

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs: Any,
    ) -> None:

        info = episode.last_info_for()

        for key in [
            "attacker_start_time",
            "num_defenses_activated",
            "num_services_online",
            "num_compromised_steps",
            "num_compromised_flags",
        ]:
            episode.custom_metrics[key] = info[key]
