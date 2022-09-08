from typing import Any, Dict, Optional, Tuple

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID, AgentID
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np

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

        for key in []:
            episode.custom_metrics[key] = info[key]

    def on_postprocess_trajectory(
        self,
        *,
        worker: "RolloutWorker",
        episode: Episode,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        # TODO normalize rewards?
        return None

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
            "perc_defenses_activated",
            "perc_assets_online",
            "perc_compromised_steps",
            "perc_compromised_flags",
            "attacker_reward",
        ]:
            episode.custom_metrics[key] = info[key]

        reward_mode = worker.env_context["reward_mode"]

        if reward_mode == "downtime-penalty":
            r_min = -((episode.length * info["max_defense_cost"]) + info["max_attack_cost"])
            r_max = 0
        elif reward_mode == "uptime-reward":
            r_min = -info["max_attack_cost"]
            r_max = episode.length * info["max_defense_cost"]
        elif reward_mode == "defense-penalty":
            r_min = -(info["max_defense_cost"] + info["max_attack_cost"])
            r_max = 0

        episode.custom_metrics["normalized_reward"] = (episode.total_reward - r_min) / (
            r_max - r_min
        )

        #num_alerts = info["num_alerts"]

        #p_alert = num_alerts / (episode.length * info['num_attack_steps'])

        #entropy = -(p_alert * np.log(p_alert) + (1 - p_alert) * np.log(1 - p_alert))

        #episode.custom_metrics["uncertainty"] = (1 - p_alert) * worker.env_context["false_positive"] + p_alert * worker.env_context["false_negative"]
        pass
