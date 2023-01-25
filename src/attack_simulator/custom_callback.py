from typing import Any, Dict, Optional, Tuple

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID

from attack_simulator.constants import AGENT_DEFENDER

from .reward_utils import (
    defender_min,
    get_normalized_attacker_reward,
    get_normalized_downtime_reward,
    harmonic_mean,
    normalize,
)


class AttackSimCallback(DefaultCallbacks):
    """Custom callback for AttackSim env."""

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: EpisodeV2,
        **kwargs,
    ) -> None:

        info = episode._last_infos[AGENT_DEFENDER]

        for key in []:
            episode.custom_metrics[key] = info[key]

    def on_postprocess_trajectory(
        self,
        *,
        worker: "RolloutWorker",
        episode: EpisodeV2,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:

        # info = episode.last_info_for()
        # defense_costs = info["defense_costs"]
        # flag_costs = info["flag_costs"]
        # avg_defense_cost = np.mean(defense_costs)
        # avg_flag_cost = np.mean(flag_costs)
        # num_defenses = len(defense_costs)
        # num_flags = len(flag_costs)

        # min_defense_rewards = [-defense_cost_for_timestep(t+1, avg_defense_cost, num_defenses) for t in range(episode.length)]
        # min_flag_rewards = [-attack_cost_for_timestep(t+1, avg_flag_cost, num_flags) for t in range(episode.length)]

        # rewards = postprocessed_batch["rewards"]

        # postprocessed_rewards = [normalize(reward, min_d+min_a, 0, 0, 1) for reward, min_d, min_a in zip(rewards, min_defense_rewards, min_flag_rewards)]
        # rewards = postprocessed_batch["rewards"]
        # value_targets = postprocessed_batch["value_targets"]
        # postprocessed_batch["rewards"] =  rewards / sum(rewards)
        # postprocessed_batch["value_targets"] = (value_targets - sum(rewards))/(0 - sum(rewards))

        return None

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: EpisodeV2,
        **kwargs: Any,
    ) -> None:

        info = episode._last_infos[AGENT_DEFENDER]

        for key in [
            "attacker_start_time",
            "perc_defenses_activated",
            "perc_compromised_steps",
            "perc_compromised_flags",
            "sum_attacker_reward",
            "sum_defender_penalty",
        ]:
            episode.custom_metrics[key] = info[key]

        defense_costs = info["defense_costs"]
        avg_defense_cost = np.mean(defense_costs)
        num_defenses = defense_costs.shape[0]

        a_min = -np.sum(info["flag_costs"])

        attacker_reward = get_normalized_attacker_reward(
            info["flag_costs"], info["sum_attacker_reward"]
        )

        flag_defense_reward = 1 - attacker_reward

        reward_mode = worker.env_context["reward_mode"]
        if reward_mode == "downtime-penalty":
            d_min = defender_min(avg_defense_cost, num_defenses, episode.length)
            r_min = d_min + a_min
            r_max = 0
            defender_penalty = get_normalized_downtime_reward(
                info["sum_defender_penalty"], avg_defense_cost, num_defenses, episode.length
            )
            episode.custom_metrics["normalized_downtime_penalty"] = defender_penalty
            episode.custom_metrics["harmonic_mean_reward"] = harmonic_mean(
                [defender_penalty, flag_defense_reward]
            )

        elif reward_mode == "uptime-reward":
            r_min = a_min
            r_max = episode.length * sum(defense_costs)
        elif reward_mode == "defense-penalty":
            r_min = -np.sum(defense_costs) + a_min
            r_max = 0

        episode.custom_metrics["normalized_reward"] = normalize(episode.total_reward, r_min, r_max)
        episode.custom_metrics["normalized_flag_defense_reward"] = flag_defense_reward

        # num_alerts = info["num_alerts"]

        # p_alert = num_alerts / (episode.length * info['num_attack_steps'])

        # entropy = -(p_alert * np.log(p_alert) + (1 - p_alert) * np.log(1 - p_alert))

        # episode.custom_metrics["uncertainty"] = (1 - p_alert) * worker.env_context["false_positive"] + p_alert * worker.env_context["false_negative"]
