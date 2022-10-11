from typing import Any, Dict, Optional, Tuple

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID, AgentID
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np

def defender_min(avg_defense_cost, num_defenses, episode_length):
    return -avg_defense_cost * (sum(range(1, num_defenses)) + num_defenses * episode_length)

def normalize(value, min_val, max_val, low_bound=0, upper_bound=1):
    return low_bound + (value - min_val)*(upper_bound-low_bound)/(max_val - min_val)

def harmonic_mean(values):
    return len(values) / sum([1 / value for value in values if value > 0])

def defense_cost_for_timestep(timestep, defense_cost, num_defenses):
    if timestep < num_defenses:
        return timestep*defense_cost
    else:
        return num_defenses*defense_cost

def attack_cost_for_timestep(timestep, attack_cost, num_attack_steps):
    if timestep < num_attack_steps:
        return timestep*attack_cost
    else:
        return num_attack_steps*attack_cost

def cumulative_defense_cost_for_timestep(timestep, defense_cost, num_defenses):
    return sum((defense_cost_for_timestep(t+1, defense_cost, num_defenses) for t in range(timestep)))


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

        postprocessed_batch["rewards"] =  postprocessed_batch["rewards"] /  sum(postprocessed_batch["rewards"])

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
            "sum_attacker_reward",
            "sum_defender_penalty",
        ]:
            episode.custom_metrics[key] = info[key]

        reward_mode = worker.env_context["reward_mode"]

        defense_costs = info["defense_costs"]
        avg_defense_cost = np.mean(defense_costs)
        num_defenses = len(defense_costs)

        d_max = 0
        d_min = defender_min(avg_defense_cost, num_defenses, episode.length)
        defender_penalty = normalize(info["sum_defender_penalty"], d_min, d_max)
        episode.custom_metrics["normalized_downtime_penalty"] = defender_penalty

        a_min = -sum(info["flag_costs"])
        a_max = 0
        attacker_reward = normalize(info["sum_attacker_reward"], a_min, a_max)
        episode.custom_metrics["normalized_attacker_reward"] = attacker_reward

        episode.custom_metrics["harmonic_mean_reward"] = harmonic_mean([defender_penalty, attacker_reward])

        if reward_mode == "downtime-penalty":
            r_min = d_min - a_min
            r_max = 0
        elif reward_mode == "uptime-reward":
            r_min = -a_min
            r_max = episode.length * sum(defense_costs)
        elif reward_mode == "defense-penalty":
            r_min = -(sum(defense_costs) + a_min)
            r_max = 0

        episode.custom_metrics["normalized_reward"] = normalize(episode.total_reward, r_min, r_max)

        # num_alerts = info["num_alerts"]

        # p_alert = num_alerts / (episode.length * info['num_attack_steps'])

        # entropy = -(p_alert * np.log(p_alert) + (1 - p_alert) * np.log(1 - p_alert))

        # episode.custom_metrics["uncertainty"] = (1 - p_alert) * worker.env_context["false_positive"] + p_alert * worker.env_context["false_negative"]
        pass
