import numpy as np
from numpy.typing import NDArray
from itertools import repeat

FLOAT_TYPE = np.float64
INT_TYPE = np.int32

def scale_rewards(rewards, defense_costs, flag_costs):
    episode_length = len(rewards)
    num_defenses = len(defense_costs)

    # Assume the worst case where the defender immediately
    # enables the most expensive defenses first.
    sorted_costs = sorted(defense_costs, reverse=True)
    initial_cost = [sum(sorted_costs[:x]) for x in range(1, len(sorted_costs) + 1)]
    # Rewards after all defenses are enabled
    min_reward = [sum(sorted_costs)] * (episode_length - num_defenses)

    # Defense costs for entire episode
    min_reward = initial_cost + min_reward

    # worst case flag cost
    # worst case for a timestep is if the attacker grabs the most
    # expensive flag in that
    min_flag_cost = np.min(flag_costs)

    # Calculate the minimum reward for each timestep as the
    # worst case flag cost + the worst case defense cost
    min_flag_rewards = repeat(min_flag_cost, episode_length)
    total_min_reward_per_timestep = min_reward #map(sum, zip(min_reward, min_flag_rewards))

    # Scale rewards to be between 0 and 1
    scaled_rewards = map(
        lambda x: normalize(
            x[0], min_val=-x[1], max_val=0, low_bound=0, upper_bound=1
        ),
        zip(rewards, total_min_reward_per_timestep),
    )
    to_array = np.array(list(scaled_rewards))
    return to_array

def defender_min(
    avg_defense_cost: FLOAT_TYPE, num_defenses: INT_TYPE, episode_length: INT_TYPE
) -> FLOAT_TYPE:
    return -avg_defense_cost * (sum(range(1, num_defenses)) + num_defenses * episode_length)


def normalize(
    value: FLOAT_TYPE,
    min_val: FLOAT_TYPE,
    max_val: FLOAT_TYPE,
    low_bound: FLOAT_TYPE = FLOAT_TYPE(0.0),
    upper_bound: FLOAT_TYPE = FLOAT_TYPE(1.0),
) -> FLOAT_TYPE:
    return low_bound + (value - min_val) * (upper_bound - low_bound) / (max_val - min_val)


def harmonic_mean(values: NDArray[FLOAT_TYPE]) -> FLOAT_TYPE:
    values = values + FLOAT_TYPE(1e-6)
    return values.shape[0] / np.sum(1.0 / values)


def action_cost_for_timestep(
    timestep: INT_TYPE, action_cost: FLOAT_TYPE, num_actions: INT_TYPE
) -> FLOAT_TYPE:
    if timestep < num_actions:
        return FLOAT_TYPE(timestep) * action_cost
    return FLOAT_TYPE(num_actions) * action_cost


def cumulative_defense_cost_for_timestep(
    timestep: INT_TYPE, defense_cost: FLOAT_TYPE, num_defenses: INT_TYPE
) -> FLOAT_TYPE:
    return np.sum(
        [action_cost_for_timestep(t + 1, defense_cost, num_defenses) for t in np.arange(timestep)]
    )


def get_normalized_downtime_reward(
    defender_penalty: FLOAT_TYPE,
    defense_cost: FLOAT_TYPE,
    num_defenses: INT_TYPE,
    episode_length: INT_TYPE,
) -> FLOAT_TYPE:
    d_max = FLOAT_TYPE(0.0)
    d_min = defender_min(defense_cost, num_defenses, episode_length)
    defender_penalty = normalize(defender_penalty, d_min, d_max)
    return defender_penalty


def get_normalized_attacker_reward(
    flag_costs: NDArray[FLOAT_TYPE], sum_attacker_reward: FLOAT_TYPE
) -> FLOAT_TYPE:
    a_max = np.sum(flag_costs)
    a_min = FLOAT_TYPE(0.0)
    attacker_reward = normalize(sum_attacker_reward, a_min, a_max)
    return attacker_reward


def get_minimum_rewards(
    action_cost: FLOAT_TYPE, num_actions: INT_TYPE, episode_length: INT_TYPE
) -> NDArray[FLOAT_TYPE]:
    return np.array(
        [
            action_cost_for_timestep(t + 1, action_cost, num_actions)
            for t in np.arange(episode_length)
        ]
    )
