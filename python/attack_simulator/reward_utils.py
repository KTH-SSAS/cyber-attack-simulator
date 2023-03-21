import numpy as np
from numpy.typing import NDArray

FLOAT_TYPE = np.float64
INT_TYPE = np.int32


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
