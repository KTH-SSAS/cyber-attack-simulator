def defender_min(avg_defense_cost: float, num_defenses: int, episode_length: int):
    return -avg_defense_cost * (sum(range(1, num_defenses)) + num_defenses * episode_length)


def normalize(
    value: float, min_val: float, max_val: float, low_bound: float = 0, upper_bound: float = 1
) -> float:
    return low_bound + (value - min_val) * (upper_bound - low_bound) / (max_val - min_val)


def harmonic_mean(values):
    return len(values) / sum([1 / value for value in values if value > 0])


def action_cost_for_timestep(timestep: int, action_cost: float, num_actions: int) -> float:
    if timestep < num_actions:
        return timestep * action_cost
    else:
        return num_actions * action_cost


def cumulative_defense_cost_for_timestep(timestep, defense_cost, num_defenses):
    return sum(
        (action_cost_for_timestep(t + 1, defense_cost, num_defenses) for t in range(timestep))
    )


def get_normalized_downtime_reward(
    defender_penalty: float, defense_cost: float, num_defenses: int, episode_length: int
) -> float:
    d_max = 0
    d_min = defender_min(defense_cost, num_defenses, episode_length)
    defender_penalty = normalize(defender_penalty, d_min, d_max)
    return defender_penalty


def get_normalized_defense_reward(flag_costs, sum_attacker_reward) -> float:
    a_min = -sum(flag_costs)
    a_max = 0
    attacker_reward = normalize(sum_attacker_reward, a_min, a_max)
    return attacker_reward


def get_minimum_rewards(action_cost, num_actions, length) -> float:
	return [-action_cost_for_timestep(t+1, action_cost, num_actions) for t in range(length)]