import shelve

with shelve.open("baselines/eval_05065eval_attackers") as db:

    keys = sorted(db.keys())
    for key in keys:
        reward_mean = db[key]["evaluation"]["episode_reward_mean"]
        perc_flags = db[key]["evaluation"]["custom_metrics"]["perc_compromised_flags_mean"]
        mean_length = db[key]["evaluation"]["episode_len_mean"]
        min_length = min(db[key]["evaluation"]["hist_stats"]["episode_lengths"])
        # fpr = db[key]['config']['env_config']['false_positive']
        # fnr = db[key]['config']['env_config']['false_negative']
        attacker = db[key]["config"]["env_config"]["attacker"]
        old_attacker = db[key]["config"]["old_attacker"]
        # print(f"{key:>10} fp: {fpr:>3}, fn: {fnr:>3}, perc_flags: {perc_flags:>6}, reward_mean: {reward_mean:>10}, mean_length: {mean_length:>10}, min_length: {min_length:>10}")
        print(
            f"{key:>10} a_t: {old_attacker:>3}, a_e: {attacker:>3}, perc_flags: {perc_flags:>6}, reward_mean: {reward_mean:>10}, mean_length: {mean_length:>10}, min_length: {min_length:>10}"
        )
