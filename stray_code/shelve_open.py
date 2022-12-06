import shelve 

with shelve.open("eval_09041eval_hipenalty") as db:

	keys = sorted(db.keys())
	for key in keys:
		reward_mean = db[key]['evaluation']['episode_reward_max']
		perc_flags = db[key]['evaluation']['custom_metrics']['perc_compromised_flags_max']
		mean_length = db[key]['evaluation']['episode_len_mean']
		min_length = min(db[key]['evaluation']['hist_stats']['episode_lengths'])
		fpr = db[key]['config']['env_config']['false_positive']
		fnr = db[key]['config']['env_config']['false_negative']
		print(f"{key:>10} fp: {fpr:>3}, fn: {fnr:>3}, perc_flags: {perc_flags:>6}, reward_mean: {reward_mean:>10}, mean_length: {mean_length:>10}, min_length: {min_length:>10}")