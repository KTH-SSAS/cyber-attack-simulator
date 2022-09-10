#!/usr/bin/env python

from pathlib import Path
from attack_simulator import evaluate
from attack_simulator.env import register_rllib_env
import attack_simulator.ids_model as ids_model
from tqdm import tqdm
import re
import ray


def get_checkpoint_path(run_dir: Path) -> Path:
	checkpoint_folder = reversed(sorted((run_dir.glob("checkpoint_*")))).__next__()
	# for f in checkpoint_folder.glob("checkpoint-*"):
	# 	if re.match(r"checkpoint-\d\d?\d?$", f.name):
	# 		checkpoint = Path(f)
	# 		break
	return checkpoint_folder

env_name = register_rllib_env()
ids_model.register_rllib_model()

parser = evaluate.create_parser()

ray_results = Path("/home/jakob/ray_results/")

sweeps = ["56ca0", "f74f5"]
local_mode = False
num_episodes = 500

ray.init(local_mode=local_mode)
for run_id in sweeps:
	for algorithm in ray_results.iterdir():
		run_folder = ray_results / algorithm
		result_folders = filter(lambda x: re.search(run_id, x.name), run_folder.iterdir())
		checkpoints = map(get_checkpoint_path, result_folders)
		for checkpoint in map(str, checkpoints):
			args = parser.parse_args([
				checkpoint,
				"--out",
				"eval_data",
				"--env",
				env_name,
				"--run",
				algorithm.name,
				"--episodes",
				str(num_episodes),
				"--steps",
				"0", # No limit on amount of steps
				"--use-shelve",
				#"--local-mode"
			])

			evaluate.run(args, parser)
			#ray.shutdown()	
