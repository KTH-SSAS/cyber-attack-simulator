from pathlib import Path
from ray.rllib import evaluate
from attack_simulator.env import register_rllib_env
from attack_simulator.ids_model import register_rllib_model
import re


def get_checkpoint_path(run_dir: Path) -> Path:
	checkpoint_folder = reversed(sorted((run_dir.glob("checkpoint_*")))).__next__()
	# for f in checkpoint_folder.glob("checkpoint-*"):
	# 	if re.match(r"checkpoint-\d\d?\d?$", f.name):
	# 		checkpoint = Path(f)
	# 		break
	return checkpoint_folder

env_name = register_rllib_env()
model_name = register_rllib_model()

parser = evaluate.create_parser()

ray_results = Path("/home/jakob/sentience/data/ray_results/")

algorithm = "PPO"

run_folder = ray_results / algorithm

run_id = "6ca42"

result_folders = filter(lambda x: re.search(run_id, x.name), run_folder.iterdir())

checkpoints = map(get_checkpoint_path, result_folders)

checkpoints = list(map(str, checkpoints))

args = parser.parse_args(checkpoints + [
	"--out",
	"eval_data.db",
	"--env",
	env_name,
	"--run",
	"PPO",
	"--episodes",
	"300",
	 "--steps",
	 "0", # No limit
	"--use-shelve",
])

evaluate.run(args, parser)
