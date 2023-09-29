import json
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path("~/ray_results").expanduser()
path = base_dir / "Defender_2023-05-17_11-49-20/"

population_dirs = [p for p in path.iterdir() if p.is_dir() and p.name.startswith("Defender")]

params_to_plot = ["lr", "vf_loss_coeff", "lambda", "clip_param"]
fig, axs = plt.subplots(1, len(params_to_plot))

for p in population_dirs:

    with open(p / "result.json") as f:

        configs = [json.loads(line)["config"] for line in f]

    data = {param: [config[param] for config in configs] for param in params_to_plot}

    for i, param in enumerate(params_to_plot):
        axs[i].plot(data[param])
        axs[i].set_title(param)

plt.show()
