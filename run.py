#!/home/jakob/miniconda3/envs/sim/bin/python

from scripts.run_rllib import main
import sys
from pathlib import Path

def run(config_file: Path, num_workers: int, debug: bool = False) -> None:
    stop_iterations = 500
    local_mode = debug
    wandb_sync = not debug

    main(
        config_file,
        stop_iterations,
        local_mode,
        wandb_sync,
        gpu_count=1,
        num_workers=num_workers,
        env_per_worker=1,
        stop_reward=None,
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit()
    config_file = Path(sys.argv[1])
    run(config_file, num_workers=10)