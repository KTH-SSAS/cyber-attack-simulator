#!/home/jakob/miniconda3/envs/sim/bin/python

from scripts.run_rllib import main

def run(num_workers: int, debug: bool = False) -> None:
    config_file = "config/graph_sweep.yaml"
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
    run(num_workers=10)