#!/home/jakob/miniconda3/envs/sim/bin/python

from scripts.run_rllib import main

def run(debug: bool = False) -> None:
    config_file = "config/graph_sweep.yaml"
    stop_iterations = 200
    local_mode = debug
    wandb_sync = not debug

    main(
        config_file,
        stop_iterations,
        local_mode,
        wandb_sync,
        gpu_count=1,
        num_workers=1,
        env_per_worker=1,
        stop_reward=None,
    )

if __name__ == "__main__":
    run()