from scripts.run_rllib import main

config_file = "config/maze_env_config.yaml"
stop_iterations = 1000
local_mode = False
wandb_sync = True

main(
    config_file,
    stop_iterations,
    local_mode,
    wandb_sync,
    gpu_count=0,
    num_workers=0,
    env_per_worker=1,
    stop_reward=None,
)
