#!/usr/bin/env python3

import sys
from pathlib import Path

from scripts.run_rllib import main


def run(config_file: Path, num_workers: int, debug: bool = False) -> None:
    stop_iterations = 100
    local_mode = debug
    wandb_sync = not debug
    fail_fast = "raise" if debug else True

    main(
        config_file,
        stop_iterations,
        local_mode,
        wandb_sync,
        gpu_count=0,
        num_workers=num_workers,
        env_per_worker=1,
        stop_reward=None,
        fail_fast=fail_fast,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: debug.py <config_file>")
        sys.exit()
    run(Path(sys.argv[1]), num_workers=1)
