#!/usr/bin/env python3

import os

import ray
from ray.rllib.agents import pg

from attack_simulator.env.env import AttackSimulationEnv

# listen on all interfaces inside a container for port-forwarding to work
dashboard_host = "0.0.0.0" if os.path.exists("/.dockerenv") else "127.0.0.1"
ray.init(dashboard_host=dashboard_host)
trainer = pg.PGTrainer(
    env=AttackSimulationEnv,
    config={
        "log_level": "DEBUG",  # show detailed info during training
        "framework": "torch",  # avoid pulling in TensorFlow
        "env_config": {"graph_size": "large"},  # config to pass to env class
    },
)

while True:
    trainer.train()
