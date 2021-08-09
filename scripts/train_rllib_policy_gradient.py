#!/usr/bin/env python3

import ray
from ray.rllib.agents import pg

from attack_simulator.env import AttackSimulationEnv

ray.init()
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
