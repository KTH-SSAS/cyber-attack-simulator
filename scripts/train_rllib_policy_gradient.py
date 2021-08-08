#!/usr/bin/env python3

import ray
from ray.rllib.agents import pg

from attack_simulator.env import AttackSimulationEnv

ray.init()
trainer = pg.PGTrainer(
    env=AttackSimulationEnv,
    config={
        "env_config": {},  # config to pass to env class
    },
)

while True:
    trainer.train()
