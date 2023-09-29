import shutil
from pathlib import Path

import ray
from ray.rllib.agents import ppo
from ray.rllib.policy.policy import PolicySpec
import attack_simulator
from attack_simulator.constants import AGENT_ATTACKER, AGENT_DEFENDER
from attack_simulator.rllib.attackers_policies import BreadthFirstPolicy
from attack_simulator.rllib.defender_policy import DefenderConfig, DefenderPolicy

if __name__ == "__main__":

    ray.init()
    # Register everything with the rrlib registry.
    attack_simulator.register_rllib()

    seed = 0

    graph_config = {
        "ttc": {"easy": 5, "hard": 10, "default": 1},
        "rewards": {
            "high_flag": 10,
            "medium_flag": 10,
            "low_flag": 1,
            "default": 0.0,
            "defense_default": 10,
        },
        # "root": "asset:0:0",
        # "root": "internet.connect",
        # "filename": "graphs/big.yaml",
        "filename": "graphs/four_ways_mod.json",
    }

    env_config = {
        "backend": "rust",
        "attacker": "depth-first",
        "seed": seed,
        "save_graphs": False,
        "save_logs": False,
        "reward_mode": "downtime-penalty",
        "sim_config": {
            "seed": seed,
            "attack_start_time": 5,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
        },
        "run_id": "simple",
        "graph_config": graph_config,
    }

    #dummy_env = AttackSimulationEnv(EnvConfig(**env_config))

    render_path = Path("render/simple")
    if render_path.is_dir():
        shutil.rmtree(render_path)

    policy_ids = {AGENT_DEFENDER: AGENT_DEFENDER, AGENT_ATTACKER: AGENT_ATTACKER}

    config = (
        DefenderConfig()
        .training(scale_rewards=False)
        .framework("torch")
        .environment("AttackSimulationEnv", env_config=env_config)
        #.callbacks(AttackSimCallback)
        .debugging(seed=seed)
        .rollouts(
            num_rollout_workers=0,
            num_envs_per_worker=1,
        )
        .multi_agent(
            policies={
                AGENT_DEFENDER: PolicySpec(
                    policy_class=DefenderPolicy,
                    config={
                        "model": {
                            "custom_model": "GNNDefenderModel",
                            "fcnet_hiddens": [32],
                            "vf_share_layers": True,
                            "custom_model_config": {},
                        }
                    },
                ),
                AGENT_ATTACKER: PolicySpec(
                    BreadthFirstPolicy,
                    config={
                    },
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: policy_ids[agent_id],
            policies_to_train=[AGENT_DEFENDER],
        )
    )

    trainer = ppo.PPOTrainer(config=config)
    # trainer = dqn.DQN(config=config)
    # trainer = RandomDefender(config=config)
    # trainer = optimal_defender.TripwireDefender(config=config | {"simple_optimizer": True, "defense_steps": dummy_graph.attack_steps_by_defense_step})

    for i in range(1):
        result = trainer.train()

    print(result)
