from dataclasses import asdict

from ray.rllib.agents import ppo
from ray.rllib.policy.policy import PolicySpec

from attack_simulator import AGENT_ATTACKER, AGENT_DEFENDER
from attack_simulator.env.env import AttackSimulationEnv, register_rllib_env
from attack_simulator.rllib.custom_callback import AttackSimCallback
from attack_simulator.rllib.defender_policy import DefenderConfig, DefenderPolicy
from attack_simulator.rllib.random_defender import RandomPolicy


def test_ppo_trainer(env: AttackSimulationEnv):
    seed = 0

    env_name = register_rllib_env()

    policy_ids = {AGENT_DEFENDER: AGENT_DEFENDER, AGENT_ATTACKER: AGENT_ATTACKER}

    config = (
        DefenderConfig()
        .training(scale_rewards=False)
        .framework("torch")
        .environment(env_name, env_config=asdict(env.config))
        .callbacks(AttackSimCallback)
        .debugging(seed=seed)
        .rollouts(
            num_envs_per_worker=5,
        )
        .multi_agent(
            policies={
                AGENT_DEFENDER: PolicySpec(
                    policy_class=DefenderPolicy,
                    config={
                        "model": {
                            "custom_model": "DefenderModel",
                            "fcnet_hiddens": [1],
                            "vf_share_layers": True,
                            "custom_model_config": {},
                        }
                    },
                ),
                AGENT_ATTACKER: PolicySpec(
                    RandomPolicy,
                    config={
                        "num_special_actions": env.num_special_actions,
                        "wait_action": env.sim.wait_action,
                        "terminate_action": env.sim.terminate_action,
                    },
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: policy_ids[agent_id],
            policies_to_train=[AGENT_DEFENDER],
        )
    )

    trainer = ppo.PPOTrainer(config=config)
    for i in range(1):
        result = trainer.train()
    assert result
