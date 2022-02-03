import pytest
from ray.rllib.agents import ppo


@pytest.mark.skip()
def test_ppo_trainer(rllib_config):

    trainer = ppo.PPOTrainer(config=rllib_config)

    # Perform single training iteration
    result = trainer.train()
    assert result["training_iteration"] == 1
    return
