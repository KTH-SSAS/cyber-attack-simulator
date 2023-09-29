"""Convert a RLLib checkpoint to a pytorch model file."""

import argparse
import json
from pathlib import Path

from ray.rllib.algorithms import Algorithm
import torch

from attack_simulator.env.env import register_rllib_env
from attack_simulator.rllib.gnn_model import register_rllib_model


def main():
    register_rllib_env()
    register_rllib_model()
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to RLLib checkpoint to load model from.")
    parser.add_argument("output_dir", type=Path, help="Path to save the model to.")
    args = parser.parse_args(
        [
            "/home/jakob/ray_results/Defender_2023-05-17_12-04-42/Defender_AttackSimulationEnv_3e896_00000_0_clip_param=0.2836,lambda=0.9838,lr=0.0001,sgd_minibatch_size=64,vf_loss_coeff=592.7216_2023-05-17_12-04-43/checkpoint_000191/",
            "weights",
        ]
    )

    checkpoint: str = args.checkpoint

    algo = Algorithm.from_checkpoint(
        checkpoint=checkpoint,
        policy_ids={"defender"},  # <- restore only those policy IDs here.
    )

    policy = algo.get_policy("defender")

    # Create a model card
    model_card = {
        "name": policy.config["model"]["custom_model"],
        "description": "Defender model trained with RLLib.",
        "obs_space": str(policy.observation_space.shape),
        "num_outputs": policy.model.num_outputs,
        "model_config": policy.config["model"],
    }

    # Save the model.
    # policy.export_model(args.output_dir)

    torch.save(policy.model.model.state_dict(), args.output_dir / "model.pt")

    with open(args.output_dir / "model_card.json", "w") as f:
        json.dump(model_card, f)


if __name__ == "__main__":
    main()
