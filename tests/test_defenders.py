import pytest

from attack_simulator.optimal_defender import TripwirePolicy
from attack_simulator.random_defender import RandomPolicy
from attack_simulator.sim import AttackSimulator

NUM_TRIALS = 1000


@pytest.mark.parametrize(
    "attacker_class",
    [
        RandomPolicy,
        TripwirePolicy,
    ],
)
def test_sim_defender_actions(simulator: AttackSimulator, defender_class) -> None:
    done = False
    obs, _ = simulator.reset()

    total_ttc = simulator.ttc_total

    attacker: Agent = attacker_class(
        dict(
            attack_graph=simulator.g,
            num_special_actions=2,
            terminate_action=ACTION_TERMINATE,
            wait_action=ACTION_WAIT,
        )
    )

    while simulator.time < total_ttc and not done:
        action = attacker.compute_action_from_dict(obs)
        obs, info = simulator.step(OrderedDict([(AGENT_ATTACKER, action)]))

        assert info["prev_action_valid"][
            AGENT_ATTACKER
        ], f"Invalid attack step {action-attacker.num_special_actions}. Valid steps are {simulator.valid_attacks}."

        done = not obs["attack_surface"].any()

    assert done, "Attacker failed to explore all attack steps"
