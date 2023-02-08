from attack_simulator.constants import ACTION_WAIT, AGENT_ATTACKER, AGENT_DEFENDER
from attack_simulator.sim import AttackSimulator


def test_sim_init(simulator: AttackSimulator) -> None:
    assert simulator.time == 0


def test_sim_defender_action(simulator: AttackSimulator) -> None:
    pass


def test_sim_attacker_action(simulator: AttackSimulator) -> None:
    pass


def test_sim_observe(simulator: AttackSimulator) -> None:
    obs = simulator.get_obs_dict()


def test_sim_step(simulator: AttackSimulator) -> None:
    # Take a step
    simulator.step([(AGENT_DEFENDER, ACTION_WAIT), (AGENT_ATTACKER, ACTION_WAIT)])

    # Check that the time has advanced
    assert simulator.time == 1

    # Check that the observations are correct
    obs = simulator.get_obs_dict()
