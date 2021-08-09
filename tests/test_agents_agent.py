import pytest

from attack_simulator.agents import Agent


@pytest.mark.xfail(raises=TypeError)
def test_agents_agent():
    Agent()
