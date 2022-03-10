"""Simple agents that can be used as baselines for performance"""
import numpy as np

from ..rng import get_rng
from .agent import Agent


class RandomAgent(Agent):
    """Agent that will pick a random action each turn."""

    def __init__(self, agent_config):
        super().__init__()
        self.rng, _ = get_rng(agent_config.get("random_seed"))
        self.num_actions = agent_config["num_actions"]

    def act(self, observation=None):
        return self.rng.integers(self.num_actions)


class SkipAgent(Agent):
    """Agent that will always select no action, i.e. does nothing (or "skips"), on each turn."""

    def __init__(self, agent_config=None):
        super().__init__()

    def act(self, observation=None):
        return 0


class DisableProbabilityAgent(Agent):
    """An agent that mixes `no action` and `random action` with a given `disable_probability`."""

    def __init__(self, agent_config):
        super().__init__()
        self.rng, _ = get_rng(agent_config.get("random_seed"))
        self.num_services = agent_config["num_actions"] - 1
        self.disable_probability = agent_config.get("disable_probability", 0.1)

    def act(self, observation):
        enabled_services = np.array(observation[: self.num_services])
        disable = self.rng.uniform(0, 1) < self.disable_probability
        return self.rng.choice(np.flatnonzero(enabled_services)) + 1 if disable else 0


class RuleBasedAgent(Agent):
    """
    Rule-based agent following a simple IF-THIS-THEN-THAT rule.

    IF the attacker seems to have compromised an attack step preceeding valuable step,
    THEN disable the corresponding service.
    """

    def __init__(self, agent_config):
        super().__init__()
        self.g = agent_config["attack_graph"]
        self.rewards = np.array(self.g.reward_params)
        self.attack_state = np.full(self.g.num_attacks, 0)

    def act(self, observation):
        service_state = np.array(observation[: self.g.num_services])
        attack_state = np.array(observation[self.g.num_services :])

        changed = (1 - self.attack_state) & attack_state
        self.attack_state = attack_state

        action = 0  # no action

        if any(changed):
            # pick the first change, there _should_ be only one at each step
            attack_index = np.flatnonzero(changed)[0]
            child_indices = self.g.child_indices[attack_index]

            # If an attack step has been compromised which has a valuable child
            # then disable the corresponding service.
            if any(self.rewards[child_indices] > 0):
                service_index = self.g.service_index_by_attack_index[attack_index]
                if all(service_state[service_index]):
                    action = service_index + 1  # + 1 because action == 0 is no action.
        return action


class NewRuleBasedAgent(RuleBasedAgent):
    """
    Rule-based agent following a simple IF-THIS-THEN-THAT rule.

    IF the attacker seems to have compromised an attack step preceeding valuable step,
    THEN disable the corresponding service.

    NewRuleBasedAgent differs from the original RuleBased agent in that
    * it only considers "eligible" children, i.e. those that have not been compromised
      and their underlying service is still enabled
    * the "corresponding service" is determined based on the valuable child not its parent

    NB: for the current pre-defined graphs, these distinctions likely do NOT lead to any change
    in the actual selection, since all "valuable children" are _flags_, and all flags happen have
    a single parent and share the underlying service with their parent...
    BUT, these properties need not hold in general (OR do they?!?).
    """

    def _should_disable(self, service_state, attack_index, child_index):
        return 0 < self.rewards[child_index]

    def act(self, observation):
        attack_state = np.array(observation[self.g.num_services :])
        service_state = np.array(observation[: self.g.num_services])

        changed = (1 - self.attack_state) & attack_state
        self.attack_state = attack_state

        action = 0  # no action

        if any(changed):
            # pick the first change, there _should_ be only one at each step
            attack_index = np.flatnonzero(changed)[0]

            # If an attack step has been compromised which has an "eligible" high-value child
            # then disable the corresponding service.
            eligible = self.g.get_eligible_indices(attack_index, attack_state, service_state)
            if any(eligible):
                most_valuable_child_index = eligible[np.argmax(self.rewards[eligible])]
                if self._should_disable(service_state, attack_index, most_valuable_child_index):
                    # + 1 because action == 0 is no action.
                    action = self.g.service_index_by_attack_index[most_valuable_child_index] + 1
        return action


class RiskAwareAgent(NewRuleBasedAgent):
    """
    Risk-aware agent that uses TTC *and* reward information to decide WHEN to disable a service.

    Greedy decision based on a single attack step, change of reward on next step:
      (a) disable service:  -S
      (b) no action:        -Q*A
    where
      * S is the reward for keeping the relevant service running,
      * Q is the [conditional] probability that the attack succeeds on the next step
      * A is the attackers reward for compromise
    NOTE: Q_1 = exp(-2/T), and Q_i = exp(-1/T) for all i > 1, where T is the expected TTC
    """

    def __init__(self, agent_config):
        super().__init__(agent_config)
        self.seen = np.full(self.g.num_attacks, 0)
        self.q = np.exp(-1 / np.array(self.g.ttc_params))

    def _should_disable(self, service_state, attack_index, child_index):
        # reward for running services
        service_index = self.g.service_index_by_attack_index[child_index]
        S = (
            0
            if service_index == -1
            else sum(service_state[self.g.dependent_services[service_index]])
        )

        # attackers reward for compromise
        A = self.rewards[child_index]

        # probability of compromise on next step
        Q = self.q[attack_index]

        if self.seen[attack_index] == 0:
            Q *= Q
            self.seen[attack_index] = 1

        return S < Q * A
