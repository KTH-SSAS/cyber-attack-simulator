#!/usr/bin/env python3


from attack_simulator.sim.graph import AttackGraph
from attack_simulator.utils.config import GraphConfig


class Attacker:
    def __init__(self, attack_graph, compromised_steps):
        self.steps = attack_graph.attack_steps
        self.compromised = compromised_steps

        self.indicator = dict()
        self.width = 0
        indicator = 1
        for attack_step in self.steps:
            self.indicator[attack_step] = indicator
            indicator <<= 1
            self.width += 1

    def state(self, compromised):
        state = 0
        for attack_step in compromised:
            state |= self.indicator[attack_step]
        return state

    def explore(self, verbose=False):
        self.states = set()
        self.counter = 0
        self.verbose = verbose
        print(self.compromised)
        self.explore_recursive(set(self.compromised))
        print("Total: " + str(len(self.states)))

    def explore_recursive(self, compromised):
        self.counter += 1
        state = self.state(compromised)
        if self.verbose:
            print(
                f"Number of states: {len(self.states)} ({self.counter})."
                f"  State: {state:0{self.width}b}"
            )
        self.states.add(state)
        self.compromised = compromised
        for child in sorted(list(self.attack_surface())):
            new_state = state | self.indicator[child]
            if new_state not in self.states:
                self.explore_recursive(compromised | set((child,)))

    def attack_surface(self):
        # The attack surface consists of all reachable but uncompromised attack steps.
        surface = set()
        for compromised_name in self.compromised:
            for child_name in self.steps[compromised_name].children:
                if self.steps[child_name].step_type == "or" or all(
                    [
                        parent_name in self.compromised
                        for parent_name in self.steps[child_name].parents
                    ]
                ):
                    surface.add(child_name)

        surface -= set(self.compromised)
        return surface


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="State explorer.")
    parser.add_argument(
        "-c",
        "--compromised_steps",
        type=str,
        default="internet.connect",
        help="Start exploring from this comma-separated list of compromised attack steps.",
    )
    parser.add_argument(
        "-v", "--verbose", type=bool, default=False, help="Print each state being explored."
    )
    args = parser.parse_args()

    config = GraphConfig.from_yaml("config/default_env_config.yaml")
    config = config.replace(graph_size=args.graph_size)
    attack_graph = AttackGraph(config)
    attacker = Attacker(attack_graph, args.compromised_steps.split(","))
    attacker.explore(args.verbose)
