#!/usr/bin/env python3

import random

import numpy as np

from attack_simulator.graph import AttackGraph


class Attacker:
    def __init__(self, attack_graph, compromised_steps):
        self.attack_graph = attack_graph
        self.compromised_steps = compromised_steps

    def binary_state(self, compromised):
        attack_step_names = list(self.attack_graph.attack_steps)
        state = [0] * len(self.attack_graph.attack_steps)
        for step_name in compromised:
            state[attack_step_names.index(step_name)] = 1
        return "".join(str(x) for x in state)

    def explore(self):
        self.states = set()
        self.counter = 0
        self.explore_recursive("internet.connect", set(["internet.connect"]))
        print("Total: " + str(len(self.states)))

    def explore_recursive(self, parent, compromised):
        self.compromised_steps = compromised
        self.counter += 1
        print(
            "Number of states: "
            + str(len(self.states))
            + ".  State: "
            + self.binary_state(compromised)
        )
        self.states.add(self.binary_state(compromised))
        for child in sorted(list(self.attack_surface())):
            new_compromised = set(compromised)
            new_compromised.add(child)
            if self.binary_state(new_compromised) not in self.states:
                self.explore_recursive(child, new_compromised)

    def get_step(self, name):
        return self.attack_graph.attack_steps[name]

    def attack_surface(self, debug=False):
        # The attack surface consists of all reachable but uncompromised attack steps.
        att_surf = set()
        for compromised_step_name in self.compromised_steps:
            for child_name in self.get_step(compromised_step_name).children:
                if self.get_step(child_name).enabled:
                    if self.get_step(child_name).step_type == "or":
                        att_surf.add(child_name)
                    else:
                        all_parents_are_compromised = True
                        for parent_name in self.get_step(child_name).parents:
                            if parent_name not in self.compromised_steps:
                                all_parents_are_compromised = False
                                break
                        if all_parents_are_compromised:
                            att_surf.add(child_name)

        att_surf -= set(self.compromised_steps)
        return att_surf


if __name__ == "__main__":
    attack_graph = AttackGraph()
    attacker = Attacker(attack_graph, ["internet.connect"])
    attacker.explore()

# vim: ft=python
