"""Process graph description into an Attack Graph."""
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import networkx as nx
import numpy as np
from yaml import safe_load

from attack_simulator.config import GraphConfig

from .constant import DEFENSE, OR


@dataclass
class AttackStep:
    reward: float
    ttc: float
    id: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    asset: str = "asset"
    name: str = "attack"
    step_type: str = OR


def replace_all_templates(node: Dict, config: GraphConfig) -> dict:
    attributes = replace_template(node, config.ttc, "ttc")
    attributes = replace_template(attributes, config.rewards, "reward")
    return attributes


def replace_template(node: Dict, templates: Dict, key: str) -> dict:
    attributes = node.copy()
    if key in attributes:
        entry = attributes[key]
        if isinstance(entry, str):
            try:
                attributes[key] = templates.get(entry.lower(), templates["default"])
            except KeyError:
                print(f"Missing configured value for {entry}, defaulting to 0.0")
                attributes[key] = 0.0
        elif entry is None:
            attributes[key] = templates["default"]
    else:
        attributes[key] = templates["default"]
    return attributes


class AttackGraph:
    """Attack Graph."""

    def __init__(self, config: Union[GraphConfig, dict]):

        self.config = GraphConfig(**config) if isinstance(config, dict) else config

        filename = self.config.filename

        # load the YAML graph spec
        
        script_path = Path(__file__)
        root_dir = script_path.parent.parent.parent

        with open(root_dir / filename, "r", encoding="utf8") as yaml_file:
            data = safe_load(yaml_file.read())

        services = {x["id"]: x["dependents"] for x in data["instance_model"]}


        steps = [
            AttackStep(**replace_all_templates(node, self.config)) for node in data["attack_graph"]
        ]

        flags = {key: self.config.rewards.get(value.lower(), self.config.rewards["default"]) for key, value in data["flags"].items()}

        self.defense_steps = {step.id: step for step in steps if step.step_type == DEFENSE}

        self.defense_costs = np.array([step.reward for step in self.defense_steps.values()])

        self.attack_steps = {step.id: step for step in steps if step.step_type != DEFENSE}

        # Add parents for attack steps.
        # Defense steps are not included as parents
        for step in self.attack_steps.values():

            step.parents = sorted(
                [
                    other_step.id
                    for other_step in self.attack_steps.values()
                    if step.id in other_step.children
                ]
            )

            # Ensure deterministic ordering of children
            step.children = sorted(step.children)

        # Store ordered list of names
        self.service_names: List[str] = sorted(services)
        self.defense_names: List[str] = sorted(self.defense_steps)
        self.attack_names: List[str] = sorted(self.attack_steps)

        # Store index-based attributes
        self.service_indices = {name: index for (index, name) in enumerate(self.service_names)}
        self.attack_indices = {name: index for (index, name) in enumerate(self.attack_names)}
        self.defense_indices = {name: index for (index, name) in enumerate(self.defense_names)}

        self.service_index_by_attack_index = [
            [
                self.service_indices[service_name]
                for service_name in self.attack_steps[attack_name].conditions
                if service_name in self.service_indices
            ]
            for attack_name in self.attack_names
        ]

        self.dependent_services = [
            [self.service_indices[dependent] for dependent in services[name]]
            for name in self.service_names
        ]

        self.attack_steps_by_defense_step = [
            [
                self.attack_indices[attack_step]
                for attack_step in self.defense_steps[defense_step].children
            ]
            for defense_step in self.defense_names
        ]

        self.defense_steps_by_attack_step = [
            [
                self.defense_indices[defense_name]
                for defense_name in self.defense_names
                if attack_step in self.defense_steps[defense_name].children
            ]
            for attack_step in self.attack_names
        ]

        self.attack_prerequisites = [
            (
                # logic function to combine prerequisites
                any if self.attack_steps[attack_name].step_type == OR else all,
                # prerequisite attack steps
                [
                    self.attack_indices[prerequisite_name]
                    for prerequisite_name in self.attack_steps[attack_name].parents
                ],
            )
            for attack_name in self.attack_names
        ]

        self.ttc_params = np.array(
            [self.attack_steps[attack_name].ttc for attack_name in self.attack_names]
        )

        reward_iterator = (
            self.attack_steps[attack_name].reward for attack_name in self.attack_names
        )

        # Don't iterate over flags to ensure determinism

        self.flags = np.array(
            [self.attack_indices[step] for step in self.attack_names if step in flags]
        )
        flag_rewards = np.array([flags[step] for step in self.attack_names if step in flags])

        self.reward_params = np.array(
            [reward if reward is not None else 0.0 for reward in reward_iterator], dtype=np.float64
        )
        self.reward_params[self.flags] = flag_rewards

        self.child_indices = [
            [
                self.attack_indices[child_name]
                for child_name in self.attack_steps[attack_name].children
            ]
            for attack_name in self.attack_names
        ]

    @property
    def root(self) -> str:
        return self.config.root

    @property
    def num_attacks(self) -> int:
        return len(self.attack_names)

    @property
    def num_defenses(self) -> int:
        return len(self.defense_steps)

    @property
    def num_services(self) -> int:
        return len(self.service_names)

    def __str__(self) -> str:
        label = os.path.basename(self.config.filename)
        return (
            f"{self.__class__.__name__}({label}, {self.num_services} services,"
            f" {self.num_attacks} attack steps)"
        )

    def get_reachable_steps(
        self, attack_index: int, attack_state: np.ndarray, defense_state: np.ndarray
    ) -> List[int]:
        """Get all reachable attack steps in the graph, given supplied current
        state vectors."""

        children = self.child_indices[attack_index]
        prerequisite_iterator = (self.attack_prerequisites[child] for child in children)
        defense_iterator = (
            defense_state[self.defense_steps_by_attack_step[child]] for child in children
        )

        reachable_steps = [
            child_index
            for (child_index, (logic, prerequisites), defenses) in zip(
                children, prerequisite_iterator, defense_iterator
            )
            if (  # Add step to attack surface if:
                not attack_state[child_index]  # Attack step isn't already compromised
                and logic(attack_state[prerequisites])  # Prerequisite(s) are compromised
                and all(defenses)  # A connected defense step isn't activated (0 if activated)
            )
        ]

        return reachable_steps

    def save_graphviz(
        self,
        filename: str = "graph.dot",
        verbose: bool = False,
        indexed: bool = False,
        ttc: dict = {},
    ) -> None:
        if indexed:
            index = {name: i + 1 for i, name in enumerate(self.attack_names)}

        def label(key: str) -> str:
            return key if not indexed else f"{index[key]} :: {key}"

        with open(filename, "w", encoding="utf8") as f:
            f.write("digraph G {\n")
            for key in self.attack_names:
                node = self.attack_steps[key]
                node_ttc = node.ttc if ttc is None else ttc[key]
                for child in node.children:
                    child_node = self.attack_steps[child]
                    child_ttc = child_node.ttc if ttc is None else ttc[child]
                    f.write(
                        f'"{label(key)}, {node_ttc:.0f}"'
                        f' -> "{label(child)}, {child_ttc:.0f}";\n'
                    )
            f.write("}\n")
        print(f"Generated a GraphViz file {filename} of the attack graph.\n")
        if verbose:
            print(
                "This can be converted to a PDF file locally, by running:\n"
                f"   dot -Tpdf {filename} -o {filename[:-3]}pdf\n"
                "or viewed online at, e.g., https://dreampuf.github.io/GraphvizOnline."
            )

    def to_networkx(self, indices: bool, system_state: np.ndarray) -> nx.DiGraph:
        """Convert the AttackGraph to an NetworkX DiGraph."""
        dig = nx.DiGraph()

        for name, a_s in self.attack_steps.items():
            dict_t = asdict(a_s)

            as_idx = self.attack_indices[name]

            # No need to add child information to node
            del dict_t["children"]
            del dict_t["id"]
            del dict_t["name"]
            # del dict_t["parents"]

            if not self.step_is_reachable(as_idx, system_state):
                # If any of the attack steps conditions are not fulfilled,
                # do not add it to the graph
                continue

            # Add the attack step to the graph

            node_name = as_idx if indices else name

            if indices:
                dig.add_edges_from(
                    (
                        (as_idx, child_index)
                        for child_index in self.child_indices[as_idx]
                        if self.step_is_reachable(child_index, system_state)
                    )
                )
            else:
                dig.add_edges_from(
                    (
                        (name, child)
                        for child in a_s.children
                        if self.step_is_reachable(self.attack_indices[child], system_state)
                    )
                )

            dig.add_node(node_name, **dict_t)

        return dig

    def step_is_reachable(self, step: int, defense_state: np.ndarray) -> bool:
        defense_enabled = all(defense_state[self.defense_steps_by_attack_step[step]])
        return defense_enabled
