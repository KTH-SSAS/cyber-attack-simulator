"""Process graph description into an Attack Graph."""
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from yaml import safe_load

from attack_simulator.config import GraphConfig

import matplotlib.pyplot as plt

from .constant import AND, DEFENSE, OR


@dataclass
class AttackStep:
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

    def __init__(self, config: GraphConfig):

        self.config = config

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

        flags = {
            key: self.config.rewards.get(value.lower(), self.config.rewards["default"])
            for key, value in data["flags"].items()
        }

        self.defense_steps = {step.id: step for step in steps if step.step_type == DEFENSE}

        self.defense_costs = np.array([self.config.rewards["defense_default"] for _ in self.defense_steps.values()])

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

        # Don't iterate over flags to ensure determinism

        self.flags = np.array(
            [self.attack_indices[step] for step in self.attack_names if step in flags]
        )      
        flag_rewards = np.array([flags[step] for step in self.attack_names if step in flags])

        self.reward_params = np.zeros(len(self.attack_names))
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

    def is_defendable(self, step: int) -> bool:
        """Return True if the given step is defendable."""
        return len(self.defense_steps_by_attack_step[step]) > 0

    def get_defendable_steps(self) -> List[int]:
        return [i for i in self.attack_indices.values() if self.is_defendable(i)]

    def get_undefendable_steps(self) -> List[int]:
        return [i for i in self.attack_indices.values() if not self.is_defendable(i)]

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

    def is_traversible(
        self, node: int, defense_state: NDArray[np.int8], calculated_steps: Dict[int, bool]
    ) -> bool:
        """Check if a node is traversible given the current state of the
        defense."""
        if node in calculated_steps:
            return calculated_steps[node]

        logic, prequisites = self.attack_prerequisites[node]

        for p in filter(lambda p: p not in calculated_steps, prequisites):
            calculated_steps[p] = self.is_traversible(p, defense_state, calculated_steps)

        conditions_met = logic(calculated_steps[p] for p in prequisites) if prequisites else True
        not_defended = all(defense_state[self.defense_steps_by_attack_step[node]])

        return conditions_met and not_defended

    def get_traversable_steps(self, start_node: int, defense_state: NDArray[np.int8]) -> List[int]:
        """Get all reachable attack steps in the graph, given supplied current
        state vectors."""

        traversible_steps: Dict[int, bool] = {}
        traversible_steps[start_node] = self.is_traversible(
            start_node, defense_state, traversible_steps
        )

        if not traversible_steps[start_node]:
            return []

        return list(
            filter(
                lambda n: self.is_traversible(n, defense_state, traversible_steps),
                self.attack_indices.values(),
            )
        )

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

        steps_to_graph = self.get_traversable_steps(self.attack_indices[self.root], system_state)

        for step_idx in steps_to_graph:

            step_name = self.attack_names[step_idx]
            a_s = self.attack_steps[step_name]
            dict_t = asdict(a_s)

            # No need to add child information to node
            del dict_t["children"]
            del dict_t["id"]
            del dict_t["name"]

            if indices:
                dict_t["parents"] = list(map(lambda x: self.attack_indices[x], a_s.parents))

            # Add the attack step to the graph
            dig.add_node(step_idx if indices else step_name, **dict_t)

            # Add edges to children
            to_add = (
                (step_idx, child_index)
                for child_index in self.child_indices[step_idx]
                if child_index in steps_to_graph
            )

            if not indices:
                to_add = map(lambda x: (self.attack_names[x[0]], self.attack_names[x[1]]), to_add)

            dig.add_edges_from(to_add)

        return dig

    def step_is_defended(self, step: int, defense_state: np.ndarray) -> bool:
        defended = not all(defense_state[self.defense_steps_by_attack_step[step]])
        return defended


    def draw(self):

        # Get the graph
        graph = self.to_networkx(True, np.ones(len(self.defense_names)))

        # Get the positions of the nodes
        pos = nx.nx_pydot.graphviz_layout(graph, prog="dot", root=self.attack_indices[self.root])

        dpi = 100
        fig = plt.figure(dpi=dpi)

        pos = {int(key): value for key, value in pos.items()}

        and_edges = {
                (i, j)
                for i, j in graph.edges
                if self.attack_steps[self.attack_names[j]].step_type == AND
        }

        nx.draw_networkx_nodes(
            graph, pos=pos, edgecolors="black", node_size=100
        )
        nx.draw_networkx_edges(graph, edgelist=graph.edges-and_edges, pos=pos, edge_color="black")
        nx.draw_networkx_edges(
            graph, edgelist=and_edges, pos=pos, edge_color="black", style="dashed"
        )

        labels = nx.get_node_attributes(graph, "ttc")

        nx.draw_networkx_labels(graph, pos=pos, font_size=8)


        plt.axis("off")
        plt.tight_layout()

        # Show the plot
        plt.show()
