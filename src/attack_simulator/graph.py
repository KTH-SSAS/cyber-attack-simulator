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
from collections import deque
from agraphlib import STEP

@dataclass
class AttackStep:
    ttc: float
    id: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    asset: str = "asset"
    name: str = "attack"
    step_type: STEP = STEP.OR


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

        self.root = data["entry_points"][0]

        nodes = (node | {'step_type': STEP(node['step_type'])} for node in data["attack_graph"])
        nodes = (replace_all_templates(node, config) for node in nodes)
        steps = [AttackStep(**node) for node in nodes]


        self.defense_steps = {step.id : step for step in steps if step.id in data["defenses"]}
        self.defense_costs = np.array([1 for _ in self.defense_steps])

        # self.defense_costs = np.array([self.config.rewards["defense_default"] for _ in self.defense_steps.values()])

        self.attack_steps = {step.id: step for step in steps if step.id not in self.defense_steps}

        flags = {
            key: self.total_ttc * 1.5 #* len(self.defense_steps)
            for key in data["flags"]
        }

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
                any if self.attack_steps[attack_name].step_type == STEP.OR else all,
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
        self.flag_rewards = flag_rewards


        self.reward_params = np.zeros(len(self.attack_names))
        self.reward_params[self.flags] = flag_rewards

        self.child_indices = [
            [
                self.attack_indices[child_name]
                for child_name in self.attack_steps[attack_name].children
            ]
            for attack_name in self.attack_names
        ]
        pass

    @property
    def total_ttc(self) -> float:
        return sum(step.ttc for step in self.attack_steps.values())

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

    def is_traversable(self, step: int, attack_state: NDArray[np.int8]) -> bool:
        """Return True if the given step is traversable, i.e. all conditions are
        met."""
        logic, prerequisites = self.attack_prerequisites[step]
        return logic(attack_state[prerequisites]) if prerequisites else False

    def is_vulnerable(self, step: int, attack_state: NDArray[np.int8]) -> bool:
        """Return True if the given step is vulnerable, i.e. can be attacked."""
        traversable = self.is_traversable(step, attack_state)
        compromised = attack_state[step]
        return (not compromised   # Attack step isn't already compromised
                and traversable)  # Prerequisite(s) are compromised

    def get_vulnerable_children(
        self, attack_index: int, attack_state: np.ndarray
    ) -> List[int]:
        """Get all child steps of a step that can be attacked, given the state
        of compromised steps."""

        children = self.child_indices[attack_index]
        prerequisite_iterator = (self.attack_prerequisites[child] for child in children)

        reachable_steps = [
            child_index
            for (child_index, (logic, prerequisites)) in zip(
                children, prerequisite_iterator
            )
            if (  # Add step to attack surface if:
                not attack_state[child_index]  # Attack step isn't already compromised
                and logic(attack_state[prerequisites])  # Prerequisite(s) are compromised
            )
        ]

        return reachable_steps

    def get_traversable_steps(self, start_node: int, graph_state: NDArray[np.int8]) -> List[int]:
        """Get all traversable attack steps in the graph, from a given step, given supplied current
        state vectors. This includes attack steps that are already compromised,
        as well as those that can be attacked given the state."""

        targets = deque([start_node])
        discovered = set()

        while targets:
            
            node = targets.popleft()
            
            if node in discovered:
                continue

            discovered.add(node)
            if self.is_traversable(node, graph_state):
                targets.extend(self.child_indices[node])
    

        return list(discovered)

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

    def to_networkx(self, indices: bool, system_state: np.ndarray, add_defenses: bool = False) -> nx.DiGraph:
        """Convert the AttackGraph to an NetworkX DiGraph."""
        dig = nx.DiGraph()

        steps_to_graph = self.get_traversable_steps(self.attack_indices[self.root], system_state)

        current_index = 0
        for step_idx in steps_to_graph:
            current_index += 1

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

        if add_defenses:
            for defense, affected_step in zip(self.defense_names, self.attack_steps_by_defense_step):
                defense_index = self.defense_indices[defense]+current_index
                dig.add_node(defense_index if indices else defense, type="defense")
                for attack in affected_step:
                    dig.add_edge(defense_index if indices else defense, attack if indices else self.attack_names[attack])


        return dig

    def step_is_defended(self, step: int, defense_state: np.ndarray) -> bool:
        defended = not all(defense_state[self.defense_steps_by_attack_step[step]])
        return defended


    def draw(self) -> None:

        # Get the graph
        graph = self.to_networkx(True, np.ones(len(self.defense_names)), True)

        # Get the positions of the nodes
        pos = nx.nx_pydot.graphviz_layout(graph, prog="dot", root=self.attack_indices[self.root])

        dpi = 100
        fig = plt.figure(dpi=dpi)

        pos = {int(key): value for key, value in pos.items()}

        and_edges = {
                (i, j)
                for i, j in graph.edges
                if self.attack_steps[self.attack_names[j]].step_type == STEP.AND
        }

        nx.draw_networkx_nodes(
            graph, pos=pos, edgecolors="black", node_size=100
        )
        nx.draw_networkx_edges(graph, edgelist=graph.edges-and_edges, pos=pos, edge_color="black")
        nx.draw_networkx_edges(
            graph, edgelist=and_edges, pos=pos, edge_color="black", style="dashed"
        )

        labels = nx.get_node_attributes(graph, "ttc")

        nx.draw_networkx_labels(graph, labels=labels, pos=pos, font_size=8)


        plt.axis("off")
        plt.tight_layout()

        # Show the plot
        plt.show()
