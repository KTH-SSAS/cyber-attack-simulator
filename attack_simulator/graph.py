"""Process graph description into an Attack Graph."""
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from agraphlib import STEP, GraphColors
from numpy.typing import NDArray
from yaml import safe_load

from .config import GraphConfig
from .constants import UINT


@dataclass
class AttackStep:
    ttc: UINT
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

        with open(filename, "r", encoding="utf8") as yaml_file:
            data = safe_load(yaml_file.read())

        services = {x["id"]: x["dependents"] for x in data["instance_model"]}

        self.root: str = data["entry_points"][0]

        nodes = (node | {"step_type": STEP(node["step_type"])} for node in data["attack_graph"])
        nodes = (replace_all_templates(node, config) for node in nodes)

        self.steps = [AttackStep(**node) for node in nodes]

        self.defense_steps = {step.id: step for step in self.steps if step.id in data["defenses"]}
        self.defense_costs = np.array([1 for _ in self.defense_steps])

        self.attack_steps = {
            step.id: step for step in self.steps if step.id not in self.defense_steps
        }

        flags = {key: self.total_ttc * 1.5 for key in data["flags"]}  # * len(self.defense_steps)
        self.flags = flags

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
        self.attack_indices: Dict[str, UINT] = {
            name: index for (index, name) in enumerate(self.attack_names)
        }
        self.defense_indices = {name: index for (index, name) in enumerate(self.defense_names)}

        self.entry_points: List[str] = data["entry_points"]

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
            [self.attack_steps[attack_name].ttc for attack_name in self.attack_names],
            dtype=np.int64,
        )

        # Don't iterate over flags to ensure determinism

        self.flag_indices = np.array(
            [self.attack_indices[step] for step in self.attack_names if step in flags]
        )
        flag_rewards = np.array([flags[step] for step in self.attack_names if step in flags])
        self.flag_rewards = flag_rewards

        self.reward_params = np.zeros(len(self.attack_names))
        self.reward_params[self.flag_indices] = flag_rewards

        self.child_indices = [
            [
                self.attack_indices[child_name]
                for child_name in self.attack_steps[attack_name].children
            ]
            for attack_name in self.attack_names
        ]

    def get_edge_list(self) ->  NDArray[np.int64]:
        """Return the attack graph as an edge list."""
        children =  [
            (self.attack_indices[step], self.attack_indices[child])
            for step in self.attack_steps
            for child in self.attack_steps[step].children
        ]
        return np.array(children, dtype=np.int64)

    @property
    def total_ttc(self) -> UINT:
        return sum(step.ttc for step in self.attack_steps.values())

    @property
    def num_attacks(self) -> UINT:
        return len(self.attack_names)

    @property
    def num_defenses(self) -> UINT:
        return len(self.defense_steps)

    @property
    def num_services(self) -> UINT:
        return len(self.service_names)

    def __str__(self) -> str:
        label = os.path.basename(self.config.filename)
        return (
            f"{self.__class__.__name__}({label}, {self.num_services} services,"
            f" {self.num_attacks} attack steps)"
        )

    def is_defendable(self, step: UINT) -> np.bool_:
        """Return True if the given step is defendable."""
        return len(self.defense_steps_by_attack_step[step]) > 0

    def get_defendable_steps(self) -> List[UINT]:
        return [i for i in self.attack_indices.values() if self.is_defendable(i)]

    def get_undefendable_steps(self) -> List[UINT]:
        return [i for i in self.attack_indices.values() if not self.is_defendable(i)]

    def is_traversable(self, step: UINT, attack_state: NDArray[np.int8]) -> np.bool_:
        """Return True if the given step is traversable, i.e. all conditions
        are met."""
        logic, prerequisites = self.attack_prerequisites[step]
        return logic(attack_state[prerequisites]) if prerequisites else False

    def is_vulnerable(
        self, step: UINT, attack_state: NDArray[np.int8], defense_state: NDArray[np.int8]
    ) -> np.bool_:
        """Return True if the given step is vulnerable, i.e. can be
        attacked."""
        traversable = self.is_traversable(step, attack_state)
        compromised = attack_state[step]
        defenses = self.defense_steps_by_attack_step[step]
        return (
            not compromised  # Attack step isn't already compromised
            and traversable  # Prerequisite(s) are compromised
            and all(
                defense_state[defenses]
            )  # A connected defense step isn't activated (0 if activated)
        )

    def get_vulnerable_children(
        self, attack_index: UINT, attack_state: NDArray[np.int8], defense_state: NDArray[np.int8]
    ) -> List[UINT]:
        """Get all child steps of a step that can be attacked, given the state
        of compromised steps."""

        children = self.child_indices[attack_index]
        vulnerable_children = list(
            filter(lambda child: self.is_vulnerable(child, attack_state, defense_state), children)
        )

        return vulnerable_children

    def is_traversible(
        self,
        node: np.uintp,
        defense_state: NDArray[np.int8],
        calculated_steps: Dict[UINT, np.bool_],
    ) -> np.bool_:
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

    def get_traversable_steps(
        self, start_node: UINT, defense_state: NDArray[np.int8]
    ) -> List[UINT]:
        """Get all reachable attack steps in the graph, given supplied current
        state vectors."""

        traversible_steps: Dict[UINT, bool] = {}
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
        ttc: dict,
        filename: str = "graph.dot",
        verbose: bool = False,
        indexed: bool = False,
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

    def to_networkx(
        self, indices: bool, system_state: np.ndarray, add_defenses: bool = False
    ) -> nx.DiGraph:
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
            for defense, affected_step in zip(
                self.defense_names, self.attack_steps_by_defense_step
            ):
                defense_index = self.defense_indices[defense] + current_index
                dig.add_node(defense_index if indices else defense)
                for attack in affected_step:
                    dig.add_edge(
                        defense_index if indices else defense,
                        attack if indices else self.attack_names[attack],
                    )

        return dig

    def step_is_defended(self, step: UINT, defense_state: np.ndarray) -> bool:
        defended = not all(defense_state[self.defense_steps_by_attack_step[step]])
        return defended

    def interpret_services(self, services: np.ndarray) -> List[str]:
        return list(np.array(self.service_names)[np.flatnonzero(services)])

    def interpret_defenses(self, active_defenses: np.ndarray) -> List[str]:
        return [name for name, state in zip(self.defense_names, active_defenses) if not state]

    def interpret_attacks(self, attacks: np.ndarray) -> List[str]:
        return list(np.array(self.attack_names)[np.flatnonzero(attacks)])

    def interpret_observation(self, observation: np.ndarray) -> Tuple[List[str], List[str]]:
        defenses = observation[: self.num_defenses]  # type: ignore[misc]
        attacks = observation[self.num_defenses :]  # type: ignore[misc]
        return self.interpret_defenses(defenses), self.interpret_attacks(attacks)

    def draw(self, width=500, height=500, add_defense=True) -> None:
        # Get the graph
        graph = self.to_networkx(False, np.ones(len(self.defense_names)), add_defense)

        attack_node_colors_dict = {
            step.id: GraphColors.NODE.value for step in self.steps if step.id not in self.flags
        }
        attack_node_colors_dict[self.root] = GraphColors.ENTRY.value

        for flag in self.flags:
            attack_node_colors_dict[flag] = GraphColors.FLAG.value

        for defense in self.defense_steps:
            attack_node_colors_dict[defense] = GraphColors.DEFENSE.value

        dpi = 100
        fig = plt.figure(figsize=(width // dpi, height // dpi), dpi=dpi)

        # Get the positions of the nodes
        pos = nx.nx_pydot.graphviz_layout(graph, prog="dot")

        # pos = {key): value for key, value in pos.items()}

        and_steps = set(map(lambda x: x.id, filter(lambda x: x.step_type == STEP.AND, self.steps)))
        and_edges = {(i, j) for i, j in graph.edges if j in and_steps}

        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            node_color=[attack_node_colors_dict[step] for step in graph.nodes()],
            edgecolors="black",
            node_size=100,
        )
        nx.draw_networkx_edges(graph, edgelist=graph.edges - and_edges, pos=pos, edge_color="black")
        nx.draw_networkx_edges(
            graph, edgelist=and_edges, pos=pos, edge_color="black", style="dashed"
        )

        labels = nx.get_node_attributes(graph, "ttc")

        nx.draw_networkx_labels(graph, labels=labels, pos=pos, font_size=8)

        plt.axis("off")
        plt.tight_layout()

        # Show the plot

        fig.canvas.draw()

        plt.show()
