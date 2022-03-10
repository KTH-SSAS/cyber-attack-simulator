"""Process YAML graph description into an Attack Graph."""
import dataclasses
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Set, Union

import networkx as nx
import numpy as np
from yaml import safe_load

from attack_simulator.config import GraphConfig


@dataclass
class AttackStep:
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    name: str = ""
    reward: float = 0.0
    step_type: str = "or"
    ttc: float = 1.0


SIZES: Dict[str, Set[str]] = {
    "tiny": set(
        (
            "lazarus.tomcat.connect",
            "energetic_bear.apache.connect",
            "sea_turtle.telnet.connect",
        )
    ),
    "small": set(
        (
            "lazarus.find_credentials",
            "energetic_bear.capture_traffic",
            "sea_turtle.telnet.connect",
        )
    ),
    "medium-small": set(("buckeye.find_vulnerability",)),
    "medium": set(
        (
            "lazarus.ftp.connect",
            "lazarus.tomcat.connect",
            "sea_turtle.telnet.connect",
        )
    ),
    "large": set(
        (
            "lazarus.ftp.connect",
            "energetic_bear.apache.connect",
        )
    ),
    "extra-large": set(("energetic_bear.apache.connect",)),
    "full": set(),
}


class AttackGraph:
    """Attack Graph."""

    def __init__(self, config: Union[GraphConfig, dict]):

        if isinstance(config, dict):
            config = GraphConfig(**config)

        # initialization
        self.attack_steps: Dict[str, AttackStep] = dict()
        services: Set[str] = set()

        # Declare attributes
        prune: Set[str] = set(config.prune)
        unmalleable_assets: Set[str] = set(config.unmalleable_assets)

        self.config = config

        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config.filename)

        # load the YAML graph spec
        with open(filename, "r", encoding="utf8") as yaml_file:
            data = safe_load(yaml_file.read())

        graph = data["attack_graph"]
        self.instance_model = data["instance_model"]

        for k in self.instance_model:
            # to deal with the concept of flags
            if "flag" in k:
                unmalleable_assets.add(k)
            if not k in unmalleable_assets:
                services.add(k)

        # traverse through the relevant subgraph for a given pass
        def traverse(perform_pass):
            keys = [self.root]
            while keys:
                next_keys = set()
                for key in keys:
                    node = graph[key]
                    children = set(node["children"]) - prune if "children" in node else set()
                    perform_pass(key, node, children)
                    next_keys |= set(children)
                keys = sorted(next_keys)

        # first pass: determine services and most fields
        def determine_fields(key: str, node: dict, children: set):
            # some leaf nodes have multiple parents, no need to re-process

            # handle `ttc` and `reward` "templates"
            for f in ("ttc", "reward"):
                if f in node:
                    # translate {easy,hard}_ttc and/or {early,late,final}_reward
                    # based on upper-case "TEMPLATE" in YAML to actual values
                    # TODO add better template handling
                    if isinstance(node[f], str):
                        node[f] = asdict(self.config)[node[f].lower()]

            if children:
                # Ensure a deterministic ordering of child nodes
                node["children"] = sorted(children)

            node["parents"] = set()

        # second pass: update parents
        def update_parents(key, _, children):
            for child in children:
                graph[child]["parents"].add(key)

        # third and final pass: freeze the relevant sub-graph
        def freeze_subgraph(key, node, _):
            # Ensure a deterministic ordering of parents
            node["parents"] = sorted(node["parents"])
            self.attack_steps[key] = AttackStep(**node)

        for perform_pass in (determine_fields, update_parents, freeze_subgraph):
            traverse(perform_pass)

        # set final attributes
        self.service_names: List[str] = sorted(services)
        self.attack_names: List[str] = sorted(self.attack_steps)

        # index-based attributes
        self.service_indices = {name: index for (index, name) in enumerate(self.service_names)}
        self.attack_indices = {name: index for (index, name) in enumerate(self.attack_names)}

        self.service_index_by_attack_index = []
        for attack_name in self.attack_names:
            attack_step = self.attack_steps[attack_name]
            indexes = []
            for service_name in attack_step.conditions:
                if service_name in self.service_indices:
                    service_index = self.service_indices[service_name]
                    indexes.append(service_index)
            self.service_index_by_attack_index.append(indexes)

        self.dependent_services = [
            [self.service_indices[dependent] for dependent in self.instance_model[name] if not dependent in unmalleable_assets]
            for name in self.service_names
        ]

        self.attack_prerequisites = [
            (
                # required services
                [
                    self.service_indices[service_name]
                    for service_name in self.attack_steps[attack_name].conditions if not service_name in unmalleable_assets
                ],
                # logic function to combine prerequisites
                any if self.attack_steps[attack_name].step_type == "or" else all,
                # prerequisite attack steps
                [
                    self.attack_indices[prerequisite_name]
                    for prerequisite_name in self.attack_steps[attack_name].parents
                ],
            )
            for attack_name in self.attack_names
        ]

        self.ttc_params = [self.attack_steps[attack_name].ttc for attack_name in self.attack_names]

        self.reward_params = [
            self.attack_steps[attack_name].reward for attack_name in self.attack_names
        ]

        self.child_indices = [
            [
                self.attack_indices[child_name]
                for child_name in self.attack_steps[attack_name].children
            ]
            for attack_name in self.attack_names
        ]

        # say hello
        # print(self)

    @property
    def root(self) -> str:
        return self.config.root

    @property
    def num_attacks(self):
        return len(self.attack_names)

    @property
    def num_services(self):
        return len(self.service_names)

    def __str__(self):
        label = os.path.basename(self.config.filename)
        if self.config.graph_size:
            label += f"[{self.config.graph_size}]"
        return (
            f"{self.__class__.__name__}({label}, {self.num_services} services,"
            f" {self.num_attacks} attack steps)"
        )

    def get_eligible_indices(self, attack_index, attack_state, service_state):
        """Get all viable attack steps."""
        eligible_indices = []
        for child_index in self.child_indices[attack_index]:
            required_services, logic, prerequisites = self.attack_prerequisites[child_index]
            if (
                not attack_state[child_index]
                and all(service_state[required_services])
                and logic(attack_state[prerequisites])
            ):
                eligible_indices.append(child_index)
        return eligible_indices

    def save_graphviz(self, filename=None, verbose=False, indexed=False, ttc=None):
        if filename is None:
            filename = f"{'graph' if not self.config.graph_size else self.config.graph_size}.dot"
        if indexed:
            index = {name: i + 1 for i, name in enumerate(self.attack_names)}

        def label(key):
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

    def to_networkx(self, indices=True, system_state=None) -> nx.DiGraph:
        """Convert the AttackGraph to an NetworkX DiGraph."""
        dig = nx.DiGraph()

        if system_state is None:
            system_state = np.ones(len(self.service_names))

        for name, a_s in self.attack_steps.items():
            dict_t = asdict(a_s)

            as_idx = self.attack_indices[name]

            # No need to add child information to node
            del dict_t["children"]
            del dict_t["name"]
            # del dict_t["parents"]

            if not self._attack_step_reachable(as_idx, system_state):
                # If any of the attack steps conditions are not fulfilled,
                # do not add it to the graph
                continue

            # Add the attack step to the graph
            if indices:
                dig.add_node(as_idx, **dict_t)
                dig.add_edges_from(
                    [
                        (as_idx, child_index)
                        for child_index in self.child_indices[as_idx]
                        if self._attack_step_reachable(child_index, system_state)
                    ]
                )
            else:
                dig.add_node(name, **dict_t)
                dig.add_edges_from(
                    [
                        (name, child)
                        for child in a_s.children
                        if self._attack_step_reachable(self.attack_indices[child], system_state)
                    ]
                )

        return dig

    def _attack_step_reachable(self, step: int, state):
        asset_enabled = state[self.service_index_by_attack_index[step]]
        return all(asset_enabled)


def save_all_default_graphviz(graph_config, indexed=False):
    sizes = len(SIZES)
    for key in SIZES:
        config = dataclasses.replace(graph_config, graph_size=key)
        g = AttackGraph(config)
        for i in g.service_names:
            print(i)
        print()
        sizes -= 1
        g.save_graphviz(verbose=(sizes == 0), indexed=indexed)
        print()
        del g
