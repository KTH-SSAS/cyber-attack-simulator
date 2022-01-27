"""
Process YAML graph description into an Attack Graph
"""
import dataclasses
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Set, Union

from yaml import safe_load

from .utils import enabled
from attack_simulator.config import GraphConfig
import networkx as nx


@dataclass
class AttackStep:
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    asset: str = ""
    service: str = ""
    flag: str = ""
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

        if len(prune) == 0 and self.config.graph_size in SIZES:
            prune = SIZES[self.config.graph_size]

        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config.filename)

        # load the YAML graph spec
        with open(filename, "r") as yaml_file:
            graph = safe_load(yaml_file.read())

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
            if "asset" in node:
                return
            parts = key.split(".")
            rest = len(parts) - 2
            if rest < 0:
                raise ValueError("Attack step key has fewer than 2 components")
            asset = parts[0]
            name = parts[-1]
            node.update(asset=asset, name=name)
            if asset not in unmalleable_assets:
                services.add(asset)
            if 0 < rest:
                if name == "capture" and parts[-2][:4] == "flag":
                    node.update(flag=parts[-2])
                    rest -= 1
                if rest:
                    service = parts[1]
                    node.update(service=service)
                    services.add(f"{asset}.{service}")

            # handle `ttc` and `reward` "templates"
            for f in ("ttc", "reward"):
                if f in node:
                    # translate {easy,hard}_ttc and/or {early,late,final}_reward
                    # based on upper-case "TEMPLATE" in YAML to actual values
                    # TODO add better template handling
                    node[f] = asdict(self.config)[node[f].lower()]

            if children:
                # Ensure a deterministic ordering of child nodes
                node["children"] = sorted(children)

            node["parents"] = set()

        # second pass: update parents
        def update_parents(key, node, children):
            for child in children:
                graph[child]["parents"].add(key)

        # third and final pass: freeze the relevant sub-graph
        def freeze_subgraph(key, node, children):
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
            service_name = attack_step.asset
            if attack_step.service:
                service_name += "." + attack_step.service
            service_index = self.service_indices.get(service_name, -1)
            self.service_index_by_attack_index.append(service_index)

        self.dependent_services = [
            [dependent.startswith(main) for dependent in self.service_names]
            for main in self.service_names
        ]

        self.attack_prerequisites = [
            (
                # required services
                [attack_name.startswith(service_name) for service_name in self.service_names],
                # logic function to combine prerequisites
                any if self.attack_steps[attack_name].step_type == "or" else all,
                # prerequisite attack steps
                [
                    prerequisite_name in self.attack_steps[attack_name].parents
                    for prerequisite_name in self.attack_names
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
        print(self)

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
        eligible_indices = []
        for child_index in self.child_indices[attack_index]:
            required_services, logic, prerequisites = self.attack_prerequisites[child_index]
            if (
                not attack_state[child_index]
                and all(enabled(required_services, service_state))
                and logic(enabled(prerequisites, attack_state))
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

        with open(filename, "w") as f:
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

    def to_networkx(self, indices=True) -> nx.DiGraph:
        dig = nx.DiGraph()
        if indices:
            dig.add_nodes_from(range(self.num_attacks))
            dig.add_edges_from(
                [
                    (attack_index, child_index)
                    for attack_index in range(self.num_attacks)
                    for child_index in self.child_indices[attack_index]
                ]
            )
        else:
            dig.add_nodes_from(self.attack_names)
            dig.add_edges_from(
                [
                    (name, child)
                    for name in self.attack_names
                    for child in self.attack_steps[name].children
                ]
            )
        return dig


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
