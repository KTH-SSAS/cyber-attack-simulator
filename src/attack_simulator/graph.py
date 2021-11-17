"""
Process YAML graph description into an Attack Graph
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from yaml import safe_load

from .utils import enabled


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


DEFAULT_CONFIG = {
    "filename": os.path.join(os.path.dirname(os.path.realpath(__file__)), "en2720.yaml"),
    "root": "internet.connect",
    "graph_size": None,
    "prune": set(),
    "unmalleable_assets": set(("internet", "office_network", "hidden_network")),
    "easy_ttc": 10,
    "hard_ttc": 100,
    "low_flag_reward": 10000,
    "medium_flag_reward": 10000,
    "high_flag_reward": 10000,
}

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
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # initialization
        self.attack_steps: Dict[str, AttackStep] = dict()
        services: Set[str] = set()

        # Declare attributes
        self.graph_size: str
        self.filename: str
        self.root: str
        self.prune: Set[str]
        self.unmalleable_assets: Set[str]

        # process incoming configuration
        if config is None:
            config = DEFAULT_CONFIG
        else:
            for key in DEFAULT_CONFIG:
                if key not in config:
                    config[key] = DEFAULT_CONFIG[key]
            for key in config.copy():
                if key not in DEFAULT_CONFIG:
                    del config[key]
        self.__dict__.update(config)

        if len(self.prune) == 0 and self.graph_size in SIZES:
            self.prune = SIZES[self.graph_size]

        # load the YAML graph spec
        with open(self.filename, "r") as yaml_file:
            graph = safe_load(yaml_file.read())

        # traverse through the relevant subgraph for a given pass
        def traverse(perform_pass):
            keys = [self.root]
            while keys:
                next_keys = set()
                for key in keys:
                    node = graph[key]
                    children = set(node["children"]) - self.prune if "children" in node else set()
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
            if asset not in self.unmalleable_assets:
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
                    node[f] = self.__dict__[node[f].lower()]

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
            service_index = self.service_indices[service_name]
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
    def num_attacks(self):
        return len(self.attack_names)

    @property
    def num_services(self):
        return len(self.service_names)

    def __str__(self):
        label = os.path.basename(self.filename)
        if self.graph_size:
            label += f"[{self.graph_size}]"
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
            filename = f"{'graph' if not self.graph_size else self.graph_size}.dot"
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


def save_all_default_graphviz(indexed=False):
    sizes = len(SIZES)
    for key in SIZES:
        g = AttackGraph({"graph_size": key})
        for i in g.service_names:
            print(i)
        print()
        sizes -= 1
        g.save_graphviz(verbose=(sizes == 0), indexed=indexed)
        print()
        del g
