"""
Process YAML graph description into an Attack Graph
"""
import os
from collections import namedtuple
from typing import Any, Dict, List, NamedTuple, Optional, Set

from yaml import safe_load

AttackStep: NamedTuple = namedtuple(
    "AttackStep",
    field_names=(
        "asset",
        "service",
        "flag",
        "name",
        "step_type",
        "ttc",
        "reward",
        "children",
        "parents",
    ),
    defaults=(None, None, None, None, "or", 1, 0, (), ()),
)

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

SIZES = {
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
        self.attack_steps: Dict[str, NamedTuple] = dict()
        services: Set[str] = set()

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

        if not self.prune and self.graph_size in SIZES:
            self.prune = SIZES[self.graph_size]

        # load the YAML graph spec
        with open(self.filename, "r") as yaml_file:
            graph = safe_load(yaml_file.read())

        # traverse through the relevant subgraph for a given pass
        def traverse(perform_pass):
            keys = set((self.root,))
            while keys:
                next_keys = set()
                for key in keys:
                    node = graph[key]
                    children = set(node["children"]) - self.prune if "children" in node else set()
                    perform_pass(key, node, children)
                    next_keys |= set(children)
                keys = sorted(next_keys)

        # first pass: determine services and most fields
        def determine_fields(key, node, children):
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
            for field in ("ttc", "reward"):
                if field in node:
                    # translate {easy,hard}_ttc and/or {early,late,final}_reward
                    # based on upper-case "TEMPLATE" in YAML to actual values
                    node[field] = self.__dict__[node[field].lower()]

            if children:
                node["children"] = tuple(sorted(children))

            node["parents"] = set()

        # second pass: update parents
        def update_parents(key, node, children):
            for child in children:
                graph[child]["parents"].add(key)

        # third and final pass: freeze the relevant sub-graph
        def freeze_subgraph(key, node, children):
            node["parents"] = tuple(sorted(node["parents"]))
            self.attack_steps[key] = AttackStep(**node)

        for perform_pass in (determine_fields, update_parents, freeze_subgraph):
            traverse(perform_pass)

        # set final attributes
        self.service_names: List[str] = sorted(services)
        self.num_services = len(self.service_names)

        self.attack_names: List[str] = sorted(self.attack_steps)
        self.num_attacks = len(self.attack_names)

        # say hello
        print(self)

    def __str__(self):
        label = os.path.basename(self.filename)
        if self.graph_size:
            label += f"[{self.graph_size}]"
        return (
            f"{self.__class__.__name__}({label}, {self.num_services} services,"
            f" {self.num_attacks} attack steps)"
        )

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
