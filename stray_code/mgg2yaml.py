from dataclasses import dataclass
import json
from pathlib import Path

from itertools import accumulate
from typing import List
from scipy.stats import bernoulli, expon
import operator
import yaml
import re

expected_fields = [
    "id",
    "type",
    "objclass",
    "objid",
    "atkname",
    "ttc",
    "links",
    "is_reachable",
    "defense_status",
    "graph_type",
    "is_traversable",
    "required_steps",
    "extra",
]


@dataclass
class TTC:
    type: str
    name: str
    arguments: List[float]


@dataclass
class AttackStep:
    id: int
    type: str
    objclass: str
    objid: int
    atkname: str
    ttc: List[TTC]
    links: List[str]
    is_reachable: bool
    defense_status: bool
    graph_type: str
    is_traversable: bool
    required_steps: List[str]
    extra: dict
    mitre_info: str


ttc_name_function_mappings = {
    "EasyAndCertain": [("Exponential", [1.0])],
    "EasyAndUncertain": [("Bernoulli", [0.5])],
    "HardAndCertain": [("Exponential", [0.1])],
    "HardAndUncertain": [("Bernoulli", [0.5]), ("Exponential", [0.1])],
    "VeryHardAndCertain": [("Exponential", [0.01])],
    "VeryHardAndUncertain": [("Bernoulli", [0.5]), ("Exponential", [0.01])],
    "Disabled": [("Bernoulli", [1.0])],
}


def replace_colons(name: str):
    return name.replace(":", "-")


cdf_mappings = {
    "Bernoulli": lambda x, _: bernoulli.pmf(1, p=x[0]),
    "Exponential": lambda x, t: expon.cdf(t, scale=1 / x[0]),
}


def convert_ttc(ttc_field, tau=100.0):

    ttc_field = TTC(**ttc_field)

    ttc = []

    assert ttc_field.type == "function"

    name = ttc_field.name
    arguments = ttc_field.arguments

    if arguments:
        distributions = [
            {
                "name": name,
                "arguments": arguments,
            }
        ]
    else:
        assert name in ttc_name_function_mappings
        distributions = ttc_name_function_mappings[name]
        distributions = [{"name": dist, "arguments": args} for dist, args in distributions]

    ps = [cdf_mappings[dist["name"]](dist["arguments"], tau) for dist in distributions]
    ttc_prob = list(
        accumulate(
            ps,
            operator.mul,
            initial=1.0,
        )
    )[-1]

    ttc = distributions
    return ttc


def convert_step(step: AttackStep):
    return {
        "id": step.id,
        "ttc": convert_ttc(step.ttc) if step.ttc else [{"name": "Bernoulli", "arguments": [1.0]}],
        "children": [step for step in step.links],
        "step_type": step.type,
        "asset": f"{step.objclass}:{step.objid}",
        "name": step.atkname,
    }


def convert_defenses(steps):
    defenses = []
    for step in steps:
        if step["step_type"] == "defense":
            defenses.append(step["id"])
            step["step_type"] = "or"

    return steps, defenses


def main() -> None:

    directory = Path("mal")
    filename = "test_atkgraph_10_51_21.json"

    basename = filename.split(".")[0]

    with open(directory / filename, "r", encoding="utf-8") as f:
        graph = json.load(f)

    steps = [AttackStep(**step) for step in graph]

    # filter steps that are not reachable
    # steps = [step for step in steps if step.is_reachable or step.is_traversable]

    # remove children that no longer exist
    for step in steps:
        step.links = [link for link in step.links if link in [step.id for step in steps]]

    new_steps = [convert_step(step) for step in steps]

    flags = [step.id for step in steps if step.atkname == "read"]

    defenses = [step.id for step in steps if step.type == "defense"]

    entry_points = [step.id for step in steps if step.atkname == "physicalAccess"]

    # new_steps, defenses = convert_defenses(new_steps)

    full_file = {
        "attack_graph": new_steps,
        "flags": flags,
        "entry_points": entry_points,
        "defenses": defenses,
        "instance_model": [],
    }

    with open(f"{basename}.yaml", "w") as f:
        text = yaml.dump(full_file)
        text = re.sub(r"(?<=\w):(?=\w)", "-", text)
        f.write(text)


if __name__ == "__main__":
    main()
