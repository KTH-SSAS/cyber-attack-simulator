import yaml
import json
from pathlib import Path
from maltoolbox.attackgraph.node import AttackGraphNode
from dataclasses import asdict
import sys

in_folder = Path("graphs")
out_folder = in_folder
filename = Path("yaml_graphs/branches.yaml")

basename = filename.stem


data = yaml.safe_load(sys.stdin.read())


def AttackGraphNodeFromYaml(d, defenses, entrypoints):
    d["type"] = d["step_type"] if d["id"] not in defenses else "defense"
    del d["step_type"]
    d["ttc"] = None #{"type": "function", "name": "VeryHardAndUncertain", "arguments": []}
    d["asset"] = d["id"].split("-")[0] + ":" + d["id"].split("-")[1]
    d["compromised_by"] = []
    d["name"] = "firstSteps" if d["id"] in entrypoints else d["name"]
    return AttackGraphNode(**d)




nodes = [asdict(AttackGraphNodeFromYaml(d, data["defenses"], data["entry_points"])) for d in data["attack_graph"]]

print(json.dumps(nodes, indent=4))
