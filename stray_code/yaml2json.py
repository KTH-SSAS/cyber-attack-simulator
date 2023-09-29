import yaml
import json
from pathlib import Path
from maltoolbox.attackgraph.node import AttackGraphNode
from dataclasses import asdict

in_folder = Path("graphs")
out_folder = in_folder
filename = Path("orand.yaml")

basename = filename.stem


def AttackGraphNodeFromYaml(d, defenses, entrypoints):
    d["type"] = d["step_type"] if d["id"] not in defenses else "defense"
    del d["step_type"]
    d["ttc"] = None #{"type": "function", "name": "VeryHardAndUncertain", "arguments": []}
    d["compromised_by"] = []
    d["name"] = "firstSteps" if d["id"] in entrypoints else d["name"]
    return AttackGraphNode(**d)


with open(in_folder / filename) as f:
    data = yaml.safe_load(f)

nodes = [asdict(AttackGraphNodeFromYaml(d, data["defenses"], data["entry_points"])) for d in data["attack_graph"]]

with open(str(out_folder / basename) + ".json", "w") as f:
    json.dump(nodes, f, indent=4)
