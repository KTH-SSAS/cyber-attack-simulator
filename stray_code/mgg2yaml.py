import json
import yaml

def replace_colons(name: str):
	return name.replace(':', '-')

def convert_ttc(ttc_field):
    if ttc_field:
        return ttc_field["name"]
    else:
        return "default"


def convert_step(step):
    return {
        "id": step["id"].replace(':', '-'),
        "ttc": convert_ttc(step["ttc"]),
        "children": [step.replace(':', '-') for step in step["links"]],
        "step_type": step["type"] if step["type"] != "notExist" else "or",
    }


def convert_defenses(steps):
    defenses = []
    for step in steps:
        if step["step_type"] == "defense":
            defenses.append(step["id"])
            step["step_type"] = "or"

    return steps, defenses


with open("graph_with_defense_status_and_attacker.json", "r", encoding="utf-8") as f:
    graph = json.load(f)


new_steps = [convert_step(step) for step in graph]

new_steps, defenses = convert_defenses(new_steps)


full_file = {"attack_graph": new_steps, "flags": ["Credentials--3746579751403632785-credentialTheft"], "entry_points": ["Attacker--8227084409955727818-firstSteps"], "defenses": defenses, 'instance_model': []}

with open("mgg.yaml", "w") as f:
    yaml.dump(full_file, f)
