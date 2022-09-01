import yaml

filename = "test_yaml.yaml"  # sys.argv[1]
outfile = "out.yaml"  # sys.argv[2]
with open(filename, encoding="utf8") as f:
    data: dict = yaml.safe_load(f)

service_names = set()
assets = set()

for k, v in data.items():

    split_key = k.split(".")
    remainder = len(split_key)

    i = 1
    conditions = []

    while remainder > 1:
        service = ".".join(split_key[:i])
        if "flag" not in service:
            service_names.add(service)
        assets.add(service)
        conditions.append(service)
        remainder -= 1
        i += 1

    name = split_key[-1]

    v["asset"] = conditions[-1]
    # v["conditions"] = conditions
    v["name"] = name


new_data = [{"id": k} for k, v in data.items()]

for entry, old_entry in zip(new_data, data):
    entry.update(data[old_entry])


defense_steps = []
for service in service_names:
    defense_step = {
        "id": service + ".defend",
        "name": "defend",
        "children": [step["id"] for step in new_data if service == step["asset"]],
        "asset": service,
    }
    defense_steps.append(defense_step)

new_data.extend(defense_steps)

dependent_services = {
    main: [dependent for dependent in assets if dependent.startswith(main) and main != dependent]
    for main in assets
}

dependent_services = [{"id": k, "children": v} for k, v in dependent_services.items()]

to_dump = {"attack_graph": new_data, "instance_model": dependent_services}

with open(outfile, "w", encoding="utf8") as f:
    yaml.dump(to_dump, f, sort_keys=False)
