import json

from rust_sim import RustAttackSimulator


def rust_sim_init(sim_config, attack_graph):
    return RustAttackSimulator(json.dumps(sim_config.to_dict()), attack_graph.config.filename)
