import json

from rust_sim import RustAttackSimulator
from .config import SimulatorConfig
from .graph import AttackGraph

def rust_sim_init(sim_config: SimulatorConfig, attack_graph: AttackGraph) -> RustAttackSimulator:
    return RustAttackSimulator(json.dumps(sim_config.to_dict()), attack_graph.config.filename)