import json

from .rusty_sim import RustAttackSimulator
from .config import SimulatorConfig
from .graph import AttackGraph
from .sim import Simulator

def rust_sim_init(sim_config: SimulatorConfig, attack_graph: AttackGraph) -> Simulator:
    return RustAttackSimulator(json.dumps(sim_config.to_dict()), attack_graph.config.filename)