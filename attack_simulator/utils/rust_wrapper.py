import json

from ..rusty_sim.cpython-310-x86_64-linux-gnu import RustAttackSimulator
from .config import SimulatorConfig
from .sim.graph import AttackGraph
from .sim import Simulator

def rust_sim_init(sim_config: SimulatorConfig, attack_graph: AttackGraph) -> Simulator:
    return RustAttackSimulator(json.dumps(sim_config.to_dict()), attack_graph.config.filename)