import json

from ..mal.graph import AttackGraph
from ..mal.sim import Simulator
from ..rusty_sim import RustAttackSimulator
from .config import SimulatorConfig


def rust_sim_init(sim_config: SimulatorConfig, attack_graph: AttackGraph) -> Simulator:
    return RustAttackSimulator(json.dumps(sim_config.to_dict()), attack_graph.config.filename)
