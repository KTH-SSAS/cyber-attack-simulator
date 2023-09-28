from attack_simulator.env.attacksimulator import env, raw_env, parallel_env
from attack_simulator.utils.config import EnvConfig
from attack_simulator.env.env import register_rllib_env
import attack_simulator.rllib.defender_model as defender_model
import attack_simulator.rllib.gnn_model as gnn_defender

def register_rllib():
	register_rllib_env()
	defender_model.register_rllib_model()
	gnn_defender.register_rllib_model()