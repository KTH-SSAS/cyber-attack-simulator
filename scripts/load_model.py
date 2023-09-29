import torch
from attack_simulator import AGENT_ATTACKER, AGENT_DEFENDER
from attack_simulator.agents.attackers.searchers import BreadthFirstAttacker, DepthFirstAttacker
from attack_simulator.env.env import AttackSimulationEnv
from attack_simulator.models.gnn import GNNRLAgent
from pathlib import Path
import torch.nn.functional as F

from attack_simulator.utils.config import EnvConfig

env_config = EnvConfig.from_yaml("config/maze_env_config.yaml")

env = AttackSimulationEnv(env_config)

layers = 4
hidden_size = 4
defender = GNNRLAgent(1, layers, hidden_size)
weight_path = Path("~/sentience/attack-simulator/weights/model.pt").expanduser()
if weight_path.exists():
    with open("/home/jakob/sentience/attack-simulator/weights/model.pt", "rb") as f:
        state_dict = torch.load(f)
        defender.load_state_dict(state_dict)


attacker = DepthFirstAttacker({})

obs, info = env.reset()

done = False
with torch.no_grad():
    while not done:

        defender_action_dist, value = defender.compute_action_from_dict(obs["defender"])
        defender_action_dist = F.softmax(defender_action_dist, dim=0)
        defender_action = defender_action_dist.argmax().item()
        attacker_action = attacker.compute_action_from_dict(obs["attacker"])
        obs, rewards, terminated, truncated, infos = env.step(
            {AGENT_ATTACKER: attacker_action, AGENT_DEFENDER: defender_action}
        )

        done = terminated[AGENT_ATTACKER]

        print(
            f"a_act: {attacker_action - obs['attacker']['action_offset']} d_act: {defender_action} reward: {rewards} done: {done}"
        )
