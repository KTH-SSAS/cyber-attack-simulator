import pygame
import torch
from attack_simulator.constants import AGENT_ATTACKER, AGENT_DEFENDER
from attack_simulator.agents.attackers.searchers import BreadthFirstAttacker
#from attack_simulator.models.sr_drl import Net
import attack_simulator
import json
from json import JSONEncoder
import numpy as np


class KeyboardAgent:
    def __init__(self, vocab):
        self.vocab = vocab

    def compute_action_from_dict(self, obs):
        assets = obs["asset"]
        asset_ids = obs["asset_id"]
        step_names = obs["step_name"]
        available_actions = np.flatnonzero(obs["node_surface"])
        assets = [self.vocab[i] for i in assets[available_actions]]
        asset_ids = asset_ids[available_actions]
        step_names = [self.vocab[i] for i in step_names[available_actions]]
        action_strings = [f"{a}:{i}:{s}" for a, i, s in zip(assets, asset_ids, step_names)]
        action_strings = [f"{i}. {a}" for i, a in enumerate(action_strings)]
        print("Available actions:")
        print("\n".join(action_strings))

        node = -1
        while node >= len(available_actions) or node < 0:
            print("Enter action:")
            node = int(input())
            if node >= len(available_actions) or node < 0:
                print("Invalid action.")

        print(f"Selected action: {action_strings[node]}")

        print(available_actions[node])

        return (1, available_actions[node])


env_config = attack_simulator.EnvConfig.from_yaml("config/maze_env_config.yaml")
env = attack_simulator.parallel_env(env_config, render_mode="human")

layers = 4
hidden_size = 4
# defender = FixedActionGNNRLAgent(1, layers, hidden_size, num_actions)
# defender = GNNRLAgent(1, layers, hidden_size)
#q_range = (-100.0, 200.0 * env.n_nodes)
defender = KeyboardAgent(env.reverse_vocab)
#target_net = Net(q_range)
#attacker = BreadthFirstAttacker({})
attacker = KeyboardAgent(env.reverse_vocab)


obs, info = env.reset()
done = False

obs_log = open("obs_log.jsonl", "w")

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return JSONEncoder.default(self, obj)


# plt.show(block=False)

with torch.no_grad():
    while not done:
        # defender_action_dist, value = defender.compute_action_from_dict(obs["defender"])
        # print(defender_action_diST.numpy())
        # defender_action_dist = F.softmax(defender_action_dist, dim=0)

        #        render = env.render()
        #        graph = pgv.AGraph(render)
        #        graph.layout(prog="dot")
        #        bytes = graph.draw(format="png")
        #        img = Image.open(io.BytesIO(bytes))
        #        plt.imshow(img)
        #        plt.draw()
        #        plt.pause(0.01)

        env.render()
        node_feats, edge_attr, edge_index = (
            obs["defender"]["ids_observation"].reshape(-1, 1),
            None,
            obs["defender"]["edges"],
        )

        s_batch = [(node_feats, edge_attr, edge_index)]
        defender_action_mask = obs["defender"]["action_mask"]
        defender_node_mask = obs["defender"]["node_surface"]
        defender_action = defender(s_batch, defender_action_mask, defender_node_mask)
        node_selection, value, action_probs, node_probs = defender_action
        action = 0 if len(node_selection[0]) == 0 else 1
        node_selection = 0 if len(node_selection[0]) == 0 else node_selection[0][0]
        attacker_action = attacker.compute_action_from_dict(obs["attacker"])
        action_dict = {AGENT_ATTACKER: attacker_action, AGENT_DEFENDER: (action, node_selection)}
        if action == 1:
            assert defender_node_mask[node_selection] == 1
        print("Press Enter to continue.")
        #input()
        obs, rewards, terminated, truncated, infos = env.step(action_dict)

        # dists = {AGENT_DEFENDER: defender_action_dist.numpy().tolist()}

        done = terminated[AGENT_ATTACKER]

        log = {
            "obs": obs,
            "actions": {k: (a, s) for k, (a, s) in action_dict.items()},
            "rewards": {k: int(v) for k, v in rewards.items()},
            "info": infos,
            "terminated": terminated,
            "truncated": truncated,
            # "action_distributions": dists,
        }

        obs_log.write(f"{json.dumps(log, cls=NumpyArrayEncoder)}\n")

env.render()
print("Done. Press Enter to exit.")
input()
env.close()

obs_log.close()
