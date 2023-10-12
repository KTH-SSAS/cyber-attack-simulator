import torch
from attack_simulator.constants import AGENT_ATTACKER, AGENT_DEFENDER
from attack_simulator.agents.attackers.searchers import BreadthFirstAttacker
import attack_simulator
import json
from json import JSONEncoder
import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.int64):
            return int(o)
        return JSONEncoder.default(self, o)


class KeyboardAgent:
    def __init__(self, vocab):
        self.vocab = vocab

    def compute_action_from_dict(self, obs):
        def valid_action(user_input):
            if user_input == "":
                return (0, None)

            try:
                node = int(user_input)
            except ValueError:
                return False

            try:
                a = associated_action[action_strings[node]]
            except IndexError:
                return False

            if a == 0:
                return True  # wait is always valid
            return node < len(available_objects) and node >= 0

        def get_action_object(user_input):
            node = int(user_input) if user_input != "" else 0
            action = associated_action[action_strings[node]] if user_input != "" else 0
            return node, action

        assets = obs["asset"]
        asset_ids = obs["asset_id"]
        step_names = obs["step_name"]
        available_objects = np.flatnonzero(obs["node_surface"])
        assets = [self.vocab[i] for i in assets[available_objects]]
        asset_ids = asset_ids[available_objects]
        step_names = [self.vocab[i] for i in step_names[available_objects]]

        action_strings = [f"{a}:{i}:{s}" for a, i, s in zip(assets, asset_ids, step_names)]
        associated_action = {i: 1 for i in action_strings}
        action_strings += ["wait"]
        associated_action["wait"] = 0

        user_input = "xxx"
        while not valid_action(user_input):
            print("Available actions:")
            print("\n".join([f"{i}. {a}" for i, a in enumerate(action_strings)]))
            print("Enter action or leave empty to wait:")
            user_input = input("> ")

            if not valid_action(user_input):
                print("Invalid action.")

        node, a = get_action_object(user_input)
        print(f"Selected action: {action_strings[node]}")

        return (a, available_objects[node] if a != 0 else None)


sim_config = attack_simulator.SimulatorConfig(
    false_negative_rate=0.0, false_positive_rate=0.0, seed=0, randomize_ttc=False, log=False, show_false=False
)
env_config = attack_simulator.EnvConfig(
    sim_config=sim_config,
    graph_filename="graphs/two_ways.json",
    seed=0,
)
env = attack_simulator.parallel_env(env_config, render_mode="human")

control_attacker = False

defender = KeyboardAgent(env.reverse_vocab)
attacker = KeyboardAgent(env.reverse_vocab) if control_attacker else BreadthFirstAttacker({})

obs, info = env.reset()
done = False

with open("sim_obs_log.jsonl", "w", encoding="utf8") as f:
    f.write("Game Start!\n")

total_reward_defender = 0
total_reward_attacker = 0

with torch.no_grad():
    while not done:
        env.render()
        defender_action = defender.compute_action_from_dict(obs["defender"])
        attacker_action = attacker.compute_action_from_dict(obs["attacker"])
        action_dict = {AGENT_ATTACKER: attacker_action, AGENT_DEFENDER: defender_action}
        obs, rewards, terminated, truncated, infos = env.step(action_dict)
        print("Attacker Reward: ", rewards[AGENT_ATTACKER])
        print("Defender Reward: ", rewards[AGENT_DEFENDER])
        total_reward_defender += rewards[AGENT_DEFENDER]
        total_reward_attacker += rewards[AGENT_ATTACKER]

        done = terminated["__all__"]

        log = {
            "obs": obs,
            "actions": {k: (a, s) for k, (a, s) in action_dict.items()},
            "rewards": {k: int(v) for k, v in rewards.items()},
            "info": infos,
            "terminated": terminated,
            "truncated": truncated,
        }

        with open("sim_log.jsonl", "a", encoding="utf8") as f:
            f.write(f"{json.dumps(log, cls=NumpyArrayEncoder)}\n")

        print("---\n")

env.render()
print("Game Over.")
print("Total Defender Reward: ", total_reward_defender)
print("Total Attacker Reward: ", total_reward_attacker)
print("Press Enter to exit.")
input()
env.close()
