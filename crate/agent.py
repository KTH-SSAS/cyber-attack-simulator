import asyncio
import json
from typing import Dict, Tuple, Any
import torch
from torch import nn
from ray import rllib

from crate.foi_wazuh_lib import query_wazuh, send_action_to_wazuh
from crate.mapping import convert_to_wazuh_action, convert_wazuh_response_to_obs
import crate.mapping
wazuh_hostname = "localhost"
poll_time = 10

num_hosts = crate.mapping.num_hosts
num_actions = crate.mapping.num_actions

async def step(action: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Execute an action and return the new state
        Args:
                action (int): The action index
        Returns:
                Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the new state and the info dictionary
    """
    command_dict = convert_to_wazuh_action(action)
    payload = json.dumps(command_dict)
    await send_action_to_wazuh(payload)
    print("Waiting for", poll_time, "seconds...")
    await asyncio.sleep(poll_time)
    wazuh_data = await query_wazuh(wazuh_hostname)
    obs, info = convert_wazuh_response_to_obs(wazuh_data)
    return obs, info


async def main():
    with torch.no_grad():
        model = torch.load("model.pt")
        wazuh_data = await query_wazuh(wazuh_hostname)
        obs, info = convert_wazuh_response_to_obs(wazuh_data)
        while True:
            action_dist = model(torch.FloatTensor(obs["alert_state"]))
            action = torch.argmax(action_dist).item()
            obs, info = await step(action)


# Save an untrained model for testing
trainer = rllib.agents.ppo.PPOTrainer(
    env = "AttackEnv",
)

with open("model.pt", "wb") as f:
    torch.save(model, f)

asyncio.run(main())
