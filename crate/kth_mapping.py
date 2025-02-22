from typing import Dict, Union

hosts = [
    "192.168.0.2",
    "10.0.0.2",
    "168.0.6.1",
    "168.0.2.2",
]

actions = [
    "block_ip_address",
    "turn off host",
]

ip_to_state_index = {k: v for v, k in enumerate(hosts)}

num_hosts_that_can_be_blocked = 3

action_to_host = {i: hosts[i] for i in range(0, num_hosts_that_can_be_blocked)}

num_hosts = len(action_to_host)
num_actions = len(action_to_host)

def convert_to_wazuh_action(action: Union[int, tuple]) -> Dict:
    """
    Convert an action index to a Wazuh command
    Args:
                action (int): The action index
        Returns:
                Dict: A dictionary containing the arguments to send to Wazuh
    """

    if isinstance(action, tuple):
        command = action[action[0]]
        host = hosts[action[1]]
    else:
        host = action_to_host[action]
        command = "block_ip_address"

    return {"command": command, "alert": {"data": {"srcip": host}}}


def convert_wazuh_response_to_obs(response: Dict) -> Dict:
    """
    Convert a Wazuh response to an observation
    Args:
                response (Dict): The response from Wazuh
        Returns:
                Dict: A dictionary containing the current alert state
    """

    state_vec = [0] * num_hosts
    for alert in response["alerts"]:
        alert_ip = alert["observer.ip"]
        index = ip_to_state_index[alert_ip]
        state_vec[index] = 1 # This may be replaced by some other mapping, for instance to MAL. This just puts it in a vector.

    return {"alert_state": state_vec}, response["info"]
