import numpy as np

AGENT_DEFENDER = "defender"
AGENT_ATTACKER = "attacker"

ACTION_TERMINATE = 0
ACTION_WAIT = 1

ACTION_STRINGS = {ACTION_TERMINATE: "terminate", ACTION_WAIT: "wait"}

special_actions = [ACTION_TERMINATE, ACTION_WAIT]

UINT = np.uintp
