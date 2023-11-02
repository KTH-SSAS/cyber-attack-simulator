from attack_simulator.env.attacksimulator import env, raw_env, parallel_env
from attack_simulator.utils.config import EnvConfig, SimulatorConfig
from attack_simulator.env.gym import DefenderEnv, AttackerEnv
import gymnasium as gym

def defender_gym() -> gym.Env:
    gym.register("DefenderEnv-v0", entry_point=DefenderEnv)
    return gym.make("DefenderEnv-v0")

def attacker_gym() -> gym.Env:
    gym.register("AttackerEnv-v0", entry_point=AttackerEnv)
    return gym.make("AttackerEnv-v0")

__all__ = ["env", "raw_env", "parallel_env", "EnvConfig", "SimulatorConfig"]