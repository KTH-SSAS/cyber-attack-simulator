import functools

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from attack_simulator.constants import AGENT_ATTACKER, AGENT_DEFENDER

from attack_simulator.utils.config import EnvConfig
from pettingzoo.test.api_test import api_test
from attack_simulator.env.env import AttackSimulationEnv
from pettingzoo.test import parallel_api_test


def env(config, render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(config, render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(config, render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(config, render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict, render_mode: str = None):
        self.possible_agents = [AGENT_ATTACKER, AGENT_DEFENDER]

        # optional: a mapping between agent name and ID
        # self.agent_name_mapping = dict(
        #     zip(self.possible_agents, list(range(len(self.possible_agents))))
        # )
        env_config = EnvConfig.from_dict(
            {
                "sim_false_positive_rate": config.get("sim_false_positive_rate", 0.0),
                "sim_false_negative_rate": config.get("sim_false_negative_rate", 0.0),
                "graph_name": config.get("graph_name", "four_ways_mod"),
            }
        )
        self.env = AttackSimulationEnv(env_config, render_mode)
        self.render_mode = render_mode
        self.vocab = self.env.vocab
        self.reverse_vocab = self.env.reverse_vocab

        self.observation_spaces = self.env.observation_space
        self.action_spaces = self.env.action_space

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self.env.observation_space[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self.env.action_space[agent]

    @property
    def agents(self) -> list:
        return self.env._agent_ids

    def render(self) -> None:
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        return self.env.render()

    def close(self) -> None:
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed: int = None, options: dict = None) -> dict:
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        return self.env.reset()

    def step(self, actions: dict) -> tuple:
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        return self.env.step(actions)


if __name__ == "__main__":
    p_env = parallel_env({})
    parallel_api_test(p_env, num_cycles=50)
    aec_env = env({})
    api_test(aec_env, num_cycles=50)
