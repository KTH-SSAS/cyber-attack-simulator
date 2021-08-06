import logging
import time
from collections import namedtuple
from typing import NamedTuple

import numpy as np
import torch

from .agents import Agent
from .env import AttackSimulationEnv
from .utils import plot_episode, plot_training_results

logger = logging.getLogger("trainer")


RunnerResults: NamedTuple = namedtuple(
    "RunnerResults",
    field_names=(
        "duration",
        "env_time",
        "agent_time",
        "returns",
        "losses",
        "lengths",
        "num_compromised_flags",
    ),
    defaults=(0.0, 0.0, 0.0, None, None, None, None),
)


class Runner:
    def __init__(self, agent: Agent, env: AttackSimulationEnv):
        self.agent = agent
        self.env = env
        self.agent_time = 0
        self.env_time = 0

    def _run_sim(self, plot_results=False):
        if plot_results:
            num_services = self.env.action_space.n - 1
            num_running_services = []
            num_compromised_flags = []

        rewards = []
        done = False

        env_start = time.time()
        state = self.env.reset()  # Intial state
        self.env_time += time.time() - env_start

        while not done:
            agent_start = time.time()
            action = self.agent.act(state)
            self.agent_time += time.time() - agent_start

            env_start = time.time()
            state, reward, done, info = self.env.step(action)
            self.env_time += time.time() - env_start

            agent_start = time.time()
            self.agent.update(state, reward, done)
            self.agent_time += time.time() - agent_start

            rewards.append(reward)

            if plot_results:
                # count number of running services
                num_running_services.append(sum(state[:num_services]))
                num_compromised_flags.append(len(info["compromised_flags"]))

        if plot_results:
            plot_episode(rewards, num_running_services, num_compromised_flags)

        return sum(rewards), info["time"], len(info["compromised_flags"])

    def _run_episodes(self, episodes, random_seed=None, evaluation=False, plot=True):
        if not episodes:
            return RunnerResults()

        start = time.time()

        returns = np.zeros(episodes)
        losses = np.zeros(episodes)
        lengths = np.zeros(episodes)
        num_compromised_flags = np.zeros(episodes)
        max_patience = 50
        patience = max_patience
        prev_loss = 1e6

        if hasattr(self.agent, "train"):  # not all agent's support setting training mode
            self.agent.train(not evaluation)

        random_seed = self.env.seed(random_seed)
        logger.info(f"Starting simulations with seed #{random_seed}")

        try:  # allow graceful manual termination
            for i in range(episodes):
                returns[i], lengths[i], num_compromised_flags[i] = self._run_sim()

                if hasattr(self.agent, "loss"):
                    loss = losses[i] = self.agent.loss

                    if (prev_loss - loss) < 0.01 and not evaluation:
                        patience -= 1
                    elif patience < max_patience:
                        patience += 1
                    prev_loss = loss
                else:  # generate random loss (FIXME: for plotting?)
                    losses[i] = np.random.rand()

                logger.debug(
                    f"Episode: {i+1}/{episodes}, Loss: {losses[i]}, Return: {returns[i]},"
                    f" Episode Length: {lengths[i]}"
                )

                if patience == 0:
                    logger.debug("Stopping due to insignificant change in loss.")
                    break

        except KeyboardInterrupt:
            print("Stopping...")

        if evaluation:
            logger.debug(f"Average returns: {returns.mean()}")

        if plot:  # TODO move this out of the Runner class
            plot_training_results(
                returns, losses, lengths, num_compromised_flags, evaluation, cutoff=i
            )

        duration = time.time() - start

        return RunnerResults(
            duration=duration,
            agent_time=self.agent_time,
            env_time=self.env_time,
            returns=returns,
            losses=losses,
            lengths=lengths,
            num_compromised_flags=num_compromised_flags,
        )

    def train(self, episodes, random_seed=None, plot=True):
        return self._run_episodes(episodes=episodes, random_seed=random_seed, plot=plot)

    def evaluate(self, episodes, random_seed=None, plot=True):
        with torch.no_grad():
            results = self._run_episodes(
                episodes=episodes, random_seed=random_seed, evaluation=True, plot=plot
            )
        return results
