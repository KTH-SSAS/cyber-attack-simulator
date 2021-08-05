import logging
import time
from functools import partial

import numpy as np
import torch

from .agents import Agent
from .env import AttackSimulationEnv
from .utils import plot_episode, plot_training_results


class Runner:
    def __init__(self, agent: Agent, env: AttackSimulationEnv):
        self.agent = agent
        self.env = env
        self.agent_time = 0
        self.env_time = 0

    def run_sim(self, plot_results=False):
        done = False

        enabled_services = np.ones(len(self.env.action_space.spaces), dtype=np.int8)

        rewards = []
        num_services = []
        compromised_flags = []

        env_start = time.time()
        state = self.env.reset()  # Intial state
        self.env_time += time.time() - env_start

        while not done:

            state = np.concatenate([state, enabled_services])

            agent_start = time.time()
            action = self.agent.act(state)
            self.agent_time += time.time() - agent_start

            if action > 0:
                enabled_services[action - 1] = 0

            env_start = time.time()
            new_state, reward, done, info = self.env.step(enabled_services)
            self.env_time += time.time() - env_start

            agent_start = time.time()
            self.agent.update(new_state, reward, done)
            self.agent_time += time.time() - agent_start

            rewards.append(reward)
            # count number of running services
            num_services.append(sum(enabled_services))
            compromised_flags.append(len(info["compromised_flags"]))

            state = new_state

        if plot_results:
            plot_episode(rewards, num_services, compromised_flags)

        return rewards, info["time"], info["compromised_flags"]

    def run_multiple_episodes(self, episodes, evaluation=False, plot=True):

        log = logging.getLogger("trainer")
        returns = np.zeros(episodes)
        losses = np.zeros(episodes)
        lengths = np.zeros(episodes)
        num_compromised_flags = np.zeros(episodes)
        max_patience = 50
        patience = max_patience
        prev_loss = 1e6

        if hasattr(self.agent, "train"):  # not all agent's support setting training mode
            self.agent.train(not evaluation)

        try:
            for i in range(episodes):
                rewards, episode_length, compromised_flags = self.run_sim()
                loss = self.agent.loss if hasattr(self.agent, "loss") else np.random.rand()
                losses[i] = loss
                returns[i] = sum(rewards)
                lengths[i] = episode_length
                num_compromised_flags[i] = len(compromised_flags)
                log.debug(
                    f"Episode: {i+1}/{episodes}, Loss: {loss}, Return: {sum(rewards)},"
                    f" Episode Length: {episode_length}"
                )

                if (prev_loss - loss) < 0.01 and not evaluation:
                    patience -= 1
                else:
                    patience = (patience + 1) if patience < max_patience else max_patience
                if patience == 0:
                    log.debug("Stopping due to insignicant loss changes.")
                    break

                prev_loss = loss

        except KeyboardInterrupt:
            print("Stopping...")

        if evaluation:
            log.debug(f"Average returns: {sum(returns)/len(returns)}")

        if plot:  # TODO move this out of the Runner class
            plot_training_results(
                returns, losses, lengths, num_compromised_flags, evaluation, cutoff=i
            )

        return returns, losses, lengths, num_compromised_flags

    def run(self, func):
        start = time.time()
        num_compromised_flags = 0
        returns, losses, lengths, num_compromised_flags = func()
        duration = time.time() - start
        return duration, returns, losses, lengths, num_compromised_flags

    def train(self, episodes, plot=True):
        func = partial(self.run_multiple_episodes, episodes=episodes, plot=plot)
        return self.run(func)

    def evaluate(self, episodes, plot=True):
        func = partial(self.run_multiple_episodes, episodes=episodes, evaluation=True, plot=plot)
        with torch.no_grad():
            return self.run(func)
