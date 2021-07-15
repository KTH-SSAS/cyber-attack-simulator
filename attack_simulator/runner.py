from attack_simulator.config import AgentConfig
from attack_simulator.attack_simulation_env import AttackSimulationEnv
from attack_simulator.utils import create_agent, plot_episode, plot_training_results
import matplotlib.pyplot as plt
import logging
import numpy as np
import torch
import time
from functools import partial


class Runner:

    def __init__(self, agent, env: AttackSimulationEnv, include_services=False):
        self.include_services = include_services # TODO move this to the environment
        self.env = env
        self.agent = agent
        self.agent_time = 0
        self.environment_time = 0

    def run_sim(self, plot_results=False):
        services = {}  # Serves as a key for which services belong to which index
        done = False
        for service, i in enumerate(self.env.attack_graph.enabled_services):
            services[service] = i

        enabled_services = np.ones(len(services), dtype=np.int8)

        rewards = []
        num_services = []
        compromised_flags = []
        state = self.env._next_observation()  # Intial state
        while not done:

            if self.include_services:
                state = np.concatenate([state, enabled_services])

            agent_start = time.time()
            action = self.agent.act(state)
            self.agent_time += time.time() - agent_start

            if self.agent.can_skip:
                if action > 0:
                    # Shift action by 1 since action==0 is treated as skip
                    enabled_services[action - 1] = 0
                else:
                    pass  # Skip action and don't disable a service
            else:
                enabled_services[action] = 0

            env_start = time.time()
            new_state, reward, done, info = self.env.step(enabled_services)
            self.environment_time += time.time() - env_start

            rewards.append(reward)
            # count number of running services
            num_services.append(sum(enabled_services))
            compromised_flags.append(len(info['compromised_flags']))
            state = new_state

        if plot_results:
            plot_episode(rewards, num_services, compromised_flags)

        return rewards, info['time'], info['compromised_flags']

    def run_multiple_episodes(self, episodes, evaluation=False, plot=True):

        log = logging.getLogger("trainer")
        returns = np.zeros(episodes)
        losses = np.zeros(episodes)
        lengths = np.zeros(episodes)
        num_compromised_flags = np.zeros(episodes)
        max_patience = 50
        patience = max_patience
        prev_loss = 1E6

        if evaluation:
            self.agent.eval()
        else:
            self.agent.train()

        try:
            for i in range(episodes):
                rewards, episode_length, compromised_flags = self.run_sim()
                if evaluation:
                    loss = self.agent.calculate_loss(rewards).item()
                else:
                    loss = self.agent.update(rewards)
                losses[i] = loss
                returns[i] = sum(rewards)
                lengths[i] = episode_length
                num_compromised_flags[i] = len(compromised_flags)
                self.env.reset()
                log.debug(
                    f"Episode: {i+1}/{episodes}, Loss: {loss}, Return: {sum(rewards)}, Episode Length: {episode_length}")

                if (prev_loss - loss) < 0.01 and not evaluation:
                    patience -= 1
                else:
                    patience = (
                        patience+1) if patience < max_patience else max_patience
                if patience == 0:
                    log.debug("Stopping due to insignicant loss changes.")
                    break

                prev_loss = loss

        except KeyboardInterrupt:
            print("Stopping...")

        if evaluation:
            log.debug(f"Average returns: {sum(returns)/len(returns)}")

        if plot: # TODO move this out of the Runner class
            plot_training_results(returns, losses, lengths, num_compromised_flags, evaluation, cutoff=i)

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


    def generate_graphviz_file(self):
        self.env.attack_graph.generate_graphviz_file()
