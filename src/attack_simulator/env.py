import logging
import os

import gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import HTMLWriter

from .agents import ATTACKERS
from .graph import AttackGraph, AttackStep
from .nx_utils import nx_dag_layout, nx_digraph
from .rng import get_rng
from .svg_tooltips import add_tooltips, postprocess_frame, postprocess_html

logger = logging.getLogger("simulator")


def enabled(value, state):
    return (value & state) == value


class AttackSimulationEnv(gym.Env):
    NO_ACTION = "no action"

    def __init__(self, env_config):
        super(AttackSimulationEnv, self).__init__()

        # process configuration, leave the graph last, as it may destroy env_config
        self.attacker_class = ATTACKERS[env_config.get("attacker", "random")]
        self.true_positive = env_config.get("true_positive", 1.0)
        self.false_positive = env_config.get("false_positive", 0.0)
        self.save_graphs = env_config.get("save_graphs")
        self.save_text = env_config.get("save_text")
        self.g = env_config.get("attack_graph")
        if self.g is None:
            self.g = AttackGraph(env_config)
        # a placeholder for a networkx-compatible version for rendering
        self.dag = None

        # prepare just enough to get dimensions sorted, do the rest on first `reset`
        self.entry_attack_index = self.g.attack_names.index(self.g.root)

        # An observation informs the defender of
        # a) which services are turned on; and,
        # b) which attack steps have been successfully taken
        self.dim_observations = self.g.num_services + self.g.num_attacks
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(2),) * self.dim_observations)

        # The defender action space allows to disable any one service or leave all unchanged
        self.num_actions = self.g.num_services + 1
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.writers = {}
        self.episode_count = 0
        self._seed = None
        self.reward = None
        self.action = 0
        self.attack_index = None
        self.compromised_flags = []
        self.compromised_steps = []

    def _extract_attack_step_field(self, field_name):
        field_index = AttackStep._fields.index(field_name)
        return np.array(
            [self.g.attack_steps[attack_name][field_index] for attack_name in self.g.attack_names]
        )

    def _setup(self):
        # prime RNG if not yet set by `seed`
        if self._seed is None:
            self.seed()

        self.dependent_services = [
            [dependent.startswith(main) for dependent in self.g.service_names]
            for main in self.g.service_names
        ]

        self.attack_prerequisites = [
            (
                # required services
                [attack_name.startswith(service_name) for service_name in self.g.service_names],
                # logic function to combine prerequisites
                any if self.g.attack_steps[attack_name].step_type == "or" else all,
                # prerequisite attack steps
                [
                    prerequisite_name in self.g.attack_steps[attack_name].parents
                    for prerequisite_name in self.g.attack_names
                ],
            )
            for attack_name in self.g.attack_names
        ]

        self.ttc_params = self._extract_attack_step_field("ttc")
        self.reward_params = self._extract_attack_step_field("reward")

    def reset(self):
        if self.episode_count == 0:
            self._setup()

        self.episode_count += 1
        # TODO connect this with ray run id/wandb run id instead of random seed.
        self.episode_id = f"{self._seed}_{self.episode_count}"
        logger.debug(f"Starting new simulation. (#{self.episode_id})")

        self.ttc_remaining = np.array(
            [max(1, int(v)) for v in self.rng.exponential(self.ttc_params)]
        )
        self.rewards = np.array([int(v) for v in self.rng.exponential(self.reward_params)])

        self.simulation_time = 0
        self.service_state = np.full(self.g.num_services, 1)
        self.attack_state = np.full(self.g.num_attacks, 0)
        self.attack_surface = np.full(self.g.num_attacks, 0)
        self.attack_surface[self.entry_attack_index] = 1

        self.attacker = self.attacker_class(
            dict(
                attack_graph=self.g,
                ttc=self.ttc_remaining,
                rewards=self.rewards,
                random_seed=self._seed + self.episode_count,
            )
        )
        return self.observe()

    def observe(self):
        # Observation of attack steps is subject to the true/false positive rates
        # of an assumed underlying intrusion detection system
        # Depending on the true and false positive rates for each step,
        # ongoing attacks may not be reported, or non-existing attacks may be spuriously reported
        probabilities = self.rng.uniform(0, 1, self.g.num_attacks)
        true_positives = self.attack_state & (probabilities <= self.true_positive)
        false_positives = (1 - self.attack_state) & (probabilities <= self.false_positive)
        detected = true_positives | false_positives
        self.observation = tuple(np.append(self.service_state, detected))
        return self.observation

    def step(self, action):
        self.action = action
        assert 0 <= action < self.num_actions

        self.simulation_time += 1
        self.done = False
        attacker_reward = 0

        # reserve 0 for no action
        if action:
            # decrement to obtain index
            service = action - 1
            # only disable services that are still on
            if self.service_state[service]:
                # disable the service itself and any dependent services
                self.service_state[self.dependent_services[service]] = 0
                # remove dependent attacks from the attack surface
                for attack_index in np.flatnonzero(self.attack_surface):
                    required_services, _, _ = self.attack_prerequisites[attack_index]
                    if not all(enabled(required_services, self.service_state)):
                        self.attack_surface[attack_index] = 0

                # end episode when attack surface becomes empty
                self.done = not any(self.attack_surface)

        self.attack_index = None

        if not self.done:
            # obtain attacker action, this _can_ be 0 for no action
            self.attack_index = self.attacker.act(self.attack_surface) - 1
            assert -1 <= self.attack_index < self.g.num_attacks

            if self.attack_index != -1:
                # compute attacker reward
                self.ttc_remaining[self.attack_index] -= 1
                if self.ttc_remaining[self.attack_index] == 0:
                    # successful attack, update reward, attack_state, attack_surface
                    attacker_reward = self.rewards[self.attack_index]
                    self.attack_state[self.attack_index] = 1
                    self.attack_surface[self.attack_index] = 0

                    # add eligible children to the attack surface
                    children = self.g.attack_steps[self.g.attack_names[self.attack_index]].children
                    for child_name in children:
                        child_index = self.g.attack_names.index(child_name)
                        required_services, logic, prerequisites = self.attack_prerequisites[
                            child_index
                        ]
                        if (
                            not self.attack_state[child_index]
                            and all(enabled(required_services, self.service_state))
                            and logic(enabled(prerequisites, self.attack_state))
                        ):
                            self.attack_surface[child_index] = 1

                    # end episode when attack surface becomes empty
                    self.done = not any(self.attack_surface)

            # TODO: placeholder, none of the current attackers learn...
            # self.attacker.update(attack_surface, attacker_reward, self.done)

        # compute defender reward
        # positive reward for maintaining services online (1 unit per service)
        # negative reward for the attacker's gains (as measured by `attacker_reward`)
        # FIXME: the reward for maintaining services is _very_ low
        self.reward = sum(self.service_state) - attacker_reward

        self.compromised_steps = self._interpret_attacks()
        self.compromised_flags = [
            step_name for step_name in self.compromised_steps if "flag" in step_name
        ]

        info = {
            "time": self.simulation_time,
            "attack_surface": self.attack_surface,
            "current_step": None
            if self.attack_index is None
            else self.NO_ACTION
            if self.attack_index == -1
            else self.g.attack_names[self.attack_index],
            "ttc_remaining_on_current_step": -1
            if self.attack_index is None or self.attack_index == -1
            else self.ttc_remaining[self.attack_index],
            "compromised_steps": self.compromised_steps,
            "compromised_flags": self.compromised_flags,
        }

        if self.done:
            logger.debug("Attacker done")
            logger.debug(f"Compromised steps: {self.compromised_steps}")
            logger.debug(f"Compromised flags: {self.compromised_flags}")

        return self.observe(), self.reward, self.done, info

    def _interpret_services(self, services=None):
        if services is None:
            services = self.service_state
        return list(np.array(self.g.service_names)[np.flatnonzero(services)])

    def _interpret_attacks(self, attacks=None):
        if attacks is None:
            attacks = self.attack_state
        return list(np.array(self.g.attack_names)[np.flatnonzero(attacks)])

    def _interpret_observation(self, observation=None):
        if observation is None:
            observation = self.observation
        services = observation[: self.g.num_services]
        attacks = observation[self.g.num_services :]
        return self._interpret_services(services), self._interpret_attacks(attacks)

    def _interpret_action(self, action):
        return (
            self.NO_ACTION
            if action == 0
            else self.g.service_names[action - 1]
            if 0 < action <= self.g.num_services
            else "invalid action"
        )

    def _interpret_action_probabilities(self, action_probabilities):
        keys = [self.NO_ACTION] + self.g.service_names
        return {key: value for key, value in zip(keys, action_probabilities)}

    def _draw_nodes(self, nodes, size, color, border, **kwargs):
        nx.draw_networkx_nodes(
            self.dag,
            self.pos,
            ax=self.ax,
            nodelist=nodes,
            node_size=size,
            node_color=color,
            edgecolors=border,
            linewidths=3,
            **kwargs,
        )

    def _draw_edges(self, edges, **kwargs):
        nx.draw_networkx_edges(
            self.dag, self.pos, ax=self.ax, edgelist=edges, width=2, node_size=1000, **kwargs
        )

    def _draw_labels(self, labels, color, **kwargs):
        nx.draw_networkx_labels(
            self.dag,
            self.pos,
            ax=self.ax,
            labels=labels,
            font_size=8,
            font_weight="bold",
            font_color=color,
            **kwargs,
        )

    def _render_frame(self):
        self.ax.clear()
        reward = self.reward if self.simulation_time else None
        self.ax.set_title(f"Step {self.simulation_time}; Reward: {reward}")

        # draw "or" edges solid (default), "and" edges dashed
        self._draw_edges(self.dag.edges - self.and_edges)
        self._draw_edges(self.and_edges, style="dashed")

        all_attacks = set(range(self.g.num_attacks))
        flags = set([i for i in all_attacks if "flag" in self.g.attack_names[i]])
        observed_attacks = np.array(self.observation[self.g.num_services :])

        observed_ok = set(np.flatnonzero(1 - observed_attacks))
        self._draw_nodes(observed_ok - flags, 1000, "white", "green")
        self._draw_nodes(observed_ok & flags, 1000, "white", "green", node_shape="s")

        observed_ko = set(np.flatnonzero(observed_attacks))
        self._draw_nodes(observed_ko - flags, 1000, "white", "red")
        self._draw_nodes(observed_ko & flags, 1000, "white", "red", node_shape="s")

        fixed_attacks = [i for i in all_attacks if not any(self.attack_prerequisites[i][0])]
        self._draw_nodes(fixed_attacks, 800, "white", "black", node_shape="h")

        # gray out disabled attacks
        disabled_attacks = set(
            [
                i
                for i in all_attacks
                if not all(enabled(self.attack_prerequisites[i][0], self.service_state))
            ]
        )
        self._draw_nodes(disabled_attacks - flags, 800, "lightgray", "lightgray")
        self._draw_nodes(disabled_attacks & flags, 800, "lightgray", "lightgray", node_shape="s")

        # use "forward" triangles for the attack surface, vary color by TTC
        nodes = np.flatnonzero(self.attack_surface)
        colors = self.ttc_remaining[nodes]
        self._draw_nodes(nodes, 800, colors, "red", vmin=0, vmax=256, cmap="RdYlGn", node_shape=">")

        # show attack state by label color
        # safe(ok): GREEN, under attack(kk): BLACK, compromised(ko): RED
        ok_labels = {
            i: f"{self.rewards[i]}\n{self.ttc_remaining[i]}"
            for i in np.flatnonzero(1 - (self.attack_state | self.attack_surface))
        }
        self._draw_labels(ok_labels, "green")
        kk_labels = {
            i: f"{self.rewards[i]}\n{self.ttc_remaining[i]}"
            for i in np.flatnonzero(self.attack_surface)
        }
        self._draw_labels(kk_labels, "black", horizontalalignment="right")
        ko_labels = {i: f"{self.rewards[i]}" for i in np.flatnonzero(self.attack_state)}
        self._draw_labels(ko_labels, "red")

    def render(self, mode="human"):

        render_dir = "render"  # TODO move this variable to a command line option

        if not os.path.isdir(render_dir):
            os.mkdir(render_dir)

        out_dir = os.path.join(render_dir, self.episode_id)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        if self.save_graphs:
            if "graph" not in self.writers:
                if not self.dag:
                    self.dag = nx_digraph(self.g)
                    self.pos = nx_dag_layout(self.dag)
                    self.and_edges = [
                        (i, j)
                        for i, j in self.dag.edges
                        if self.g.attack_steps[self.g.attack_names[j]].step_type == "and"
                    ]
                    self.xlim, self.ylim = tuple(
                        map(lambda l: (min(l), max(l)), zip(*self.pos.values()))
                    )
                xmin, xmax = self.xlim
                ymin, ymax = self.ylim
                fig, self.ax = plt.subplots(figsize=(xmax - xmin, ymax - ymin))
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                plt.axis("off")

                writer = HTMLWriter()
                html_path = os.path.join(out_dir, "render.html")
                writer.setup(fig, html_path, dpi=None)
                writer.frame_format = "svg"
                self.writers["graph"] = writer

        if self.save_text:
            if "text" not in self.writers:
                txt_path = os.path.join(out_dir, "log.txt")
                writer = open(txt_path, "w")
                self.writers["text"] = writer

        if self.save_graphs:
            self._render_frame()
            add_tooltips(self.pos, self.g.attack_names, ax=self.ax)
            writer = self.writers["graph"]
            writer.grab_frame()
            postprocess_frame(writer._temp_paths[-1], self.pos.keys())

        if self.save_text:
            writer = self.writers["text"]
            string_to_write = ""
            string_to_write += f"Step {self.simulation_time}: "

            if self.simulation_time:
                string_to_write += f"Defender disables {self._interpret_action(self.action)}. "
                if self.attack_index is None:
                    string_to_write += "Attacker didn't have a chance"
                elif self.attack_index == -1:
                    string_to_write += "Attacker chose not to attack. "
                else:
                    string_to_write += f"Attacker attacks {self.g.attack_names[self.attack_index]}."
                    string_to_write += f" Remaining TTC: {self.ttc_remaining[self.attack_index]}. "
                string_to_write += f"Reward: {self.reward}. "
            string_to_write += f"Attack surface: {self._interpret_attacks(self.attack_surface)}.\n"
            if self.simulation_time and self.done:
                string_to_write += "Attack is complete.\n"
                string_to_write += f"Compromised steps: {self.compromised_steps}\n"
                string_to_write += f"Compromised flags: {self.compromised_flags}\n"
            writer.write(string_to_write)

        if self.done:
            if self.save_graphs:
                writer = self.writers["graph"]
                writer.finish()
                plt.close()
                postprocess_html(writer.outfile)
            if self.save_text:
                self.writers["text"].close()

            self.writers = {}

        return True

    def seed(self, seed=None):
        self.rng, self._seed = get_rng(seed)
        return self._seed
