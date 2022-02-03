import os
import shutil

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import HTMLWriter

from .nx_utils import nx_dag_layout
from .sim import AttackSimulator
from .svg_tooltips import add_tooltips, postprocess_frame, postprocess_html
from .utils import enabled


class AttackSimulationRenderer:
    RENDER_DIR = "render"
    HTML = "index.html"
    LOGS = "attack.log"

    def __init__(
        self,
        sim: AttackSimulator,
        episode,
        rewards,
        subdir=None,
        destructive=True,
        save_graph=False,
        save_logs=False,
    ):
        self.sim: AttackSimulator = sim
        self.dag: nx.DiGraph = None
        self.writers = {}
        self.save_graph = save_graph
        self.save_logs = save_logs
        self.episode = episode
        self.rewards = rewards

        if subdir is None:
            self.run_dir = os.path.join(self.RENDER_DIR, f"seed={self.sim.config.seed}")
        else:
            self.run_dir = os.path.join(self.RENDER_DIR, f"{subdir}_seed={self.sim.config.seed}")

        if os.path.exists(self.run_dir):
            if destructive:
                shutil.rmtree(self.run_dir)
            else:
                raise RuntimeError("Render subdir already exists.")

        for d in [self.RENDER_DIR, self.run_dir, self.out_dir]:
            if not os.path.isdir(d):
                os.mkdir(d)

        if self.save_logs:
            self.writers["logs"] = open(os.path.join(self.out_dir, self.LOGS), "w", encoding="utf8")

        if self.save_graph:
            self.dag = self.sim.g.to_networkx()
            self.pos = nx_dag_layout(self.dag)
            self.and_edges = [
                (i, j)
                for i, j in self.dag.edges
                if self.sim.g.attack_steps[self.sim.g.attack_names[j]].step_type == "and"
            ]
            self.xlim, self.ylim = tuple(map(lambda l: (min(l), max(l)), zip(*self.pos.values())))

            xmin, xmax = self.xlim
            ymin, ymax = self.ylim
            dx = xmax - xmin
            dy = ymax - ymin

            # create a figure with two areas: one for the graph and another for the logs
            fig, (self.ax, ax) = plt.subplots(
                nrows=2,
                figsize=(dx, dy + 1.25),
                gridspec_kw=dict(height_ratios=[dy, 1]),
                constrained_layout=True,
            )

            # graph area
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.ax.axes.get_xaxis().set_visible(False)
            self.ax.axes.get_yaxis().set_visible(False)

            # log area
            ax.set_frame_on(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            self.log = ax.text(
                0.25 / dx,
                0.9,
                "",
                horizontalalignment="left",
                verticalalignment="top",
                fontfamily="fantasy",
                wrap=True,
                bbox=dict(facecolor="lightgray", boxstyle="Round"),
            )

            writer: HTMLWriter = HTMLWriter()
            html_path = os.path.join(self.out_dir, self.HTML)
            writer.setup(fig, html_path, dpi=None)
            writer.frame_format = "svg"
            self.writers["graph"] = writer

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

    def _render_graph(self, rewards):
        self.ax.clear()

        # draw "or" edges solid (default), "and" edges dashed
        self._draw_edges(self.dag.edges - self.and_edges)
        self._draw_edges(self.and_edges, style="dashed")

        all_attacks = set(range(self.sim.g.num_attacks))
        flags = set([i for i in all_attacks if "flag" in self.sim.g.attack_names[i]])
        observed_attacks = np.array(self.sim.observe()[self.sim.g.num_services :])

        observed_ok = set(np.flatnonzero(1 - observed_attacks))
        self._draw_nodes(observed_ok - flags, 1000, "white", "green")
        self._draw_nodes(observed_ok & flags, 1000, "white", "green", node_shape="s")

        observed_ko = set(np.flatnonzero(observed_attacks))
        self._draw_nodes(observed_ko - flags, 1000, "white", "red")
        self._draw_nodes(observed_ko & flags, 1000, "white", "red", node_shape="s")

        fixed_attacks = [i for i in all_attacks if not any(self.sim.g.attack_prerequisites[i][0])]
        self._draw_nodes(fixed_attacks, 800, "white", "black", node_shape="h")

        # gray out disabled attacks
        disabled_attacks = set(
            [
                i
                for i in all_attacks
                if not all(enabled(self.sim.g.attack_prerequisites[i][0], self.sim.service_state))
            ]
        )
        self._draw_nodes(disabled_attacks - flags, 800, "lightgray", "lightgray")
        self._draw_nodes(disabled_attacks & flags, 800, "lightgray", "lightgray", node_shape="s")

        # use "forward" triangles for the attack surface, vary color by TTC
        nodes = np.flatnonzero(self.sim.attack_surface)
        colors = self.sim.ttc_remaining[nodes]
        self._draw_nodes(nodes, 800, colors, "red", vmin=0, vmax=256, cmap="RdYlGn", node_shape=">")

        # show attack state by label color
        # safe(ok): GREEN, under attack(kk): BLACK, compromised(ko): RED
        ok_labels = {
            i: f"{rewards[i]}\n{self.sim.ttc_remaining[i]}"
            for i in np.flatnonzero(1 - (self.sim.attack_state | self.sim.attack_surface))
        }
        self._draw_labels(ok_labels, "green")
        kk_labels = {
            i: f"{rewards[i]}\n{self.sim.ttc_remaining[i]}"
            for i in np.flatnonzero(self.sim.attack_surface)
        }
        self._draw_labels(kk_labels, "black", horizontalalignment="right")
        ko_labels = {i: f"{rewards[i]}" for i in np.flatnonzero(self.sim.attack_state)}
        self._draw_labels(ko_labels, "red")

    def _generate_logs(self, defender_reward):
        logs = f"Step {self.sim.time}: "
        if self.sim.time:
            logs += f"Defender disables {self.sim.interpret_action(self.sim.defender_action)}. "
            if self.sim.attack_index is None:
                logs += "Attacker didn't have a chance"
            elif self.sim.attack_index == -1:
                logs += "Attacker chose not to attack."
            else:
                logs += f"Attacker attacks {self.sim.g.attack_names[self.sim.attack_index]}. "
                logs += f"Remaining TTC: {self.sim.ttc_remaining[self.sim.attack_index]}. "
            logs += f"Reward: {defender_reward}. "
        logs += f"Attack surface: {self.sim.interpret_attacks(self.sim.attack_surface)}.\n"
        if self.sim.time and self.sim.done:
            logs += "Attack is complete.\n"
            logs += f"Compromised steps: {self.sim.compromised_steps}\n"
            logs += f"Compromised flags: {self.sim.compromised_flags}\n"
        return logs

    @property
    def out_dir(self):
        return os.path.join(self.run_dir, f"ep-{self.episode}")

    def render(self, defender_reward):

        logs = self._generate_logs(defender_reward)

        if self.save_logs:
            self.writers["logs"].write(logs)

        if self.save_graph:
            self.log.set_text(logs)
            self._render_graph(self.rewards)
            add_tooltips(self.pos, self.sim.g.attack_names, ax=self.ax)
            writer: HTMLWriter = self.writers["graph"]
            writer.grab_frame()
            postprocess_frame(
                writer._temp_paths[-1], self.pos.keys()
            )  # pylint: disable=protected-access

        if self.sim.done:
            if self.save_graph:
                writer: HTMLWriter = self.writers["graph"]
                writer.finish()
                plt.close()
                postprocess_html(writer.outfile)
            if self.save_logs:
                self.writers["logs"].close()
            self.writers = {}

        return True
