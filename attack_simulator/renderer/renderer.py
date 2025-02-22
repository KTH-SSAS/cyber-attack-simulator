import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import HTMLWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from numpy.typing import NDArray

from ..constants import ACTION_WAIT
from ..mal.graph import AttackGraph
from .svg_tooltips import add_tooltips, make_paths_relative, postprocess_frame, postprocess_html

NODE_SIZE = 1000
INNER_NODE_SIZE = 800
ATTACKER_SYMBOL_SIZE = 700


def create_HTML_writer(fig: Figure, html_path: Path) -> HTMLWriter:
    writer: HTMLWriter = HTMLWriter(fps=2)
    writer.setup(fig, html_path, dpi=None)
    writer.frame_format = "svg"
    return writer


def create_axes(pos: dict, width: int, height: int, dpi: int) -> Tuple[Axes, Text, Figure]:
    # create a figure with two areas: one for the graph and another for the logs
    log_ax: Axes
    graph_ax: Axes
    fig, (graph_ax, log_ax) = plt.subplots(
        nrows=2,
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        # gridspec_kw=dict(height_ratios=[0.7, 0.3]),
        constrained_layout=False,
        tight_layout=True,
    )

    # graph area

    # Graph window dimensions
    # xlim, ylim = tuple(map(lambda l: (min(l), max(l)), zip(*pos.values())))
    # xmin, xmax = xlim
    # ymin, ymax = ylim
    # graph_ax.set_xlim(xmin, xmax)
    # graph_ax.set_ylim(ymin, ymax)
    graph_ax.set_axis_off()

    # log area
    log_ax.set_frame_on(False)
    log_ax.set_axis_off()
    text_ax: Text = log_ax.text(
        0.25 / width,
        0.9,
        "",
        horizontalalignment="left",
        verticalalignment="top",
        fontfamily="DejaVu Sans",
        wrap=True,
        bbox=dict(facecolor="lightgray", boxstyle="Round"),
    )

    return graph_ax, text_ax, fig


def add_to_logline(logline: str, string: str) -> str:
    return " ".join([logline, string])


def _generate_logs(
    state: Dict[str, Any], graph: AttackGraph, defender_reward: float, done: bool
) -> str:
    time = state["time"]
    attacker_action = state["attacker_action"]
    defender_action = state["defender_action"]
    attack_surface = state["node_surface"]
    defense_state = state["defense_state"]
    ttc_remaining = state["ttc_remaining"]
    attack_surface_empty = np.all(attack_surface == 0)

    logs = []
    logs.append(f"Step {time}:")

    if time == 0:
        logs.append("Simulation starting.")
    else:
        logs.append(f"Defender selects {defender_action}:{defender_action}.")
        if attack_surface_empty:
            logs.append("Attacker can not attack anything.")
        elif attacker_action == ACTION_WAIT:
            logs.append("Attacker does nothing.")
        else:
            logs.append(f"Attacker attacks {graph.attack_names[attacker_action]}.")
            logs.append(f"Remaining TTC: {ttc_remaining[attacker_action]}.")
        logs.append(f"Defender reward: {defender_reward}.")

    logs.append(f"Attack surface: {graph.interpret_attacks(attack_surface)}.")
    logs.append(f"Defense steps used: {graph.interpret_defenses(defense_state)}")

    if done:
        logs.append("Simulation finished.")

    logline = " ".join(logs) + "\n"

    return logline


class AttackSimulationRenderer:
    """Render a simulation."""

    HTML = "index.html"
    LOGS = "attack.log"

    def __init__(
        self,
        run_id: str,
        episode: int,
        save_graph: bool = False,
        save_logs: bool = False,
    ):
        self.graph = AttackGraph(None)
        self.save_graph = save_graph
        self.save_logs = save_logs
        self.add_tooltips = False
        self.episode = episode

        render_dir = Path("render")

        self.run_dir = render_dir / run_id

        if self.out_dir.is_dir():
            shutil.rmtree(self.out_dir)

        self.out_dir.mkdir(parents=True)

        if self.save_graph:
            self.dag = self.graph.to_networkx(
                indices=True, system_state=np.ones(self.graph.num_defenses)
            )
            self.pos: Dict[int, Tuple[float, float]] = nx.nx_pydot.graphviz_layout(
                self.dag, root=self.graph.root, prog="sfdp"
            )
            self.and_edges = {
                (i, j)
                for i, j in self.dag.edges
                if self.graph.attack_steps[self.graph.attack_names[j]].step_type == STEP.AND
            }

            height = 1500
            width = int(height * 1.77777)
            dpi = 100
            self.ax, self.log, fig = create_axes(self.pos, width, height, dpi)
            self.graph_writer = create_HTML_writer(fig, self.out_dir / self.HTML)

    def _draw_nodes(
        self,
        nodes: set,
        size: int,
        color: str,
        border: str,
        vmax: Optional[float] = None,
        vmin: Optional[float] = None,
        **kwargs: str,
    ) -> None:
        nx.draw_networkx_nodes(
            self.dag,
            self.pos,
            ax=self.ax,
            nodelist=nodes,
            node_size=size,
            node_color=color,
            edgecolors=border,
            linewidths=2,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

    def _draw_edges(self, edges: Set[Tuple[int, int]], **kwargs: str) -> None:
        nx.draw_networkx_edges(
            self.dag, self.pos, ax=self.ax, edgelist=edges, width=2, node_size=1000, **kwargs
        )

    def _draw_labels(self, labels: dict, color: str, **kwargs: str) -> None:
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

    def _render_graph(self, state: Dict[str, Any]) -> None:
        self.ax.clear()

        # draw "or" edges solid (default), "and" edges dashed
        self._draw_edges(self.dag.edges - self.and_edges)
        self._draw_edges(self.and_edges, style="dashed")

        all_attacks = set(range(self.graph.num_attacks))
        flags = set(self.graph.flag_indices)
        # observed_attacks = np.array(self.observe()[self.g.num_defenses :])

        attack_state = state["attack_state"]
        false_alerts = state["false_positives"]
        missed_alerts = state["false_negatives"]
        defense_state = state["defense_state"]
        attack_surface = state["node_surface"]
        ttc_remaining = state["ttc_remaining"]
        attacker_action = state["attacker_action"]

        # Draw uncompromised steps as green squares
        observed_ok = set(np.flatnonzero(1 - attack_state))
        self._draw_nodes(observed_ok - flags, NODE_SIZE, "white", "green")
        self._draw_nodes(observed_ok & flags, NODE_SIZE, "white", "green", node_shape="s")

        # Draw compromised steps as red squares
        observed_ko = set(np.flatnonzero(attack_state))
        self._draw_nodes(observed_ko - flags, NODE_SIZE, "white", "red")
        self._draw_nodes(observed_ko & flags, NODE_SIZE, "white", "red", node_shape="s")

        # Draw false positives as yellow squares
        false_alerts = set(np.flatnonzero(false_alerts))
        self._draw_nodes(false_alerts - flags, INNER_NODE_SIZE - 100, "white", "purple")
        self._draw_nodes(
            false_alerts & flags, INNER_NODE_SIZE - 100, "white", "purple", node_shape="s"
        )

        # Draw false negatives as blue squares
        missed_alerts = set(np.flatnonzero(missed_alerts))
        self._draw_nodes(missed_alerts - flags, INNER_NODE_SIZE - 100, "white", "blue")
        self._draw_nodes(
            missed_alerts & flags, INNER_NODE_SIZE - 100, "white", "blue", node_shape="s"
        )

        # Draw attacks without defense steps as hexagons
        fixed_attacks = self.graph.get_undefendable_steps()
        self._draw_nodes(set(fixed_attacks), INNER_NODE_SIZE, "white", "black", node_shape="H")

        # Gray out disabled attacks
        disabled_attacks = {
            i
            for i in all_attacks
            if not all(defense_state[self.graph.defense_steps_by_attack_step[i]])
        }

        self._draw_nodes(disabled_attacks - flags, INNER_NODE_SIZE, "lightgray", "lightgray")
        self._draw_nodes(
            disabled_attacks & flags, INNER_NODE_SIZE, "lightgray", "lightgray", node_shape="s"
        )

        # use "forward" triangles for the attack surface, vary color by TTC
        nodes = np.flatnonzero(attack_surface)
        colors = ttc_remaining[nodes]

        self._draw_nodes(
            set(nodes),
            ATTACKER_SYMBOL_SIZE,
            colors,
            "red",
            vmin=0,
            vmax=256,
            cmap="RdYlGn",
            node_shape="H",
        )

        attacked_node = attacker_action
        if attacked_node != ACTION_WAIT:
            self._draw_nodes({attacked_node}, 700, "yellow", "orange", node_shape="H")

        # show attack state by label color
        # safe(ok): GREEN, under attack(kk): BLACK, compromised(ko): RED
        ok_labels: Dict[int, str] = {
            i: self.get_node_label(i) for i in np.flatnonzero(1 - (attack_state | attack_surface))
        }
        self._draw_labels(ok_labels, "green")
        kk_labels = {i: self.get_node_label(i) for i in np.flatnonzero(attack_surface)}
        self._draw_labels(kk_labels, "black")
        ko_labels = {i: self.get_node_label(i) for i in np.flatnonzero(attack_state)}
        self._draw_labels(ko_labels, "red")

        nx.draw_networkx_labels(
            self.dag,
            {k: (x + 10, y + 10) for k, (x, y) in self.pos.items()},
            ax=self.ax,
            font_size=8,
            font_weight="bold",
            font_color="gray",
        )

    def get_node_label(
        self, step_id: int, ttc_remaining: Optional[NDArray[np.int64]] = None
    ) -> str:
        rewards = self.graph.reward_params

        step_defense = [str(i) for i in self.graph.defense_steps_by_attack_step[step_id]]
        defense_string = ",".join(step_defense)

        stats = []

        if ttc_remaining:
            stats += [f"T:{str(ttc_remaining[step_id])}"]

        stats += [f"R{str(rewards[step_id])}"] if rewards[step_id] > 0 else []
        stats_label = "\n".join(stats)

        full_label = []
        full_label += [stats_label] if stats_label != "" else []
        full_label += [defense_string] if defense_string != "" else []

        return "\n".join(full_label)

    @property
    def out_dir(self) -> Path:
        return self.run_dir / f"ep-{self.episode}"

    def finish(self, state: Dict[str, Any]) -> None:
        compromised_steps = state["attack_state"]
        compromised_flags = state["attack_state"][self.graph.flag_indices["flags"]]

        if self.save_graph:
            plt.close()
            self.graph_writer.finish()
            make_paths_relative(self.graph_writer.outfile)
            if self.add_tooltips:
                postprocess_html(self.graph_writer.outfile)

        if self.save_logs:
            with open(self.out_dir / self.LOGS, "a", encoding="utf8") as f:
                logs = "Attack is complete.\n"
                logs += f"Compromised steps: {compromised_steps}\n"
                logs += f"Compromised flags: {compromised_flags}\n"
                f.write(logs)

        try:
            # Try to remove directory if empty
            self.out_dir.rmdir()
        except OSError:
            # Will fail if dir is not empty
            pass

    def render(self, state: Dict[str, Any], defender_reward: float, done: bool) -> None:
        """Render a frame."""
        logs = _generate_logs(state, self.graph, defender_reward, done)

        if self.save_logs:
            with open(self.out_dir / self.LOGS, "a", encoding="utf8") as f:
                f.write(logs)

        if self.save_graph:
            self.log.set_text(logs)
            self._render_graph(state)
            if self.add_tooltips:
                add_tooltips(self.pos, self.graph.attack_names, ax=self.ax)
            writer: HTMLWriter = self.graph_writer
            writer.grab_frame()
            if self.add_tooltips:
                postprocess_frame(writer._temp_paths[-1], list(self.pos.keys()))  # pylint: disable=protected-access

        if done:
            self.finish(state)
