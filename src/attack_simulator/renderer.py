from pathlib import Path
from typing import Dict, Optional, Set, Tuple
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import HTMLWriter
from matplotlib.axes import Axes
from matplotlib.text import Text

from .constant import AND
from .sim import AttackSimulator
from .svg_tooltips import add_tooltips, make_paths_relative, postprocess_frame, postprocess_html

def create_HTML_writer(fig: Figure, html_path: Path) -> HTMLWriter:
    writer: HTMLWriter = HTMLWriter()
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
        gridspec_kw=dict(height_ratios=[0.7, 0.3]),
        constrained_layout=False,
    )

    # graph area

    # Graph window dimensions
    xlim, ylim = tuple(map(lambda l: (min(l), max(l)), zip(*pos.values())))
    xmin, xmax = xlim
    ymin, ymax = ylim
    graph_ax.set_xlim(xmin, xmax)
    graph_ax.set_ylim(ymin, ymax)
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

def _generate_logs(sim: AttackSimulator, defender_reward: float) -> str:
    logs = f"Step {sim.time}: "

    if sim.time == 0:
        logs += "Simulation starting."
    else:
        logs += f"Defender disables {sim.interpret_action(sim.defender_action)}. "
        if sim.attack_index is None:
            logs += "Attacker can not attack anything."
        elif sim.attack_index == -1:
            logs += "Attacker does nothing."
        else:
            logs += f"Attacker attacks {sim.g.attack_names[sim.attack_index]}. "
            logs += f"Remaining TTC: {sim.ttc_remaining[sim.attack_index]}."
        logs += f"Defender reward: {defender_reward}."

    logs += f" Attack surface: {sim.interpret_attacks(sim.attack_surface)}.\n"

    return logs
class AttackSimulationRenderer:
    """Render a simulation."""

    HTML = "index.html"
    LOGS = "attack.log"

    def __init__(
        self,
        sim: AttackSimulator,
        run_id: str,
        episode: int,
        save_graph: bool = False,
        save_logs: bool = False,
    ):
        self.sim: AttackSimulator = sim
        self.save_graph = save_graph
        self.save_logs = save_logs
        self.add_tooltips = False
        self.episode = episode

        render_dir = Path("render")

        self.run_dir = render_dir / run_id

        if not self.out_dir.is_dir():
            self.out_dir.mkdir(parents=True)

        if self.save_graph:
            self.dag = self.sim.g.to_networkx(indices=True, system_state=np.ones(self.sim.num_defense_steps))
            self.pos = nx.nx_pydot.graphviz_layout(
                self.dag, root=self.sim.entry_attack_index, prog="dot"
            )
            self.and_edges = {
                (i, j)
                for i, j in self.dag.edges
                if self.sim.g.attack_steps[self.sim.g.attack_names[j]].step_type == AND
            }

            width = 1920
            height = 1080
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
            linewidths=3,
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

    def _render_graph(self) -> None:
        self.ax.clear()

        rewards = self.sim.g.reward_params

        # draw "or" edges solid (default), "and" edges dashed
        self._draw_edges(self.dag.edges - self.and_edges)
        self._draw_edges(self.and_edges, style="dashed")

        all_attacks = set(range(self.sim.g.num_attacks))
        flags = set(self.sim.g.flags)
        observed_attacks = np.array(self.sim.observe()[self.sim.g.num_defenses :])

        # Draw uncompromised steps as green squares
        observed_ok = set(np.flatnonzero(1 - observed_attacks))
        self._draw_nodes(observed_ok - flags, 1000, "white", "green")
        self._draw_nodes(observed_ok & flags, 1000, "white", "green", node_shape="s")

        # Draw compromised steps as red squares
        observed_ko = set(np.flatnonzero(observed_attacks))
        self._draw_nodes(observed_ko - flags, 1000, "white", "red")
        self._draw_nodes(observed_ko & flags, 1000, "white", "red", node_shape="s")

        # Draw attacks without defense steps as hexagons
        fixed_attacks = {i for i in all_attacks if not self.sim.g.defense_steps_by_attack_step[i]}
        self._draw_nodes(fixed_attacks, 800, "white", "black", node_shape="h")

        # Gray out disabled attacks
        disabled_attacks = {
            i
            for i in all_attacks
            if not all(self.sim.defense_state[self.sim.g.defense_steps_by_attack_step[i]])
        }

        self._draw_nodes(disabled_attacks - flags, 800, "lightgray", "lightgray")
        self._draw_nodes(disabled_attacks & flags, 800, "lightgray", "lightgray", node_shape="s")

        # use "forward" triangles for the attack surface, vary color by TTC
        nodes = np.flatnonzero(self.sim.attack_surface)
        colors = self.sim.ttc_remaining[nodes]
        self._draw_nodes(
            set(nodes), 800, colors, "red", vmin=0, vmax=256, cmap="RdYlGn", node_shape=">"
        )

        # show attack state by label color
        # safe(ok): GREEN, under attack(kk): BLACK, compromised(ko): RED
        ok_labels: Dict[int, str] = {
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



    @property
    def out_dir(self) -> Path:
        return self.run_dir / f"ep-{self.episode}"

    def finish(self) -> None:
        if self.save_graph:
            plt.close()
            self.graph_writer.finish()
            make_paths_relative(self.graph_writer.outfile)
            if self.add_tooltips:
                postprocess_html(self.graph_writer.outfile)

        if self.save_logs:
            with open(self.out_dir / self.LOGS, "a", encoding="utf8") as f:
                logs = "Attack is complete.\n"
                logs += f"Compromised steps: {self.sim.compromised_steps}\n"
                logs += f"Compromised flags: {self.sim.compromised_flags}\n"
                f.write(logs)

        try:
            # Try to remove directory if empty
            self.out_dir.rmdir()
        except OSError:
            # Will fail if dir is not empty
            pass

    def render(self, defender_reward: float, done: bool) -> None:
        """Render a frame."""

        logs = _generate_logs(self.sim, defender_reward)

        if self.save_logs:
            with open(self.out_dir / self.LOGS, "a", encoding="utf8") as f:
                f.write(logs)

        if self.save_graph:
            self.log.set_text(logs)
            self._render_graph()
            if self.add_tooltips:
                add_tooltips(self.pos, self.sim.g.attack_names, ax=self.ax)
            writer: HTMLWriter = self.graph_writer
            writer.grab_frame()
            if self.add_tooltips:
                postprocess_frame(
                    writer._temp_paths[-1], self.pos.keys()
                )  # pylint: disable=protected-access

        if done:
            self.finish()