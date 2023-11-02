from ..mal.observation import Info, Observation
import numpy as np
from gymnasium import spaces
from typing import Any, Dict
BIG_INT = 2 ** 63 - 2

class Defender:
    @staticmethod
    def obs_space(n_actions: int, n_objects: int, n_edges: int) -> spaces.Dict:
        return spaces.Dict(
            {
                "action_mask": spaces.Box(0, 1, shape=(n_actions,), dtype=np.int8),
                "node_surface": spaces.Box(0, 1, shape=(n_objects,), dtype=np.int8),
                "ids_observation": spaces.Box(0, 1, shape=(n_objects,), dtype=np.int8),
                "asset": spaces.Box(0, BIG_INT, shape=(n_objects,), dtype=np.int64),
                "asset_id": spaces.Box(0, BIG_INT, shape=(n_objects,), dtype=np.int64),
                "step_name": spaces.Box(0, BIG_INT, shape=(n_objects,), dtype=np.int64),
                "edges": spaces.Box(
                    0,
                    BIG_INT,
                    shape=(2, n_edges),
                    dtype=np.int64,
                ),
            }
        )

    @staticmethod
    def get_info(info: Info) -> Dict[str, Any]:
        return {
            "perc_defenses_activated": info.perc_defenses_activated,
        }

    @staticmethod
    def get_obs(obs: Observation) -> Dict[str, Any]:
        #state = np.array(obs.nodes, dtype=np.int8)
        #edges = np.array(obs.edges, dtype=np.int64)
        # defense_indices = np.flatnonzero(obs.defense_surface)

        # wait_index = len(state)
        # new_state = np.concatenate([state, np.array([1], dtype=np.int8)])
        # Add edges for wait action

        # wait_edges = [[i, wait_index] for i in defense_indices]

        # defense_indices = np.concatenate([np.array([wait_index], dtype=np.int64), defense_indices])

        # Flip the edges for defense steps
        #flipped_edges = [edge[::-1] for edge in edges if edge[0] in defense_indices]

        # remove old edges
        #edges_without_defense = [edge for edge in edges if edge[0] not in defense_indices]

        #new_edges = np.concatenate([edges_without_defense, flipped_edges], axis=0)
        # new_edges = np.concatenate([edges_without_defense, wait_edges, flipped_edges], axis=0)

        # import networkx as nx
        # G = nx.DiGraph()

        # for i, node in enumerate(state):
        #     G.add_node(i, label=node)

        # G.add_edges_from(new_edges)
        # node_colors = ["red" if node in defense_indices else "blue" for node in G.nodes()]
        # pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        # nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors)
        # nx.draw_networkx_edges(G, pos=pos)
        # nx.draw_networkx_labels(G, pos=pos, labels=dict(zip(range(len(state)), state)))
        # import matplotlib.pyplot as plt
        # plt.show()

        return {
            "action_mask": obs.defender_possible_actions,
            "node_surface": obs.defender_possible_objects,
            "ids_observation": obs.state,
            "asset": obs.assets,
            "asset_id": obs.asset_ids,
            "step_name": obs.names,
            "edges": obs.edges.T,
        }


class Attacker:
    @staticmethod
    def get_info(info: Info) -> Dict[str, Any]:
        return {
            "num_compromised_steps": info.num_compromised_steps,
            "perc_compromised_steps": info.perc_compromised_steps,
            "perc_compromised_flags": info.perc_compromised_flags,
            "sum_ttc_remaining": info.sum_ttc,
        }

    @staticmethod
    def obs_space(n_actions:int, n_nodes:int) -> spaces.Dict:
        return spaces.Dict(
            {
                "action_mask": spaces.Box(
                    0,
                    1,
                    shape=(n_actions,),
                    dtype=np.int8,
                ),
                "node_surface": spaces.Box(
                    0,
                    1,
                    shape=(n_nodes,),
                    dtype=np.int8,
                ),
                "ttc_remaining": spaces.Box(
                    0,
                    BIG_INT,
                    shape=(n_nodes,),
                    dtype=np.int64,
                ),
                "state": spaces.Box(0, 1, shape=(n_nodes,), dtype=np.int8),
                "asset": spaces.Box(0, BIG_INT, shape=(n_nodes,), dtype=np.int64),
                "asset_id": spaces.Box(0, BIG_INT, shape=(n_nodes,), dtype=np.int64),
                "step_name": spaces.Box(0, BIG_INT, shape=(n_nodes,), dtype=np.int64),
                "nop_index": spaces.Discrete(n_actions),
            }
        )

    @staticmethod
    def get_obs(obs: Observation) -> Dict[str, Any]:
        return {
            "action_mask": obs.attacker_possible_actions,
            "node_surface": obs.attacker_possible_objects,
            "state": obs.state,
            "asset": obs.assets,
            "asset_id": obs.asset_ids,
            "step_name": obs.names,
            "ttc_remaining": obs.ttc_remaining,
            "nop_index": 0,
        }

    @staticmethod
    def done(obs: Observation) -> bool:
        attack_surface = obs.attacker_possible_objects
        return not any(attack_surface)
