from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker


class AttackSimCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):

        info = episode.last_info_for()

        episode.custom_metrics["compromised_steps"] = len(info["compromised_steps"])
        episode.custom_metrics["compromised_flags"] = len(info["compromised_flags"])
