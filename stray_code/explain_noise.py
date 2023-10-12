import numpy as np

from attack_simulator.agents.attackers.searchers import DepthFirstAttacker
from attack_simulator.sim import AttackSimulator
from attack_simulator.utils.config import EnvConfig

ids_fpr = 0.5
ids_fnr = 0.5

env_config = {
    "attacker": "depth-first",
    "save_logs": False,
    "seed": 22,
    "sim_config": {
        "seed": 22,
        "attack_start_time": 5,
        "false_positive_rate": ids_fpr,
        "false_negative_rate": ids_fnr,
    },
    "reward_mode": "downtime-penalty",
    "run_id": "simple",
    "graph_config": {
        "ttc": {"easy": 5, "hard": 10, "default": 1},
        "rewards": {
            "high_flag": 10,
            "medium_flag": 10,
            "low_flag": 1,
            "default": 0.0,
            "defense_default": 10,
        },
        # "root": "asset:0:0",
        "root": "attacker:0:enter:0",
        # "root": "internet.connect",
        # "filename": "graphs/big.yaml",
        "filename": "graphs/1way.yaml",
    },
}


env_config = EnvConfig(**env_config)

print(f"Starting simulation with IDS FPR={ids_fpr} and FNR={ids_fnr}")


def prettyprint_pred(pred, true) -> str:
    if pred == true:
        if pred == 1:
            return "TP"
        return "TN"
    if pred == 1:
        return "FP"
    return "FN"


def get_true_positives(preds, trues) -> int:
    return sum(pred == true and pred == 1 for pred, true in zip(preds, trues))


def get_true_negatives(preds, trues) -> int:
    return sum(pred == true and pred == 0 for pred, true in zip(preds, trues))


def prettyprint_classifier(preds, trues) -> str:
    return "[" + " ".join([prettyprint_pred(pred, true) for pred, true in zip(preds, trues)]) + "]"


def run_episode(seed, verbose=False):
    attacker = DepthFirstAttacker({"random_seed": seed})
    sim = AttackSimulator(env_config, seed)
    done = False
    total_true_positives = 0
    total_true_negatives = 0
    total_positives = 0
    total_negatives = 0

    if verbose:
        titles = ["Attack State", "Predicted", "Result", "TPR", "TNR", "FPR", "FNR"]
        print(
            f"{titles[0]:<20}{titles[1]:<20}{titles[2]:<30}{titles[3]:<10}{titles[4]:<10}{titles[5]:<10}{titles[6]:<10}"
        )

    while not done:
        done, _ = sim.attack_action(attacker.act(sim.attacker_observation))
        sim.observe_alt()

        total_positives += np.sum(sim.attack_state)
        total_negatives += np.sum(np.logical_not(sim.attack_state))

        true_positives = get_true_positives(sim.last_observation, sim.attack_state)
        total_true_positives += true_positives
        tpr = true_positives / np.sum(sim.attack_state) if np.sum(sim.attack_state) > 0 else 0.0
        fnr = 1 - tpr
        true_negatives = get_true_negatives(sim.last_observation, sim.attack_state)
        total_true_negatives += true_negatives
        tnr = (
            true_negatives / np.sum(np.logical_not(sim.attack_state))
            if np.sum(np.logical_not(sim.attack_state)) > 0
            else 0.0
        )
        fpr = 1 - tnr

        if verbose:
            prettified = prettyprint_classifier(sim.last_observation, sim.attack_state)
            print(
                f"{str(sim.attack_state):<20}{str(sim.last_observation):<20}{prettified:<30}{tpr:<10.2}{tnr:<10.2}{fpr:<10.2}{fnr:<10.2}"
            )

        sim.step()
        if done:
            break

    precision = (
        total_true_positives / (total_true_positives + total_true_negatives)
        if total_true_positives + total_true_negatives > 0
        else 0.0
    )
    recall = total_true_positives / total_positives

    if verbose:
        print("Summary:")
        print(f"{'FNR':<10}{'FPR':<10}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}")
        print(
            f"{1 - recall:<10.2}"
            f"{1 - total_true_negatives/total_negatives:<10.2}"
            f"{(total_true_positives + total_true_negatives)/(total_positives + total_negatives):<10.2}"
            f"{precision:<10.2}"
            f"{recall:<10.2}"
        )

    return {"fnr": 1 - recall, "fpr": 1 - total_true_negatives / total_negatives}


if __name__ == "__main__":
    num_runs = 100
    print("Running over", num_runs, "runs")
    results = [run_episode(seed) for seed in range(10)]
    print("Average results:")
    print(f"{'FNR':<10}{'FPR':<10}")
    print(
        f"{np.mean([result['fnr'] for result in results]):<10.2}"
        f"{np.mean([result['fpr'] for result in results]):<10.2}"
    )
