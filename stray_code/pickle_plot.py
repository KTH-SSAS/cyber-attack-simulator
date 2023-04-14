import pickle

import matplotlib.pyplot as plt

from attack_simulator.renderer.renderer import AttackSimulationRenderer
from attack_simulator.sim import AttackSimulator

with open("sim_dump.pkl", "rb") as f:
    data: AttackSimulator = pickle.load(f)

renderer = AttackSimulationRenderer(data, "test", 0, save_graph=True, save_logs=True)
renderer.render([], 0, False)
plt.show()
