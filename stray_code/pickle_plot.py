import pickle
from attack_simulator.sim import AttackSimulator
from attack_simulator.renderer import AttackSimulationRenderer
import matplotlib.pyplot as plt


with open('sim_dump.pkl', 'rb') as f:
	data: AttackSimulator = pickle.load(f)

renderer = AttackSimulationRenderer(data, 'test', 0, save_graph=True, save_logs=True)
renderer.render(0, False)
plt.show()