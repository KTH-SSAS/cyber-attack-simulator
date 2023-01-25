import matplotlib.pyplot as plt
import numpy as np

alert_probs = np.arange(0.1, 1.1, 0.1)

fn_rates = np.arange(0.1, 1.1, 0.1)
fp_rates = np.arange(0.1, 1.1, 0.1)

joint_prob = lambda p, p_xy: p * p_xy

conditional_entropy = 0

fn_rate = 0
fp_rate = 0
alert_prob = 0.3
conditional_probabilites = [1 - fn_rate, fn_rate, fp_rate, 1 - fp_rate]
probs = [
    alert_prob,
]

joint_probs = [
    joint_prob(p, p_xy) for p, p_xy in zip(conditional_probabilites, conditional_probabilites)
]

for i in range(4):
    conditional_entropy += joint_prob(
        conditional_probabilites[i],
    )


plt.figure(figsize=(10, 10))

entropy = [-(p * np.log(p) + (1 - p) * np.log(1 - p)) for p in alert_probs]

plt.plot(alert_probs, entropy)
plt.show()
