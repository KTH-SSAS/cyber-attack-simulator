# Reinforcement learning on attack simulations

An agent learns to defend a computer network by playing games against an unlearning attacker in a [static attack graph](docs/graphviz.pdf) of the computer network. The computer network is modelled on the cyber range for the KTH Ethical Hacking course. The defender receives positive rewards for maintaining services (FTP, web server, etc) and negative rewards when the attacker captures flags. 

Checkout the training and simulation options with 

```
python3 scripts/train_reinforce.py -h
```
