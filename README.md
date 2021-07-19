# Reinforcement learning on attack simulations

An agent learns to defend a computer network by playing games against an unlearning attacker in a [static attack graph](docs/graphviz.pdf) of the computer network. The computer network is modelled on the cyber range for the KTH Ethical Hacking course. The defender receives positive rewards for maintaining services (FTP, web server, etc) and negative rewards when the attacker captures flags. 

Assuming all prerequisites are installed on the target system/environment,
the training and simulation options can be listed by running the following
command inside a local checkout of this repo:

```
PYTHONPATH=src ./scripts/train-reinforce -h
```

Alternatively, a more generic version that relies on an environment variable
`REPO_ROOT` that points to a local checkout of this repo looks like this:
```
PYTHONPATH=$REPO_ROOT/src $REPO_ROOT/scripts/train-reinforce -h
```
