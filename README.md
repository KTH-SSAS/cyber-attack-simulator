# Reinforcement learning on attack simulations

An agent learns to defend a computer network by playing games against an unlearning attacker in a [static attack graph](docs/graphviz.pdf) of the computer network. The computer network is modelled on the cyber range for the KTH Ethical Hacking course. The defender receives positive rewards for maintaining services (FTP, web server, etc) and negative rewards when the attacker captures flags. 

[Project Kanban](https://github.com/KTH-SSAS/attack-simulator/projects/1)

## Usage

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


## Packaging and dependency managament.

This Python project uses [`poetry`](https://python-poetry.org)
for packaging and dependency managament.

### Prerequisites

Make sure you have Python (incuding `pip`) installed and
run the following commands to get your environment set up.

```
$ pip install poetry
$ poetry install
```

The first command installs `poetry` itself, while the second one
creates a virtual environment with all dependencies and development
tools installed.


### Using the project environment

To run a single command in the context of the project environment use

```
$ poetry run CMD ARGS
```

For example, the main `train-reinforce` script can be invoked as
```
$ poetry run scripts/train-reinforce -h
```

Alternatively, an interactive session can be executed inside
the project environment by issuing the command

```
$ poetry shell
(attack-simulator-XXXXXXXX-py3.8) $
```

This is particularly useful for testing and development.
For a battery of tests and quality checks, for instance, run

```
(attack-simulator-XXXXXXXX-py3.8) $ tox
```
