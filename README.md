# Reinforcement learning on attack simulations

## Packaging and dependency managament.

This Python project uses [`poetry`](https://python-poetry.org)
for packaging and dependency managament.

It also uses [maturin](https://github.com/PyO3/maturin) to build the rust version of the simulator backend.

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

To build and install the rust package into the current venv, run

```
maturin develop
```
