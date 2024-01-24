# Reinforcement learning on attack simulations

## Installation

```
pip install -i https://test.pypi.org/simple/ attack-simulator
```

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

# Interface

The unwrapped `Gymnasium` environments `DefenderEnv` and `AttackerEnv` will provide sates on the following form, using a `spaces.Dict`

| Tables        | `spaces` Class| Shape | Type
| ------------- |:-------------| ------ | ---- | 
| `observation` | Box(-1, 1) | (n_objects_in_instance,) | np.int8 | 
| `asset_id`    | Box(0, n_assets_in_instance) | (n_objects_in_instance,) | np.int64 |  
| `asset`       | Box(0, n_object_types_in_domain) | (n_objects_in_instance,) | np.int64 |
| `step_name`   | Box(0, n_step_names_in_domain) | (n_objects_in_instance,) | np.int64  |
| `edges`       | Box(0, n_objects_in_instance) | (2, n_edges) | np.int64 |
| `action_mask` | Tuple((Box(0, 1, shape=(n_actions,), dtype=np.int8,) Box(0, 1, shape=(n_objects,), dtype=np.int8,))) | (n_action, n_objects) | (np.int8, np.int8) | 

The wrapper `wrappers.GraphWrapper` will give the output as a `spaces.Graph`. The action mask is moved to the info dictionary.

| Tables        | `spaces` Class| Shape | Type
| ------------- |:-------------| ------ | ---- |
| `nodes`       | Box(0, MAXINT) | (n_objects_in_instance, 4)    | np.int64 | 
| `edges`       | Box(0, n_objects_in_instance) | (2, n_edges) | np.int64 |
