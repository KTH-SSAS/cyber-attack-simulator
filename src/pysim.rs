use std::{collections::HashMap, fmt::Display};

use crate::{
    config::SimulatorConfig,
    loading::{load_graph_from_yaml, load_graph_from_json},
    observation::{Info, Observation},
    runtime::{SimulatorRuntime, ACTIONS, ACTION_NOP, ACTION_TERMINATE},
};
use pyo3::{exceptions::PyRuntimeError, pyclass, pymethods, PyErr, PyResult};

fn pyresult<T, E: Display>(result: Result<T, E>) -> PyResult<T> {
    pyresult_with(result, "Error in rust_sim")
}

fn pyresult_with<T, E: Display>(result: Result<T, E>, msg: &str) -> PyResult<T> {
    result.map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{}: {}", msg, e)))
}

#[pyclass]
pub(crate) struct RustAttackSimulator {
    runtime: SimulatorRuntime<usize>,
    #[pyo3(get)]
    config: SimulatorConfig,
    #[pyo3(get)]
    num_actions: u64,
    #[pyo3(get)]
    wait_action: usize,
    #[pyo3(get)]
    terminate_action: usize,
    #[pyo3(get)]
    num_attack_steps: usize,
    #[pyo3(get)]
    num_defense_steps: usize,
    #[pyo3(get)]
    attacker_impact: Vec<i64>,
    #[pyo3(get)]
    defender_impact: Vec<i64>,
}

#[pymethods]
impl RustAttackSimulator {
    #[new]
    pub(crate) fn new(config_str: String, graph_filename: String) -> PyResult<RustAttackSimulator> {
        let graph = match load_graph_from_json(&graph_filename) {
            Ok(graph) => graph,
            Err(e) => return pyresult_with(Err(e), "Error in rust_sim"),
        };
        let num_attack_steps = graph.number_of_attacks();
        let num_defense_steps = graph.number_of_defenses();

        let config = SimulatorConfig::from_json(&config_str).unwrap();
        let runtime = match SimulatorRuntime::new(graph, config) {
            Ok(runtime) => runtime,
            Err(e) => return pyresult_with(Err(e), "Error in rust_sim"),
        };
        let config = runtime.config.clone();
        let num_actions = ACTIONS.len() as u64;
        Ok(RustAttackSimulator {
            defender_impact: runtime.defender_impact(),
            attacker_impact: runtime.attacker_impact(),
            num_attack_steps,
            num_defense_steps,
            runtime,
            config,
            num_actions,
            wait_action: ACTION_NOP,
            terminate_action: ACTION_TERMINATE,
        })
    }

    pub(crate) fn reset(&mut self, seed: Option<u64>) -> PyResult<(Observation, Info)> {
        pyresult(self.runtime.reset(seed))
    }

    pub(crate) fn step(
        &mut self,
        actions: HashMap<String, (usize, usize)>,
    ) -> PyResult<(Observation, Info)> {
        pyresult(self.runtime.step(actions))
    }
}
