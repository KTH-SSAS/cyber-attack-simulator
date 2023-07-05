use std::{collections::HashMap, fmt::Display};

use crate::{
    config::SimulatorConfig,
    runtime::{SimulatorRuntime, ACTION_NOP, ACTION_TERMINATE, SPECIAL_ACTIONS}, observation::{Observation, Info},
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
    runtime: SimulatorRuntime<u64>,
    #[pyo3(get)]
    config: SimulatorConfig,
    #[pyo3(get)]
    num_special_actions: u64,
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
        let graph = crate::attackgraph::load_graph_from_yaml(&graph_filename);
        let config = SimulatorConfig::from_json(&config_str).unwrap();
        let runtime = match SimulatorRuntime::new(graph, config) {
            Ok(runtime) => runtime,
            Err(e) => return pyresult_with(Err(e), "Error in rust_sim"),
        };
        let config = runtime.config.clone();
        let num_special_actions = SPECIAL_ACTIONS.len() as u64;
        Ok(RustAttackSimulator {
            defender_impact: runtime.defender_impact(),
            attacker_impact: runtime.attacker_impact(),
            num_attack_steps: runtime.num_attacks(),
            num_defense_steps: runtime.num_defenses(),
            runtime,
            config,
            num_special_actions,
            wait_action: ACTION_NOP,
            terminate_action: ACTION_TERMINATE,
        })
    }

    pub(crate) fn reset(&mut self, seed: Option<u64>) -> PyResult<(Observation, Info)> {
        pyresult(self.runtime.reset(seed))
    }
    
    pub(crate) fn step(
        &mut self,
        actions: HashMap<String, usize>,
    ) -> PyResult<(Observation, Info)> {
        pyresult(self.runtime.step(actions))
    }
}
