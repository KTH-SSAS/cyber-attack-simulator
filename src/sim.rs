use std::{fmt::Display, collections::HashMap};

use crate::{
    config::SimulatorConfig,
    runtime::{SimulatorRuntime, Observation, Info},
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
    runtime: SimulatorRuntime,
    #[pyo3(get)]
    ttc_total: u64,
    #[pyo3(get)]
    config: SimulatorConfig,
}

#[pymethods]
impl RustAttackSimulator {
    #[new]
    pub(crate) fn new(
        config_str: String,
        graph_filename: String,
    ) -> PyResult<RustAttackSimulator> {
        let graph = crate::attackgraph::load_graph_from_yaml(&graph_filename);
        let config = SimulatorConfig::from_json(&config_str).unwrap();
        let runtime = match SimulatorRuntime::new(graph, config) {
            Ok(runtime) => runtime,
            Err(e) => return pyresult_with(Err(e), "Error in rust_sim"), 
        };
        let ttc_total = runtime.total_ttc_remaining();
        let config = runtime.config.clone();
        Ok(RustAttackSimulator { runtime, ttc_total, config})
    }

    pub(crate) fn reset(&mut self, seed: Option<u64>) -> PyResult<(Observation, Info)> {
        pyresult(self.runtime.reset(seed))
    }

    pub(crate) fn step(&mut self, actions: HashMap<String, usize>) -> PyResult<(Observation, Info)> {
        pyresult(self.runtime.step(actions))
    }
}
