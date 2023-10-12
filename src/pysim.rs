use std::{collections::HashMap, fmt::Display};

use crate::{
    config::SimulatorConfig,
    loading::load_graph_from_json,
    observation::{Info, Observation},
    runtime::SimulatorRuntime,
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
    actions: HashMap<String, usize>,
    #[pyo3(get)]
    actors: HashMap<String, usize>,
    #[pyo3(get)]
    vocab: HashMap<String, usize>,
    show_false: bool,
}

#[pymethods]
impl RustAttackSimulator {
    #[new]
    pub(crate) fn new(
        config_str: String,
        graph_filename: String,
        vocab_filename: Option<&str>,
    ) -> PyResult<RustAttackSimulator> {
        let config = SimulatorConfig::from_json(&config_str).unwrap();
        let graph = match load_graph_from_json(
            &graph_filename,
            vocab_filename,
            config.false_negative_rate,
            config.false_positive_rate,
        ) {
            Ok(graph) => graph,
            Err(e) => return pyresult_with(Err(e), "Error in rust_sim"),
        };

        let runtime = match SimulatorRuntime::new(graph, config) {
            Ok(runtime) => runtime,
            Err(e) => return pyresult_with(Err(e), "Error in rust_sim"),
        };
        let config = runtime.config.clone();
        Ok(RustAttackSimulator {
            show_false: config.show_false,
            config,
            actions: runtime.actions.clone(),
            actors: runtime.actors.clone(),
            vocab: runtime.vocab(),
            runtime,
        })
    }

    pub(crate) fn reset(&mut self, seed: Option<u64>) -> PyResult<(Observation, Info)> {
        pyresult(self.runtime.reset(seed))
    }

    pub(crate) fn step(
        &mut self,
        actions: HashMap<String, (usize, Option<usize>)>,
    ) -> PyResult<(Observation, Info)> {
        pyresult(self.runtime.step(actions))
    }

    pub fn render(&self) -> String {
        self.runtime.to_graphviz(self.show_false)
    }
}
