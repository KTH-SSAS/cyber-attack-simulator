use pyo3::pyclass;
use serde::{Deserialize, Serialize};
use serde_json::Error;

#[derive(Serialize, Deserialize, Clone)]
#[pyclass]
pub struct SimulatorConfig {
    #[pyo3(get)]
    pub seed: Option<u64>,
    #[pyo3(get)]
    pub false_negative_rate: f64,
    #[pyo3(get)]
    pub false_positive_rate: f64,
    #[pyo3(get)]
    pub randomize_ttc: bool,
    #[pyo3(get)]
    pub log: bool,
    #[pyo3(get)]
    pub show_false: bool,
}

impl SimulatorConfig {
    pub fn from_json(json: &str) -> Result<SimulatorConfig, Error> {
        return serde_json::from_str(json);
    }

    #[allow(dead_code)]
    pub fn to_json(&self) -> Result<String, Error> {
        return serde_json::to_string_pretty(self);
    }
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        SimulatorConfig {
            seed: None,
            false_negative_rate: 0.0,
            false_positive_rate: 0.0,
            randomize_ttc: false,
            log: false,
            show_false: false,
        }
    }
}
