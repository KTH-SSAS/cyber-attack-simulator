use pyo3::pyclass;

#[pyclass]
pub(crate) struct Observation {
    #[pyo3(get)]
    pub attack_surface: Vec<bool>,
    #[pyo3(get)]
    pub defense_surface: Vec<bool>,
    #[pyo3(get)]
    pub attacker_action_mask: Vec<bool>,
    #[pyo3(get)]
    pub defender_action_mask: Vec<bool>,
    #[pyo3(get)]
    pub defense_state: Vec<bool>,
    #[pyo3(get)]
    pub attack_state: Vec<bool>,
    #[pyo3(get)]
    pub ids_observation: Vec<bool>,
    #[pyo3(get)]
    pub ttc_remaining: Vec<u64>,
}

#[pyclass]
pub(crate) struct Info {
    #[pyo3(get)]
    pub time: u64,
    #[pyo3(get)]
    pub num_compromised_steps: usize,
    #[pyo3(get)]
    pub perc_compromised_steps: f64,
    #[pyo3(get)]
    pub perc_defenses_activated: f64,
    #[pyo3(get)]
    pub num_observed_alerts: usize,
    #[pyo3(get)]
    pub num_compromised_flags: usize,
    #[pyo3(get)]
    pub perc_compromised_flags: f64,
}