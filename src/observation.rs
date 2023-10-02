use pyo3::pyclass;

pub(crate) type StateTuple = (bool, usize, usize, usize);
#[pyclass]
pub struct Observation {
    #[pyo3(get)]
    pub attack_surface: Vec<bool>, // Attack surface of the attacker, all attackable nodes
    #[pyo3(get)]
    pub defense_surface: Vec<bool>, // All defense steps that can be activated
    #[pyo3(get)]
    pub attacker_action_mask: Vec<bool>, // All available actions for the attacker
    #[pyo3(get)]
    pub defender_action_mask: Vec<bool>, // All available actions for the defender
    #[pyo3(get)]
    pub nodes: Vec<StateTuple>,
    #[pyo3(get)]
    pub ids_observation: Vec<bool>,
    #[pyo3(get)]
    pub ttc_remaining: Vec<u64>,
    #[pyo3(get)]
    pub edges: Vec<(usize, usize)>,
    //#[pyo3(get)]
    //pub defense_indices: Vec<usize>,
    #[pyo3(get)]
    pub flags: Vec<usize>,
    #[pyo3(get)]
    pub defender_reward: i64,
    #[pyo3(get)]
    pub attacker_reward: i64,
}

#[pyclass]
pub struct Info {
    #[pyo3(get)]
    pub time: u64,
    #[pyo3(get)]
    pub sum_ttc: u64,
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
