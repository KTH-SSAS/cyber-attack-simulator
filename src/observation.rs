use pyo3::pyclass;

pub trait Observation<T> {
    fn state(&self) -> Vec<T>;
    fn possible_actions(&self) -> Vec<T>;
    fn possible_objects(&self) -> Vec<T>;
    fn is_terminal(&self) -> bool;
}

pub(crate) type StepInfo = (usize, usize, usize);
#[pyclass]
#[derive(Clone, Debug, Hash, Eq)]
pub struct VectorizedObservation {
    #[pyo3(get)]
    pub attacker_possible_objects: Vec<bool>, // Attack surface of the attacker, all attackable nodes
    #[pyo3(get)]
    pub defender_possible_objects: Vec<bool>, // All defense steps that can be activated
    #[pyo3(get)]
    pub attacker_possible_actions: Vec<bool>, // All available actions for the attacker
    #[pyo3(get)]
    pub defender_possible_actions: Vec<bool>, // All available actions for the defender
    #[pyo3(get)]
    pub defender_observation: Vec<bool>,
    #[pyo3(get)]
    pub attacker_observation: Vec<bool>,
    #[pyo3(get)]
    pub defender_reward: i64,
    #[pyo3(get)]
    pub attacker_reward: i64,
    #[pyo3(get)]
    pub state: Vec<bool>,
    #[pyo3(get)]
    pub step_info: Vec<StepInfo>,
    #[pyo3(get)]
    pub ttc_remaining: Vec<u64>,
    #[pyo3(get)]
    pub edges: Vec<(usize, usize)>,
    #[pyo3(get)]
    pub flags: Vec<usize>,
}

impl PartialEq for VectorizedObservation {
    fn eq(&self, other: &Self) -> bool {
        self.state == other.state && self.ttc_remaining == other.ttc_remaining
    }
}

impl VectorizedObservation {
    pub(crate) fn is_terminal(&self) -> bool {
        self.attacker_possible_actions.iter().all(|&x| !x)
    }
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
    pub num_compromised_flags: usize,
    #[pyo3(get)]
    pub perc_compromised_flags: f64,
}
