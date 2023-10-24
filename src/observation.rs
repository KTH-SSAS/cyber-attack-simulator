use pyo3::pyclass;
use std::hash::{Hash, Hasher};

pub trait Observation<T> {
    fn state(&self) -> Vec<T>;
    fn possible_actions(&self) -> Vec<T>;
    fn possible_objects(&self) -> Vec<T>;
    fn is_terminal(&self) -> bool;
}

pub(crate) type StepInfo = (usize, usize, usize);
#[pyclass]
#[derive(Clone, Debug, Eq)]
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

impl Default for VectorizedObservation {
    fn default() -> Self {
        VectorizedObservation {
            attacker_possible_objects: vec![],
            defender_possible_objects: vec![],
            attacker_possible_actions: vec![],
            defender_possible_actions: vec![],
            defender_observation: vec![],
            attacker_observation: vec![],
            defender_reward: 0,
            attacker_reward: 0,
            state: vec![],
            step_info: vec![],
            ttc_remaining: vec![],
            edges: vec![],
            flags: vec![],
        }
    }
}

impl PartialEq for VectorizedObservation {
    fn eq(&self, other: &Self) -> bool {
        self.state == other.state
    }
}

impl Hash for VectorizedObservation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state.hash(state);
    }
}

impl serde::ser::Serialize for VectorizedObservation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        fn vec2string<T>(vec: &Vec<T>) -> String
        where
            T: ToString,
        {
            vec.iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(" ")
        }
        self.state.serialize(serializer)
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


// Tests
#[cfg(test)]
mod tests {
    use super::VectorizedObservation;


    #[test]
    fn test_serialize() {
        let obs = VectorizedObservation::default();
        let serialized = serde_json::to_string(&obs).unwrap();
        println!("{}", serialized)
    }
}