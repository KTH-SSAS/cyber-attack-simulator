pub mod sim;
pub mod attackgraph;
pub mod runtime;
pub mod config;
pub mod graph;
pub mod observation;

use pyo3::prelude::*;

use observation::{Observation, Info};
use sim::RustAttackSimulator;


/// A Python module implemented in Rust.
#[pymodule]
fn rusty_sim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustAttackSimulator>()?;
    m.add_class::<Observation>()?;
    m.add_class::<Info>()?;
    Ok(())
}



#[cfg(test)]
mod tests {
    use std::{cmp::max, collections::HashMap};

    use crate::{attackgraph, runtime::{self, Observation, Info}, config};
    use rand::seq::SliceRandom;

    fn get_sim_from_filename(filename: &str) -> runtime::SimulatorRuntime {
        let graph = attackgraph::load_graph_from_yaml(filename);
        let config = config::SimulatorConfig::default();
        let sim = runtime::SimulatorRuntime::new(graph, config).unwrap();
        return sim;
    }

    fn random_action(action_mask: &Vec<bool>) -> usize {
        let actions = available_actions(action_mask.clone());
        let mut rng = rand::thread_rng();

        let action = match actions.choose(&mut rng) {
            Some(x) => *x,
            None => return runtime::ACTION_NOP,
        };
        return action + runtime::SPECIAL_ACTIONS.len();
    }

    fn available_actions(state: Vec<bool>) -> Vec<usize> {
        return state.iter().enumerate().filter_map(|(x, d)| match *d {
            true => Some(x),
            false => None,
        }).collect::<Vec<usize>>();
    }

    #[test]
    fn test_sim_obs() {
        let filename = "four_ways.yaml";
        let mut sim = get_sim_from_filename(filename);

        let observation: Observation;
        let _info: Info;

        (observation, _info) = sim.reset(None).unwrap();

        assert_eq!(observation.defense_state.iter().filter(|&x| *x).count(), 4);
        assert_eq!(observation.attack_state.iter().filter(|&x| *x).count(), 1);
        assert_eq!(observation.ttc_remaining.iter().filter(|&x| *x == 0).count(), 1);
        assert_eq!(observation.attack_surface.iter().filter(|&x| *x).count(), 4);
    }

    #[test]
    fn test_attacker() {
        let filename = "four_ways.yaml";
        let mut sim = get_sim_from_filename(filename);

        let mut observation: Observation;
        let mut info: Info;

        (observation, info) = sim.reset(None).unwrap();

        let mut time = info.time;
        while time < sim.ttc_sum {
            let action = random_action(&observation.attack_surface);
            let action_dict = HashMap::from([("attacker".to_string(), action)]);
            let (new_observation, new_info) = sim.step(action_dict).unwrap();

            if action != runtime::ACTION_NOP {
                let step_idx = action - runtime::SPECIAL_ACTIONS.len();
                assert_eq!(new_observation.ttc_remaining[step_idx], max(observation.ttc_remaining[step_idx].try_into().unwrap_or(0) - 1, 0) as u64);
                let old_ttc_sum = observation.ttc_remaining.iter().sum::<u64>();
                let new_ttc_sum = new_observation.ttc_remaining.iter().sum::<u64>();
                assert_eq!(old_ttc_sum - 1, new_ttc_sum);
            }
            
            observation = new_observation;
            info = new_info;
            time = info.time;
        }


        assert_eq!(observation.attack_surface.iter().filter(|&x| *x).count(), 0);
        assert_eq!(observation.attack_state.iter().all(|x| *x), true);

        //sim.step(action_dict)
    }

    #[test]
    fn test_defender() {
        let filename = "four_ways.yaml";
        let mut sim = get_sim_from_filename(filename);

        let mut observation: Observation;
        let mut info: Info;

        (observation, info) = sim.reset(None).unwrap();
        let defense_surface = available_actions(observation.defense_state.clone());

        let mut time = info.time;
        while defense_surface.len() > 0 && time < sim.ttc_sum {
            let action = random_action(&observation.defense_state.clone());
            let action_dict = HashMap::from([("defender".to_string(), action)]);
            (observation, info) = sim.step(action_dict).unwrap();     
            time = info.time;
        }

        assert_eq!(observation.defense_state.iter().filter(|&x| *x).count(), 0);
        let attack_state = observation.ttc_remaining.iter().map(|x| *x == 0).collect::<Vec<bool>>();
        assert_eq!(observation.attack_surface.iter().filter(|&x| *x).count(), 4);
        assert_eq!(attack_state.iter().filter(|b| **b).count(), 1);
    }
}