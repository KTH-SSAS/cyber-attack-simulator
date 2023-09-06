pub mod sim;
pub mod attackgraph;
pub mod runtime;
pub mod config;
pub mod graph;
pub mod observation;
mod loading;

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
    use std::{cmp::max, collections::HashMap, io::Write};

    use crate::{attackgraph, config::SimulatorConfig, runtime, observation::{Observation, Info}};
    use rand::seq::SliceRandom;

    fn get_sim_from_filename(filename: &str) -> runtime::SimulatorRuntime<u64> {
        let graph = attackgraph::load_graph_from_yaml(filename);
        let config = SimulatorConfig {
            seed: 0,
            false_negative_rate: 0.0,
            false_positive_rate: 0.0,
            randomize_ttc: false,
        };
        let sim = runtime::SimulatorRuntime::new(graph, config).unwrap();
        return sim;
    }

    fn random_action(action_mask: &Vec<bool>) -> usize {
        let actions = available_actions(action_mask);
        let mut rng = rand::thread_rng();

        let action = match actions.choose(&mut rng) {
            Some(x) => *x,
            None => return runtime::ACTION_NOP,
        };
        return action;
    }

    fn available_actions(state: &Vec<bool>) -> Vec<usize> {
        // Skip the special actions
        return state.iter().enumerate().skip(runtime::SPECIAL_ACTIONS.len()).filter_map(|(x, d)| match *d {
            true => Some(x),
            false => None,
        }).collect::<Vec<usize>>();
    }



    #[test]
    fn test_attacker() {
        let filename = "graphs/four_ways.yaml";
        let mut sim = get_sim_from_filename(filename);

        let mut observation: Observation;
        let mut info: Info;

        (observation, info) = sim.reset(None).unwrap();

        let mut time = info.time;
        while time < sim.ttc_sum {
            let action = random_action(&observation.attacker_action_mask);

            assert!(action != runtime::ACTION_TERMINATE); // We should never terminate, for now
            assert!(action != runtime::ACTION_NOP); // We should never NOP, for now

            let action_dict = HashMap::from([("attacker".to_string(), action)]);
            let (new_observation, new_info) = sim.step(action_dict).unwrap();

            let graphviz = sim.to_graphviz();
            let file = std::fs::File::create(format!("test_attacker_{}.dot", time)).unwrap();
            std::io::Write::write_all(&mut std::io::BufWriter::new(file), graphviz.as_bytes()).unwrap();

            let step_idx = action - runtime::SPECIAL_ACTIONS.len();
            assert_eq!(new_observation.ttc_remaining[step_idx], max(observation.ttc_remaining[step_idx] as i64 - 1, 0) as u64);
            let old_ttc_sum = observation.ttc_remaining.iter().sum::<u64>();
            let new_ttc_sum = new_observation.ttc_remaining.iter().sum::<u64>();
            assert_eq!(old_ttc_sum - 1, new_ttc_sum);
            
            
            observation = new_observation;
            info = new_info;
            time = info.time;
        }


        assert_eq!(observation.attack_surface.iter().filter(|&x| *x).count(), 0);
        assert_eq!(observation.state.iter().all(|x| *x), true);

        //sim.step(action_dict)
    }

    #[test]
    fn test_defender() {
        let filename = "graphs/four_ways.yaml";
        let mut sim = get_sim_from_filename(filename);

        let mut observation: Observation;
        let mut info: Info;

        (observation, info) = sim.reset(None).unwrap();
        let defense_surface = available_actions(&observation.defender_action_mask);

        let mut time = info.time;
        while defense_surface.len() > 0 && time < sim.ttc_sum {
            let action = random_action(&observation.defender_action_mask);
            let action_dict = HashMap::from([("defender".to_string(), action)]);
            (observation, info) = sim.step(action_dict).unwrap();     
            time = info.time;
        }

        assert_eq!(observation.defender_action_mask.iter().filter(|&x| *x).count(), 1);
        assert_eq!(observation.attack_surface.iter().filter(|&x| *x).count(), 4);
    }
}