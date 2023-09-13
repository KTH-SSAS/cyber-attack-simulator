pub mod attackgraph;
pub mod config;
pub mod graph;
mod loading;
pub mod observation;
pub mod runtime;
pub mod sim;
mod state;

use pyo3::prelude::*;

use observation::{Info, Observation};
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

    use crate::{
        attackgraph,
        config::SimulatorConfig,
        loading::{load_graph_from_json, load_graph_from_yaml},
        observation::{Info, Observation},
        runtime,
    };
    use rand::{seq::SliceRandom, SeedableRng};
    use rand_chacha::ChaChaRng;

    const TEST_FILENAME: &str = "mal/attackgraph.json";

    fn get_sim_from_filename(filename: &str) -> runtime::SimulatorRuntime<usize> {
        let graph = load_graph_from_json(filename).unwrap();
        let config = SimulatorConfig {
            seed: 0,
            false_negative_rate: 0.0,
            false_positive_rate: 0.0,
            randomize_ttc: false,
        };
        let sim = runtime::SimulatorRuntime::new(graph, config).unwrap();
        return sim;
    }

    fn random_step(node_mask: &Vec<bool>, rng: &mut ChaChaRng) -> Option<usize> {
        let actions = available_actions(node_mask);

        let action = match actions.choose(rng) {
            Some(x) => Some(*x),
            None => None,
        };

        return action;
    }

    fn available_actions(node_mask: &Vec<bool>) -> Vec<usize> {
        // Skip the special actions
        return node_mask
            .into_iter()
            .enumerate()
            .filter_map(|(x, &d)| match d {
                true => Some(x),
                false => None,
            })
            .collect::<Vec<usize>>();
    }

    #[test]
    fn test_attacker() {
        let filename = TEST_FILENAME;
        let mut sim = get_sim_from_filename(filename);

        let mut observation: Observation;
        let mut info: Info;

        (observation, info) = sim.reset(None).unwrap();

        let mut rng = ChaChaRng::seed_from_u64(0);
        let action = runtime::ACTION_USE;
        let mut time = info.time;
        let mut attack_surface = observation.attack_surface.clone();
        let mut available_steps = available_actions(&attack_surface);
        while available_steps.len() > 0 && time < sim.ttc_sum {
            let step = random_step(&attack_surface, &mut rng).unwrap();

            assert!(action != runtime::ACTION_TERMINATE); // We should never terminate, for now
            assert!(action != runtime::ACTION_NOP); // We should never NOP, for now

            let action_dict = HashMap::from([("attacker".to_string(), (action, step))]);
            let (new_observation, new_info) = sim.step(action_dict).unwrap();

            let graphviz = sim.to_graphviz();
            let file = std::fs::File::create(format!("test_attacker_{:0>2}.dot", time)).unwrap();
            std::io::Write::write_all(&mut std::io::BufWriter::new(file), graphviz.as_bytes())
                .unwrap();

            let a = new_observation.ttc_remaining[step];
            let b = observation.ttc_remaining[step];
            assert_eq!(a, max(b as i64 - 1, 0) as u64);
            let old_ttc_sum = observation.ttc_remaining.iter().sum::<u64>();
            let new_ttc_sum = new_observation.ttc_remaining.iter().sum::<u64>();
            assert_eq!(old_ttc_sum - 1, new_ttc_sum);

            observation = new_observation;
            info = new_info;
            time = info.time;
            attack_surface = observation.attack_surface.clone();
            available_steps = available_actions(&attack_surface);
        }

        assert_eq!(observation.attack_surface.iter().filter(|&x| *x).count(), 0);

        //sim.step(action_dict)
    }

    #[test]
    fn test_defender() {
        let filename = TEST_FILENAME;
        let mut sim = get_sim_from_filename(filename);
        let mut rng = ChaChaRng::seed_from_u64(0);
        let mut observation: Observation;
        let mut info: Info;
        let action = runtime::ACTION_USE;
        (observation, info) = sim.reset(None).unwrap();
        let mut defense_surface = observation.defense_surface.clone();
        let mut available_defenses = available_actions(&defense_surface);
        let mut time = info.time;
        while available_defenses.len() > 0 && time < sim.ttc_sum {
            let step = random_step(&defense_surface, &mut rng).unwrap();
            let action_dict = HashMap::from([("defender".to_string(), (action, step))]);
            (observation, info) = sim.step(action_dict).unwrap();
            defense_surface = observation.defense_surface.clone();
            available_defenses = available_actions(&defense_surface);
            time = info.time;
        }

        assert_eq!(observation.defense_surface.iter().filter(|&x| *x).count(), 0);
    }
}
