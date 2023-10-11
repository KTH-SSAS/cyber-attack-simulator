mod attackgraph;
pub mod config;
mod graph;
mod loading;
mod observation;
mod pysim;
mod runtime;
mod state;
mod defender_state;
mod attacker_state;

use config::SimulatorConfig;

use loading::load_graph_from_json;
use pyo3::prelude::*;

use observation::{Info, Observation};
use pysim::RustAttackSimulator;

use std::collections::HashMap;

use crate::runtime::SimulatorRuntime;

/// A Python module implemented in Rust.
#[pymodule]
fn rusty_sim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustAttackSimulator>()?;
    m.add_class::<Observation>()?;
    m.add_class::<Info>()?;
    Ok(())
}

#[derive(Debug)]
pub struct AttackSimError {
    pub msg: String,
}
pub type AttackSimResult<T> = Result<T, AttackSimError>;

pub struct AttackSimulator<T> {
    runtime: SimulatorRuntime<T>,
    pub config: SimulatorConfig,
    pub actions: HashMap<String, usize>,
    pub actors: HashMap<String, usize>,
    pub num_attack_steps: usize,
    pub num_defense_steps: usize,
    pub show_false: bool,
}

impl AttackSimulator<usize> {
    pub fn new(
        config: SimulatorConfig,
        graph_filename: String,
        vocab_filename: Option<&str>,
    ) -> AttackSimResult<AttackSimulator<usize>> {
        let graph = match load_graph_from_json(&graph_filename, vocab_filename) {
            Ok(graph) => graph,
            Err(e) => {
                return Err(AttackSimError {
                    msg: format!("Error in rust_sim: {}", e),
                })
            }
        };
        let num_attack_steps = graph.number_of_attacks();
        let num_defense_steps = graph.number_of_defenses();

        let runtime = match SimulatorRuntime::new(graph, config) {
            Ok(runtime) => runtime,
            Err(e) => {
                return Err(AttackSimError {
                    msg: format!("Error in rust_sim: {}", e),
                })
            }
        };
        let config = runtime.config.clone();
        Ok(AttackSimulator {
            num_attack_steps,
            num_defense_steps,
            show_false: config.show_false,
            config,
            actions: runtime.actions.clone(),
            actors: runtime.actors.clone(),
            runtime,
        })
    }

    pub fn reset(&mut self, seed: Option<u64>) -> AttackSimResult<(Observation, Info)> {
        match self.runtime.reset(seed) {
            Ok((obs, info)) => Ok((obs, info)),
            Err(e) => Err(AttackSimError {
                msg: format!("Error in rust_sim: {}", e),
            }),
        }
    }

    pub fn step(
        &mut self,
        actions: HashMap<String, (usize, Option<usize>)>,
    ) -> AttackSimResult<(Observation, Info)> {
        match self.runtime.step(actions) {
            Ok((obs, info)) => Ok((obs, info)),
            Err(e) => Err(AttackSimError {
                msg: format!("Error in rust_sim: {}", e),
            }),
        }
    }

    pub fn render(&self) -> String {
        self.runtime.to_graphviz(self.show_false)
    }
}

#[cfg(test)]
mod tests {
    use std::{cmp::max, collections::HashMap};

    use crate::{
        config::SimulatorConfig, loading::load_graph_from_json, observation::Observation, runtime,
    };
    use rand::{seq::SliceRandom, SeedableRng};
    use rand_chacha::ChaChaRng;

    const TEST_FILENAME: &str = "graphs/corelang.json";
    const TEST_VOCAB: Option<&str> = None; // Some("mal/corelang_vocab_merged.json");

    fn get_sim_from_filename(filename: &str) -> runtime::SimulatorRuntime<usize> {
        let graph = load_graph_from_json(filename, TEST_VOCAB).unwrap();
        let config = SimulatorConfig {
            seed: 0,
            false_negative_rate: 0.0,
            false_positive_rate: 0.0,
            randomize_ttc: false,
            log: false,
            show_false: false,
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

        (observation, _) = sim.reset(None).unwrap();

        let mut rng = ChaChaRng::seed_from_u64(0);
        let action = sim.actions["use"];
        let mut attack_surface = observation.attacker_possible_objects.clone();
        let mut available_steps = available_actions(&attack_surface);
        while available_steps.len() > 0 {
            let step = random_step(&attack_surface, &mut rng).unwrap();

            //assert!(action != sim.actions["terminate"]); // We should never terminate, for now
            assert!(action != sim.actions["wait"]); // We should never NOP, for now

            let action_dict = HashMap::from([("attacker".to_string(), (action, Some(step)))]);
            let (new_observation, _) = sim.step(action_dict).unwrap();

            //let graphviz = sim.to_graphviz();
            //let file = std::fs::File::create(format!("test_attacker_{:0>2}.dot", time)).unwrap();
            //std::io::Write::write_all(&mut std::io::BufWriter::new(file), graphviz.as_bytes())
            //    .unwrap();

            let a = new_observation.ttc_remaining[step];
            let b = observation.ttc_remaining[step];
            assert_eq!(a, max(b as i64 - 1, 0) as u64);
            let old_ttc_sum = observation.ttc_remaining.iter().sum::<u64>();
            let new_ttc_sum = new_observation.ttc_remaining.iter().sum::<u64>();
            if observation.state == new_observation.state {
                assert_eq!(old_ttc_sum - 1, new_ttc_sum);
            } else {
                assert_eq!(old_ttc_sum, new_ttc_sum);
            }

            observation = new_observation;
            attack_surface = observation.attacker_possible_objects.clone();
            available_steps = available_actions(&attack_surface);
        }

        assert_eq!(
            observation
                .attacker_possible_objects
                .iter()
                .filter(|&x| *x)
                .count(),
            0
        );

        //sim.step(action_dict)
    }

    #[test]
    fn test_defender() {
        let filename = TEST_FILENAME;
        let mut sim = get_sim_from_filename(filename);
        let mut rng = ChaChaRng::seed_from_u64(0);
        let mut observation: Observation;
        let action = sim.actions["use"];
        (observation, _) = sim.reset(None).unwrap();
        let mut defense_surface = observation.defender_possible_objects.clone();
        let num_entrypoints = observation.state.iter().filter(|&x| *x).count();
        let num_defense = defense_surface.iter().filter(|&x| *x).count();
        let mut available_defenses = available_actions(&defense_surface);
        while available_defenses.len() > 0 {
            let step = random_step(&defense_surface, &mut rng).unwrap();
            let action_dict = HashMap::from([("defender".to_string(), (action, Some(step)))]);
            (observation, _) = sim.step(action_dict).unwrap();
            defense_surface = observation.defender_possible_objects.clone();
            available_defenses = available_actions(&defense_surface);
        }

        assert_eq!(
            observation
                .defender_possible_objects
                .iter()
                .filter(|&x| *x)
                .count(),
            0
        );
        assert_eq!(
            observation.state.iter().filter(|&x| *x).count(),
            num_defense + num_entrypoints
        );
    }
}
