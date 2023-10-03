use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::{fmt, vec};

use itertools::Itertools;
use log::LevelFilter;
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Root};
use log4rs::encode::pattern::PatternEncoder;
use log4rs::Config;

use crate::attackgraph::{AttackGraph, TTCType};
use crate::config::SimulatorConfig;
use crate::observation::{Info, Observation, StateTuple};
use crate::state::SimulatorState;

use rand::Rng;

pub type SimResult<T> = std::result::Result<T, SimError>;
#[derive(Debug, Clone)]
pub struct SimError {
    pub error: String,
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SimError: {}", self.error)
    }
}

pub struct ActionResult<I> {
    pub ttc_diff: HashMap<I, i32>,
    pub enabled_defenses: HashSet<I>,
    pub valid_action: bool,
}

impl<I> Default for ActionResult<I> {
    fn default() -> Self {
        ActionResult {
            ttc_diff: HashMap::new(),
            enabled_defenses: HashSet::new(),
            valid_action: false,
        }
    }
}

//type ActionIndex = usize;
// type ActorIndex = usize;

type ParameterAction = (usize, usize);

pub(crate) struct SimulatorRuntime<I> {
    g: AttackGraph<I>,
    state: RefCell<SimulatorState<I>>,
    history: Vec<SimulatorState<I>>,
    pub config: SimulatorConfig,
    pub confusion_per_step: HashMap<I, (f64, f64)>,
    pub ttc_sum: TTCType,

    pub actions: HashMap<String, usize>,
    pub actors: HashMap<String, usize>,

    pub id_to_index: HashMap<I, usize>,
    pub index_to_id: Vec<I>,
    //pub defender_action_to_graph: Vec<I>,
    //pub attacker_action_to_graph: Vec<I>,
}

// def get_initial_attack_surface(self, attack_start_time: int) -> NDArray:
// attack_surface = np.zeros(self.g.num_attacks, dtype="int8")
// if attack_start_time == 0:
//     attack_surface[self.entry_attack_index] = 0
//     self.attack_state[self.entry_attack_index] = 1
//     # add reachable steps to the attack surface
//     attack_surface[self._get_vulnerable_children(self.entry_attack_index)] = 1
// else:
//     attack_surface[self.entry_attack_index] = 1

// return attack_surface

impl<I> SimulatorRuntime<I>
where
    I: Eq + Hash + Ord + Display + Copy + Debug,
{
    pub fn vocab(&self) -> HashMap<String, usize> {
        return self.g.vocab.clone();
    }

    // Path: src/sim.rs
    pub fn new(graph: AttackGraph<I>, config: SimulatorConfig) -> SimResult<SimulatorRuntime<I>> {
        if config.log {
            // clear log

            let _ = std::fs::remove_file("log/output.log");

            let logfile = FileAppender::builder()
                .encoder(Box::new(PatternEncoder::new("{l} - {m}\n")))
                .build("log/output.log")
                .unwrap();

            let log_config = Config::builder()
                .appender(Appender::builder().build("logfile", Box::new(logfile)))
                .build(Root::builder().appender("logfile").build(LevelFilter::Info))
                .unwrap();

            log4rs::init_config(log_config).unwrap();
        }

        log::info!("Simulator initiated.");

        let index_to_id = graph
            .nodes()
            .iter()
            .map(|(&x, _)| x)
            .sorted() // ensure deterministic order
            .collect::<Vec<I>>();

        // Maps the id of a node in the graph to an index in the state vector
        let id_to_index: HashMap<I, usize> = index_to_id
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        // Maps the index of the action to the id of the node in the graph

        /*

        let defender_action_to_graph = index_to_id
            .iter()
            .filter_map(|id| match graph.has_defense(id) {
                true => Some(*id),
                false => None,
            })
            .collect::<Vec<I>>();

        let attacker_action_to_graph = index_to_id.iter().filter_map(|id| {
                match graph.has_attack(id) {
                    true => Some(*id),
                    false => None,
                }
            }).collect::<Vec<I>>();

        */

        let initial_state = SimulatorState::new(&graph, config.seed, config.randomize_ttc)?;

        let fnr_fpr_per_step = index_to_id
            .iter()
            .map(|id| {
                match graph.has_attack(&id) {
                    true => (id, (config.false_negative_rate, config.false_positive_rate)),
                    false => (id, (0.0, 0.0)), // No false negatives or positives for defense steps
                }
            })
            .map(|(id, (fnr, fpr))| (id.clone(), (fnr, fpr)))
            .collect::<HashMap<I, (f64, f64)>>();

        let attacker_string = "attacker".to_string();
        let defender_string = "defender".to_string();
        let roles = [attacker_string, defender_string];
        let actions = ["wait", "use"];

        let actors = HashMap::from_iter(roles.into_iter().enumerate().map(|(i, x)| (x, i)));
        let actions =
            HashMap::from_iter(actions.iter().enumerate().map(|(i, &x)| (x.to_string(), i)));

        let sim = SimulatorRuntime {
            state: RefCell::new(initial_state),
            g: graph,
            confusion_per_step: fnr_fpr_per_step,
            config,
            ttc_sum: 0,
            id_to_index,
            index_to_id,
            history: Vec::new(),
            actions,
            actors,
        };

        return Ok(sim);
    }

    #[allow(dead_code)]
    pub fn translate_node_vec(&self, node_vec: &Vec<I>) -> Vec<String> {
        return node_vec
            .iter()
            .enumerate()
            .map(|(i, _)| self.index_to_id[i])
            .map(|id| self.g.name_of_step(&id))
            .collect();
    }

    #[allow(dead_code)]
    pub fn translate_index_vec(&self, index_vec: &Vec<usize>) -> Vec<String> {
        return index_vec
            .iter()
            .map(|&i| self.index_to_id[i])
            .map(|id| self.g.name_of_step(&id))
            .collect();
    }

    fn get_color(&self, id: &I, state: &SimulatorState<I>) -> String {
        match id {
            id if self.g.entry_points().contains(id) => "crimson".to_string(), // entry points
            id if state.defense_surface.contains(id) => "chartreuse4".to_string(), // disabled defenses
            id if state.enabled_defenses.contains(id) => "chartreuse".to_string(), // enabled defenses
            id if state.attack_surface.contains(id) => "gold".to_string(),         // attack surface
            id if state.compromised_steps.contains(id) => "firebrick1".to_string(), // compromised steps
            id if self.g.flags.contains(id) => "darkmagenta".to_string(),           // flags
            _ => "white".to_string(),
        }
    }

    pub fn to_graphviz(&self) -> String {
        let attributes = self
            .g
            .nodes()
            .iter()
            .map(|(id, node)| {
                let mut attrs = Vec::new();
                attrs.push(("style".to_string(), "filled".to_string()));
                attrs.push((
                    "fillcolor".to_string(),
                    self.get_color(id, &self.state.borrow()),
                ));
                // Add TTC
                attrs.push((
                    "label".to_string(),
                    format!(
                        "{}\\nTTC={}",
                        node.data,
                        self.state.borrow().remaining_ttc[id]
                    ),
                ));
                (id, attrs)
            })
            .collect();

        return self.g.to_graphviz(Some(&attributes));
    }

    pub fn reset(&mut self, seed: Option<u64>) -> SimResult<(Observation, Info)> {
        if let Some(seed) = seed {
            self.config.seed = seed;
        }

        let new_state = SimulatorState::new(&self.g, self.config.seed, self.config.randomize_ttc)?;

        log::info!("Resetting simulator with seed {}", self.config.seed);
        log::info!("Initial state:\n {:?}", new_state);

        let flag_status = self.g.get_flag_status(&new_state.compromised_steps);

        let result = Ok((
            self.map_state_to_observation(&new_state, (0, 0)),
            new_state.to_info(
                self.g.number_of_attacks(),
                self.g.number_of_defenses(),
                flag_status,
            ),
        ));
        self.ttc_sum = new_state.total_ttc_remaining();
        self.history.clear();
        self.state.replace(new_state);
        return result;
    }

    fn map_state_to_observation(
        &self,
        state: &SimulatorState<I>,
        rewards: (i64, i64),
    ) -> Observation {
        // reverse graph id to action index mapping

        let mut attack_surface_vec = vec![false; self.id_to_index.len()];
        state
            .attack_surface
            .iter()
            .map(|node_id| self.id_to_index[&node_id])
            .for_each(|index| {
                attack_surface_vec[index] = true;
            });

        let mut ttc_remaining = vec![0; self.id_to_index.len()];
        state
            .remaining_ttc
            .iter()
            .map(|(node_id, &ttc)| (self.id_to_index[&node_id], ttc))
            .for_each(|(index, ttc)| {
                ttc_remaining[index] = ttc;
            });

        /*
        let disabled_defenses = self.g.disabled_defenses(&state.enabled_defenses);

        disabled_defenses
            .iter()
            .map(|&node_id| self.id_to_index[&node_id])
            .for_each(|index| {
                step_state[index] = true;
            });
        */

        let step_state = self
            .index_to_id
            .iter()
            .map(|id| {
                let step = self.g.get_step(id).unwrap();
                let enabled =
                    state.enabled_defenses.contains(id) || state.compromised_steps.contains(id);
                let state_tuple = step.to_state_tuple(enabled);
                state_tuple
            })
            .map(|x| (x.0, self.g.word2idx(x.1), x.2, self.g.word2idx(x.3)))
            .collect::<Vec<StateTuple>>();

        let ids_observed = state.get_ids_obs();

        let mut ids_observed_vec = vec![false; self.id_to_index.len()];
        ids_observed.iter().for_each(|node_id| {
            ids_observed_vec[self.id_to_index[node_id]] = true;
        });

        state.enabled_defenses.iter().for_each(|&node_id| {
            ids_observed_vec[self.id_to_index[&node_id]] = true;
        });

        //let mut action_mask = vec![false; SPECIAL_ACTIONS.len()];
        //action_mask[ACTION_NOP] = true;
        //action_mask[ACTION_TERMINATE] = false;

        let mut defense_surface_vec = vec![false; self.id_to_index.len()];
        state
            .defense_surface
            .iter()
            .map(|node_id| self.id_to_index[&node_id])
            .for_each(|index| {
                defense_surface_vec[index] = true;
            });

        /*
        let defender_action_mask = action_mask
            .iter()
            .chain(defense_surface.iter())
            .cloned()
            .collect::<Vec<bool>>();

        let attacker_action_mask = action_mask
            .iter()
            .chain(attack_surface_vec.iter())
            .cloned()
            .collect::<Vec<bool>>();

        */

        let defender_action_mask = vec![
            true,                                                        // wait
            self.g.disabled_defenses(&state.enabled_defenses).len() > 0, // can use as long as there are disabled defenses
        ];
        let attacker_action_mask = vec![false, state.attack_surface.len() > 0];

        let edges = &self.g.edges();

        // map edge indices to the vector indices
        let vector_edges = edges
            .iter()
            .map(|(from, to)| (self.id_to_index[from], self.id_to_index[to]))
            .collect::<Vec<(usize, usize)>>();

        Observation {
            attack_surface: attack_surface_vec,
            defense_surface: defense_surface_vec,
            defender_action_mask,
            attacker_action_mask,
            ttc_remaining,
            ids_observation: ids_observed_vec,
            nodes: step_state,
            edges: vector_edges,
            //defense_indices: self.defender_action_to_state(),
            flags: self.g.flag_to_index(&self.id_to_index),
            attacker_reward: rewards.0,
            defender_reward: rewards.1,
        }
    }

    pub fn step(
        &mut self,
        action_dict: HashMap<String, ParameterAction>,
    ) -> SimResult<(Observation, Info)> {
        log::info!("Step with action dict {:?}", action_dict);

        let (new_state, rewards) = self.calculate_next_state(action_dict)?;

        log::info!("New state:\n{:?}", new_state);

        let flag_status = self.g.get_flag_status(&new_state.compromised_steps);

        let result = Ok((
            self.map_state_to_observation(&new_state, rewards),
            new_state.to_info(
                self.g.number_of_attacks(),
                self.g.number_of_defenses(),
                flag_status,
            ),
        ));

        self.history.push(self.state.replace(new_state));

        return result;
    }

    fn wait_idx(&self) -> usize {
        return self.actions["wait"];
    }

    /*
    fn terminate_idx(&self) -> usize {
        return self.actions["terminate"];
    }
    */

    fn calculate_next_state(
        &self,
        action_dict: HashMap<String, ParameterAction>,
    ) -> SimResult<(SimulatorState<I>, (i64, i64))> {
        let old_state = self.state.borrow();

        // Attacker selects and attack step from the attack surface
        // Defender selects a defense step from the defense surface, which is
        // the vector of all defense steps that are not disabled

        let defender_action = match action_dict.get("defender") {
            Some(action) => (action.0, self.index_to_id.get(action.1)),
            None => (self.wait_idx(), None),
        };

        let attacker_action = match action_dict.get("attacker") {
            Some(action) => (action.0, self.index_to_id.get(action.1)),
            None => (self.wait_idx(), None),
        };

        let defender_reward = old_state.defender_reward(&self.g, defender_action.1);

        let attacker_reward = old_state.attacker_reward(&self.g);

        let mut rng = old_state.export_rng();

        let missed_alerts = old_state
            .compromised_steps
            .iter()
            .filter_map(
                |id| match rng.gen::<f64>() < self.confusion_per_step[id].0 {
                    true => Some(*id), // if p < fnr, then we missed an alert
                    false => None,
                },
            )
            .collect::<HashSet<I>>();

        let false_alerts = old_state
            .uncompromised_steps(&self.g)
            .iter()
            .filter_map(
                |id| match rng.gen::<f64>() < self.confusion_per_step[id].1 {
                    true => Some(*id), // if p < fpr, then we got a false alert
                    false => None,
                },
            )
            .collect::<HashSet<I>>();

        Ok((
            SimulatorState {
                attack_surface: old_state.attack_surface(
                    &self.g,
                    defender_action.1,
                    attacker_action.1,
                ),
                defense_surface: old_state.defense_surface(&self.g, defender_action.1),
                enabled_defenses: old_state.enabled_defenses(&self.g, defender_action.1),
                remaining_ttc: old_state.remaining_ttc(
                    &self.g,
                    attacker_action.1,
                    defender_action.1,
                ),
                compromised_steps: old_state.compromised_steps(
                    &self.g,
                    attacker_action.1,
                    defender_action.1,
                ),
                time: old_state.time + 1,
                rng,
                num_observed_alerts: 0,
                //actions: action_dict,
                missed_alerts,
                false_alerts,
            },
            (attacker_reward, defender_reward),
        ))
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        config,
        loading::load_graph_from_json,
        observation::{Info, Observation},
        runtime::SimulatorRuntime,
    };

    #[test]
    fn test_sim_init() {
        let filename = "mal/attackgraph.json";
        let graph = load_graph_from_json(filename, None).unwrap();
        //let num_defenses = graph.number_of_defenses();
        let num_entrypoints = graph.entry_points().len();
        let config = config::SimulatorConfig::default();
        let sim = SimulatorRuntime::new(graph, config).unwrap();

        let initial_state = sim.state.borrow();

        assert_eq!(initial_state.enabled_defenses.len(), 0);
        assert_eq!(initial_state.compromised_steps.len(), num_entrypoints);
        assert_eq!(initial_state.remaining_ttc.len(), sim.g.nodes().len());
        /*
        assert_eq!(
            sim.defender_action_to_graph.len(),
            sim.g.number_of_defenses()
        );
        */
        //assert_eq!(sim.attacker_action_to_graph.len(), sim.g.nodes().len());
    }

    #[test]
    fn test_sim_obs() {
        let filename = "mal/attackgraph.json";
        let graph = load_graph_from_json(filename, None).unwrap();
        //let num_attacks = graph.number_of_attacks();
        let num_defenses = graph.number_of_defenses();
        let num_entrypoints = graph.entry_points().len();
        let config = config::SimulatorConfig::default();
        let mut sim = SimulatorRuntime::new(graph, config).unwrap();

        let observation: Observation;
        let _info: Info;

        (observation, _info) = sim.reset(None).unwrap();

        assert_eq!(
            observation.defense_surface.iter().filter(|&x| *x).count(),
            num_defenses
        );

        //println!("AS: {:?}", observation.attack_surface);

        let _steps = observation
            .attack_surface
            .iter()
            .enumerate()
            .filter_map(|(i, x)| match x {
                true => Some(i),
                false => None,
            })
            .collect::<Vec<usize>>();
        //let strings = sim.translate_index_vec(&steps);
        //println!("AS: {:?}", strings);

        assert_eq!(observation.attacker_action_mask.len(), sim.actions.len());

        assert_eq!(
            observation.defense_surface.iter().filter(|&x| *x).count(),
            num_defenses
        );

        assert_eq!(observation.nodes.len(), sim.g.nodes().len());
        assert_eq!(
            observation.nodes.iter().filter(|&(x, _, _, _)| *x).count(),
            num_entrypoints
        ); // 4 available defenses + 1 compromised attack step

        //check that all defense steps are disabled

        let defense_indices = observation
            .defense_surface
            .iter()
            .enumerate()
            .filter_map(|(i, x)| match x {
                true => Some(i),
                false => None,
            })
            .collect::<Vec<usize>>();

        for i in defense_indices.iter() {
            assert_eq!(observation.nodes[*i].0, false); // Defense steps should be disabled
        }

        //assert!(observation.ttc_remaining.iter().sum::<u64>() > 0);

        let edges = observation.edges;
        let entrypoints = sim.g.entry_points();
        let entrypoint_index = sim
            .id_to_index
            .iter()
            .filter_map(|(id, _)| match entrypoints.get(&id) {
                Some(i) => Some(i),
                None => None,
            })
            .collect::<Vec<&usize>>();

        for index in entrypoint_index {
            let indices_from_entrypoint = edges
                .iter()
                .filter_map(|(from, to)| match from == index {
                    true => Some(to),
                    false => None,
                })
                .collect::<Vec<&usize>>();

            //assert_eq!(indices_from_entrypoint.len(), graph.children(&sim.index_to_id[*index]).len());

            // All the outgoing edges should be in the attack surface
            for index in indices_from_entrypoint {
                assert_eq!(observation.attack_surface[*index], true);
            }
        }
    }
}
