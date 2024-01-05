use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::{fmt, vec};

use itertools::Itertools;
use log::LevelFilter;
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Root};
use log4rs::encode::pattern::PatternEncoder;
use log4rs::Config;

use crate::attacker_state::AttackerObs;
use crate::attackgraph::AttackGraph;
use crate::config::SimulatorConfig;
use crate::defender_state::DefenderObs;
use crate::observation::{Info, StepInfo, VectorizedObservation};
use crate::state::{SimulatorObs, SimulatorState};

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

type ParameterAction = (usize, Option<usize>);

pub(crate) struct SimulatorRuntime<I>
where
    I: Ord + Hash,
{
    g: AttackGraph<I>,
    state: RefCell<SimulatorState<I>>,
    history: RefCell<Vec<SimulatorState<I>>>,
    pub config: SimulatorConfig,
    pub actors: HashMap<String, usize>,

    pub action2idx: HashMap<String, usize>,
    pub idx2action: Vec<String>,
    pub id_to_index: HashMap<I, usize>,
    pub index_to_id: Vec<I>,
    //state_cache : RefCell<HashMap<CacheIndex, SimulatorState<I>>>,
    //pub defender_action_to_graph: Vec<I>,
    //pub attacker_action_to_graph: Vec<I>,
}

/*
#[derive(Eq)]
struct CacheIndex((VectorizedObservation, ParameterAction, ParameterAction));

impl PartialEq for CacheIndex {
    fn eq(&self, other: &Self) -> bool {
        return self.0.0 == other.0.0 && self.0.1 == other.0.1 && self.0.2 == other.0.2;
    }
}

impl Hash for CacheIndex {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.0.hash(state);
        self.0.1.hash(state);
        self.0.2.hash(state);
    }
}
*/

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
    I: Eq + Hash + Ord + Debug + Copy + Debug,
{
    pub fn vocab(&self) -> HashMap<String, usize> {
        return self.g.vocab.export();
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

        let initial_state = SimulatorState::new(&graph, None, config.randomize_ttc)?;

        let attacker_string = "attacker".to_string();
        let defender_string = "defender".to_string();
        let roles = vec![attacker_string, defender_string];
        let actors = HashMap::from_iter(roles.into_iter().enumerate().map(|(i, x)| (x, i)));

        let use_actions_from_graph = false;
        let idx2action: Vec<String>;
        let action2idx: HashMap<String, usize>;

        if use_actions_from_graph {
            let unique_actions = graph.distinct_actions();
            idx2action = unique_actions
                .iter()
                .sorted()
                .cloned()
                .collect::<Vec<String>>();
            action2idx = idx2action
                .iter()
                .enumerate()
                .map(|(i, x)| (x.clone(), i))
                .collect();
        } else {
            idx2action = vec!["wait".to_string(), "use".to_string()];
            action2idx =
                HashMap::from_iter(idx2action.iter().enumerate().map(|(i, x)| (x.clone(), i)));
        }

        let sim = SimulatorRuntime {
            state: RefCell::new(initial_state),
            g: graph,
            config,
            id_to_index,
            index_to_id,
            history: RefCell::new(Vec::new()),
            //state_cache: RefCell::new(HashMap::new()),
            action2idx,
            idx2action,
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

    fn get_color(&self, id: &I, state: &SimulatorState<I>, show_false: bool) -> String {
        let defender_obs = DefenderObs::new(state, &self.g);
        let attacker_obs = AttackerObs::new(state, &self.g);
        let steps_observed_as_compromised = defender_obs.steps_observed_as_compromised(&self.g);
        let false_negatives: HashSet<&I> = state
            .compromised_steps
            .difference(&steps_observed_as_compromised)
            .collect();
        let false_positives: HashSet<&I> = steps_observed_as_compromised
            .difference(&state.compromised_steps)
            .collect();
        match id {
            id if false_negatives.contains(id) && show_false => "deeppink".to_string(), // show false negatives for debugging
            id if false_negatives.contains(id) && !show_false => "white".to_string(), // false negatives hide true positives
            id if false_positives.contains(id) && show_false => "darkorchid1".to_string(), // show false positives for debugging
            id if false_positives.contains(id) && !show_false => "firebrick1".to_string(), // observered steps
            id if self.g.entry_points().contains(id) => "crimson".to_string(), // entry points
            id if defender_obs.possible_objects.contains(id) => "chartreuse4".to_string(), // disabled defenses
            id if state.enabled_defenses.contains(id) => "chartreuse".to_string(), // enabled defenses
            id if attacker_obs.possible_objects.contains(id) => "gold".to_string(), // attack surface
            id if state.compromised_steps.contains(id) => "firebrick1".to_string(), // compromised steps
            id if self.g.flags.contains(id) => "darkmagenta".to_string(),           // flags
            _ => "white".to_string(),
        }
    }

    pub fn to_graphviz(&self, show_false: bool) -> String {
        let attributes = self
            .g
            .nodes()
            .iter()
            .map(|(id, node)| {
                let mut attrs = Vec::new();
                attrs.push(("style".to_string(), "filled".to_string()));
                attrs.push((
                    "fillcolor".to_string(),
                    self.get_color(id, &self.state.borrow(), show_false),
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

    pub fn reset(
        &self,
        seed: Option<u64>,
    ) -> SimResult<((SimulatorObs<I>, AttackerObs<I>, DefenderObs<I>), Info)> {
        let new_state = SimulatorState::new(&self.g, seed, self.config.randomize_ttc)?;

        log::info!("Resetting simulator with seed {:?}", seed);
        log::info!("Initial state:\n {:?}", new_state);
        let flag_status = self.g.get_flag_status(&new_state.compromised_steps);
        self.history.borrow_mut().clear();
        let info = new_state.to_info(
            self.g.number_of_attacks(),
            self.g.number_of_defenses(),
            flag_status,
        );

        let sim_obs = SimulatorObs::from(&new_state);
        let attacker_obs = AttackerObs::new(&new_state, &self.g);
        let defender_obs = DefenderObs::new(&new_state, &self.g);

        self.state.replace(new_state);

        return Ok(((sim_obs, attacker_obs, defender_obs), info));
    }

    pub fn reset_vec(&self, seed: Option<u64>) -> SimResult<(VectorizedObservation, Info)> {
        let ((sim_obs, a_obs, d_obs), info) = self.reset(seed)?;
        let result = Ok((self.vectorize_obs(&sim_obs, &a_obs, &d_obs, (0, 0)), info));
        return result;
    }

    fn to_vec<T, F>(&self, predicate: F) -> Vec<T>
    where
        F: Fn(&I) -> T,
    {
        return self.index_to_id.iter().map(predicate).collect();
    }

    fn vectorize_obs(
        &self,
        sim_obs: &SimulatorObs<I>,
        attacker_obs: &AttackerObs<I>,
        defender_obs: &DefenderObs<I>,
        rewards: (i64, i64),
    ) -> VectorizedObservation {
        // reverse graph id to action index mapping

        let attack_surface_vec = self.to_vec(|id| attacker_obs.possible_objects.contains(&id));
        let ttc_remaining = self.to_vec(|id| sim_obs.remaining_ttc[id]);

        let step_state = self.to_vec(|id| {
            sim_obs.enabled_defenses.contains(id) || sim_obs.compromised_steps.contains(id)
        });

        let step_info = self
            .index_to_id
            .iter()
            .map(|id| {
                let step = self.g.get_step(id).unwrap();
                step.to_info_tuple()
            })
            .map(|x| (self.g.word2idx(x.0), x.1, self.g.word2idx(x.2)))
            .collect::<Vec<StepInfo>>();

        let defender_observed_vec: Vec<i64> =
            self.to_vec(|id| match defender_obs.observed_steps.get(&id) {
                Some(i) => i64::from(*i),
                None => -1,
            });
        let attacker_observed_vec: Vec<i64> =
            self.to_vec(|id| match attacker_obs.observed_steps.get(&id) {
                Some(i) => i64::from(*i),
                None => -1,
            });

        let defense_surface_vec = self.to_vec(|id| defender_obs.possible_objects.contains(&id));

        let defender_action_mask = self
            .idx2action
            .iter()
            .map(|i| defender_obs.possible_actions.contains(i))
            .collect::<Vec<bool>>();

        let attacker_action_mask = self
            .idx2action
            .iter()
            .map(|i| attacker_obs.possible_actions.contains(i))
            .collect::<Vec<bool>>();

        // map edge indices to the vector indices
        let vector_edges = self
            .g
            .edges()
            .iter()
            .map(|(from, to)| (self.id_to_index[from], self.id_to_index[to]))
            .collect::<Vec<(usize, usize)>>();

        VectorizedObservation {
            attacker_possible_objects: attack_surface_vec,
            defender_possible_objects: defense_surface_vec,
            defender_possible_actions: defender_action_mask,
            attacker_possible_actions: attacker_action_mask,
            ttc_remaining,
            defender_observation: defender_observed_vec,
            attacker_observation: attacker_observed_vec,
            step_info,
            state: step_state,
            edges: vector_edges,
            //defense_indices: self.defender_action_to_state(),
            flags: self.g.flag_to_index(&self.id_to_index),
            attacker_reward: rewards.0,
            defender_reward: rewards.1,
        }
    }

    pub fn step_vec(
        &self,
        action_dict: HashMap<String, ParameterAction>,
    ) -> SimResult<(VectorizedObservation, Info)> {
        let (state, info, rewards) = self.step(action_dict)?;
        let result = Ok((
            self.vectorize_obs(&state.0, &state.1, &state.2, rewards),
            info,
        ));

        return result;
    }

    pub fn step(
        &self,
        action_dict: HashMap<String, ParameterAction>,
    ) -> SimResult<(
        (SimulatorObs<I>, AttackerObs<I>, DefenderObs<I>),
        Info,
        (i64, i64),
    )> {
        log::info!("Step with action dict {:?}", action_dict);

        let (rewards, new_state) = self.calculate_next_state(action_dict)?;

        log::info!("New state:\n{:?}", new_state);

        let flag_status = self.g.get_flag_status(&new_state.compromised_steps);
        let sim_obs = SimulatorObs::from(&new_state);
        let attacker_obs = AttackerObs::new(&new_state, &self.g);
        let defender_obs = DefenderObs::new(&new_state, &self.g);
        let info = new_state.to_info(
            self.g.number_of_attacks(),
            self.g.number_of_defenses(),
            flag_status,
        );
        self.history
            .borrow_mut()
            .push(self.state.replace(new_state));
        return Ok(((sim_obs, attacker_obs, defender_obs), info, rewards));
    }

    /*
    fn terminate_idx(&self) -> usize {
        return self.actions["terminate"];
    }
    */

    fn get_step_from_action(&self, action: Option<&ParameterAction>) -> Option<&I> {
        match action {
            Some(action) => match action.1 {
                Some(a) => self.index_to_id.get(a),
                None => None,
            },
            None => None,
        }
    }

    fn check_action(
        selected_step: Option<&I>,
        valid_steps: HashSet<I>,
        agent_name: String,
    ) -> SimResult<()> {
        match selected_step {
            Some(step) => {
                if !valid_steps.contains(step) {
                    return Err(SimError {
                        error: format!(
                            "Invalid step for {}: {:?} not in {:?}",
                            agent_name, step, valid_steps
                        ),
                    });
                }
                return Ok(());
            }
            None => {
                return Ok(());
            }
        }
    }

    fn calculate_next_state(
        &self,
        action_dict: HashMap<String, ParameterAction>,
    ) -> SimResult<((i64, i64), SimulatorState<I>)> {
        // Attacker selects and attack step from the attack surface
        // Defender selects a defense step from the defense surface, which is
        // the vector of all defense steps that are not disabled
        let defender_selected_step = self.get_step_from_action(action_dict.get("defender"));
        let attacker_selected_step = self.get_step_from_action(action_dict.get("attacker"));
        let old_state = self.state.borrow();

        if self.config.strict {
            // Be strict about agents selecting valid actions
            let attacker_obs = AttackerObs::new(&old_state, &self.g);
            let valid_steps = attacker_obs.possible_objects.clone();
            Self::check_action(attacker_selected_step, valid_steps, "attacker".to_string())?;
            let defender_obs = DefenderObs::new(&old_state, &self.g);
            let valid_steps = defender_obs.possible_objects.clone();
            Self::check_action(defender_selected_step, valid_steps, "defender".to_string())?;
        }

        match old_state.get_new_state(&self.g, attacker_selected_step, defender_selected_step) {
            Ok(new_state) => {
                return Ok((
                    (
                        new_state.attacker_reward(&self.g, attacker_selected_step),
                        new_state.defender_reward(&self.g, defender_selected_step),
                    ),
                    new_state,
                ));
            }
            Err(e) => {
                return Err(SimError {
                    error: e.to_string(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use std::collections::{HashMap, HashSet};

    use crate::{
        attacker_state::AttackerObs,
        config,
        defender_state::DefenderObs,
        loading::load_graph_from_json,
        observation::{Info, VectorizedObservation},
        runtime::SimulatorRuntime,
    };

    const FILENAME: &str = "attack_simulator/graphs/four_ways.json";

    #[test]
    fn test_sim_fnr() {
        let filename = FILENAME;
        let config = config::SimulatorConfig {
            randomize_ttc: false,
            false_negative_rate: 1.0,
            false_positive_rate: 0.0,
            log: false,
            show_false: true,
            strict: true,
        };
        let graph = load_graph_from_json(
            filename,
            None,
            config.false_negative_rate,
            config.false_positive_rate,
        )
        .unwrap();
        //let num_defenses = graph.number_of_defenses();
        let num_entrypoints = graph.entry_points().len();

        let sim = SimulatorRuntime::new(graph, config).unwrap();

        let (initial_state, _info) = sim.reset(None).unwrap();
        let steps_observed_as_compromised = initial_state.2.steps_observed_as_compromised(&sim.g);
        //println!("Confusion: {:?}", sim.confusion_per_step);
        let false_negatives: HashSet<&_> = initial_state
            .0
            .compromised_steps
            .difference(&steps_observed_as_compromised)
            .collect();

        let false_positives: HashSet<&_> = steps_observed_as_compromised
            .difference(&initial_state.0.compromised_steps)
            .collect();

        assert_eq!(
            false_negatives.len(),
            initial_state.0.compromised_steps.len()
        );
        assert_eq!(false_negatives.len(), num_entrypoints);
        assert_eq!(false_positives.len(), 0);
    }

    #[test]
    fn test_sim_fpr() {
        let filename = FILENAME;
        let config = config::SimulatorConfig {
            randomize_ttc: false,
            false_negative_rate: 0.0,
            false_positive_rate: 1.0,
            log: false,
            show_false: true,
            strict: true,
        };
        let graph = load_graph_from_json(
            filename,
            None,
            config.false_negative_rate,
            config.false_positive_rate,
        )
        .unwrap();
        let num_attacks = graph.number_of_attacks();
        let num_entrypoints = graph.entry_points().len();

        let sim = SimulatorRuntime::new(graph, config).unwrap();

        let (initial_state, _info) = sim.reset(None).unwrap();

        //println!("Confusion: {:?}", sim.confusion_per_step);
        let steps_observed_as_compromised = initial_state.2.steps_observed_as_compromised(&sim.g);
        let false_negatives: HashSet<&_> = initial_state
            .0
            .compromised_steps
            .difference(&steps_observed_as_compromised)
            .collect();

        let false_positives: HashSet<&_> = steps_observed_as_compromised
            .difference(&initial_state.0.compromised_steps)
            .collect();

        assert_eq!(false_negatives.len(), 0);
        assert_eq!(false_positives.len(), num_attacks - num_entrypoints);
    }

    #[test]
    fn test_sim_init() {
        let filename = FILENAME;
        let graph = load_graph_from_json(filename, None, 0.0, 0.0).unwrap();
        //let num_defenses = graph.number_of_defenses();
        let num_entrypoints = graph.entry_points().len();
        let config = config::SimulatorConfig {
            randomize_ttc: false,
            false_negative_rate: 0.0,
            false_positive_rate: 0.0,
            log: false,
            show_false: true,
            strict: true,
        };
        let sim = SimulatorRuntime::new(graph, config).unwrap();

        let initial_state = sim.state.borrow();

        let attacker_obs = AttackerObs::new(&initial_state, &sim.g);
        let defender_obs = DefenderObs::new(&initial_state, &sim.g);

        assert_eq!(
            attacker_obs.observed_steps.len(),
            num_entrypoints + attacker_obs.possible_objects.len()
        );
        assert_eq!(
            defender_obs.steps_observed_as_compromised(&sim.g).len(),
            num_entrypoints
        );
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
    fn test_test_graph() {
        let filename = "attack_simulator/graphs/test_graph.json";
        let graph = load_graph_from_json(filename, None, 0.0, 0.0).unwrap();
        let config = config::SimulatorConfig::default();
        let sim = SimulatorRuntime::new(graph, config).unwrap();

        let initial_state = sim.state.borrow();

        let attacker_obs = AttackerObs::new(&initial_state, &sim.g);

        let graph = load_graph_from_json(filename, None, 0.0, 0.0).unwrap();

        assert!(attacker_obs
            .possible_objects
            .contains(&graph.translate_id("b:1:attack")));
        //assert!(attacker_obs.possible_objects.contains(&graph.translate_id("c:1:attack")));

        let expected_obs = HashMap::from([
            (graph.translate_id("a:1:firstSteps"), true),
            (graph.translate_id("b:1:attack"), false),
            (graph.translate_id("c:1:attack"), false),
        ]);
        assert_eq!(attacker_obs.observed_steps, expected_obs);

        assert_eq!(attacker_obs.observed_steps.len(), 3);
        assert_eq!(attacker_obs.possible_objects.len(), 2);
    }

    #[test]
    fn test_sim_obs() {
        let filename = FILENAME;
        let graph = load_graph_from_json(filename, None, 0.0, 0.0).unwrap();
        //let num_attacks = graph.number_of_attacks();
        let num_defenses = graph.number_of_defenses();
        let num_entrypoints = graph.entry_points().len();
        let config = config::SimulatorConfig::default();
        let sim = SimulatorRuntime::new(graph, config).unwrap();

        let observation: VectorizedObservation;
        let _info: Info;

        (observation, _info) = sim.reset_vec(None).unwrap();

        assert_eq!(
            observation
                .defender_possible_objects
                .iter()
                .filter(|&x| *x)
                .count(),
            num_defenses
        );

        //println!("AS: {:?}", observation.attack_surface);

        let _steps = observation
            .attacker_possible_objects
            .iter()
            .enumerate()
            .filter_map(|(i, x)| match x {
                true => Some(i),
                false => None,
            })
            .collect::<Vec<usize>>();
        //let strings = sim.translate_index_vec(&steps);
        //println!("AS: {:?}", strings);

        assert_eq!(
            observation.attacker_possible_actions.len(),
            sim.action2idx.len()
        );

        assert_eq!(
            observation
                .defender_possible_objects
                .iter()
                .filter(|&x| *x)
                .count(),
            num_defenses
        );

        assert_eq!(observation.state.len(), sim.g.nodes().len());
        assert_eq!(
            observation.state.iter().filter(|&x| *x).count(),
            num_entrypoints
        ); // 4 available defenses + 1 compromised attack step

        //check that all defense steps are disabled

        let defense_indices = observation
            .defender_possible_objects
            .iter()
            .enumerate()
            .filter_map(|(i, x)| match x {
                true => Some(i),
                false => None,
            })
            .collect::<Vec<usize>>();

        for i in defense_indices.iter() {
            assert_eq!(observation.state[*i], false); // Defense steps should be disabled
        }

        //assert!(observation.ttc_remaining.iter().sum::<u64>() > 0);

        let edges = observation.edges;
        let entrypoints = sim.g.entry_points();
        let entrypoint_indices: Vec<&usize> = sim
            .id_to_index
            .iter()
            .filter_map(|(id, idx)| match entrypoints.get(id) {
                Some(_i) => Some(idx),
                None => None,
            })
            .collect();

        for index in entrypoint_indices {
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
                assert_eq!(observation.attacker_possible_objects[*index], true);
            }
        }
    }
}
