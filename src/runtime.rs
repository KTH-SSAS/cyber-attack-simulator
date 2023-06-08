use rand_chacha::ChaChaRng;
use rand_distr::Distribution;
use std::cmp::max;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::{fmt, vec};

use itertools::Itertools;

use crate::attackgraph::{AttackGraph, TTCType};
use crate::config::SimulatorConfig;
use crate::graph::NodeID;
use crate::observation::{Info, Observation};

use rand::{Rng, SeedableRng};
use rand_distr::Exp;

pub const ACTION_NOP: usize = 0; // No action
pub const ACTION_TERMINATE: usize = 1; // Terminate the simulation
pub const SPECIAL_ACTIONS: [usize; 1] = [ACTION_NOP];

pub const ATTACKER: usize = 0;
pub const DEFENDER: usize = 1;

pub const DEFENSE_ENABLED: bool = false;
pub const DEFENSE_DISABLED: bool = true;

type SimResult<T> = std::result::Result<T, SimError>;

#[derive(Debug, Clone)]
pub struct SimError {
    pub error: String,
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SimError: {}", self.error)
    }
}

pub struct ActionResult {
    pub ttc_diff: HashMap<NodeID, i32>,
    pub enabled_defenses: HashSet<NodeID>,
}

pub(crate) struct SimulatorState {
    time: u64,
    rng: ChaChaRng,
    enabled_defenses: HashSet<NodeID>,
    compromised_steps: HashSet<NodeID>,
    attack_surface: HashSet<NodeID>,
    remaining_ttc: HashMap<u64, TTCType>,
    num_observed_alerts: usize,
    false_alerts: HashSet<u64>,
    missed_alerts: HashSet<u64>,
    //actions: HashMap<String, usize>,
}

impl SimulatorState {
    fn new(graph: &AttackGraph, seed: u64, randomize_ttc: bool) -> SimResult<SimulatorState> {
        let mut rng = ChaChaRng::seed_from_u64(seed);
        let enabled_defenses = HashSet::new();
        let ttc_params = graph.ttc_params();
        let remaining_ttc = match randomize_ttc {
            true => get_initial_ttc_vals(&mut rng, &ttc_params),
            false => HashMap::from_iter(ttc_params),
        };
        let compromised_steps = graph.entry_points.clone();
        let attack_surface = calculate_attack_surface(graph, &compromised_steps, &HashSet::new())?;

        Ok(SimulatorState {
            time: 0,
            rng,
            enabled_defenses,
            attack_surface,
            remaining_ttc,
            num_observed_alerts: 0,
            compromised_steps,
            //actions: HashMap::new(),
            false_alerts: HashSet::new(),
            missed_alerts: HashSet::new(),
        })
    }

    pub fn total_ttc_remaining(&self) -> TTCType {
        return self.remaining_ttc.iter().map(|(_, &ttc)| ttc).sum();
    }

    fn to_info(
        &self,
        num_attacks: usize,
        num_defenses: usize,
        flag_status: HashMap<u64, bool>,
    ) -> Info {
        let num_compromised_steps = self.compromised_steps.len();
        let num_enabled_defenses = self.enabled_defenses.len();
        let num_compromised_flags = flag_status.iter().filter(|(_, v)| **v).count();

        Info {
            time: self.time,
            sum_ttc: self.total_ttc_remaining(),
            num_compromised_steps,
            num_observed_alerts: self.num_observed_alerts,
            perc_compromised_steps: num_compromised_steps as f64 / num_attacks as f64,
            perc_defenses_activated: num_enabled_defenses as f64 / num_defenses as f64,
            num_compromised_flags,
            perc_compromised_flags: num_compromised_flags as f64 / flag_status.len() as f64,
        }
    }
}

//type ActionIndex = usize;
type ActorIndex = usize;

pub(crate) struct SimulatorRuntime {
    g: AttackGraph,
    state: RefCell<SimulatorState>,
    history: Vec<SimulatorState>,
    pub config: SimulatorConfig,
    pub confusion_per_step: HashMap<u64, (f64, f64)>,
    pub ttc_sum: TTCType,

    pub actors: HashMap<String, ActorIndex>,

    pub id_to_index: HashMap<NodeID, usize>,
    pub defender_action_to_graph: Vec<NodeID>,
    pub attacker_action_to_graph: Vec<NodeID>,
}

pub fn get_initial_ttc_vals(
    rng: &mut ChaChaRng,
    ttc_params: &Vec<(NodeID, TTCType)>,
) -> HashMap<NodeID, TTCType> {
    // Note! The sampling order has to be deterministic!
    let ttc_remaining = ttc_params.iter().map(|(id, ttc)| {
        if *ttc == 0 {
            (*id, *ttc)
        } else {
            let exp = Exp::new(1.0 / *ttc as f64).unwrap();
            (*id, max(1, exp.sample(rng) as TTCType)) //Min ttc is 1
        }
    });
    return ttc_remaining.collect();
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

impl SimulatorRuntime {
    // Path: src/sim.rs
    pub fn new(graph: AttackGraph, config: SimulatorConfig) -> SimResult<SimulatorRuntime> {
        let index_to_id = graph
            .graph
            .nodes
            .iter()
            .map(|(x, _)| *x)
            .sorted() // ensure deterministic order
            .collect::<Vec<u64>>();

        // Maps the id of a node in the graph to an index in the state vector
        let id_to_index: HashMap<u64, usize> = index_to_id
            .iter()
            .enumerate()
            .map(|(i, n)| (*n, i))
            .collect();

        // Maps the index of the action to the id of the node in the graph
        let attacker_action_to_graph = index_to_id.clone();

        let defender_action_to_graph = index_to_id
            .iter()
            .filter_map(|id| match graph.defense_steps.contains(id) {
                true => Some(*id),
                false => None,
            })
            .collect::<Vec<u64>>();

        let initial_state = SimulatorState::new(&graph, config.seed, config.randomize_ttc)?;

        let fnr_fpr_per_step = index_to_id
            .iter()
            .map(|id| {
                match graph.attack_steps.contains(&id) {
                    true => (
                        *id,
                        (config.false_negative_rate, config.false_positive_rate),
                    ),
                    false => (*id, (0.0, 0.0)), // No false negatives or positives for defense steps
                }
            })
            .collect::<HashMap<u64, (f64, f64)>>();

        let sim = SimulatorRuntime {
            state: RefCell::new(initial_state),
            g: graph,
            confusion_per_step: fnr_fpr_per_step,
            config,
            ttc_sum: 0,
            id_to_index,
            attacker_action_to_graph,
            defender_action_to_graph,
            history: Vec::new(),
            actors: HashMap::from_iter(vec![
                ("defender".to_string(), DEFENDER),
                ("attacker".to_string(), ATTACKER),
            ]),
        };

        return Ok(sim);
    }

    pub fn reset(&mut self, seed: Option<u64>) -> SimResult<(Observation, Info)> {
        if let Some(seed) = seed {
            self.config.seed = seed;
        }

        let new_state = SimulatorState::new(&self.g, self.config.seed, self.config.randomize_ttc)?;

        let flag_status = get_flag_status(&self.g.flags, &new_state.compromised_steps);

        let result = Ok((
            self.map_state_to_observation(&new_state),
            new_state.to_info(
                self.g.attack_steps.len(),
                self.g.defense_steps.len(),
                flag_status,
            ),
        ));
        self.ttc_sum = new_state.remaining_ttc.iter().map(|(_, &ttc)| ttc).sum();
        self.history.clear();
        self.state.replace(new_state);
        return result;
    }

    pub fn defender_action_to_state(&self) -> Vec<usize> {
        return self
            .defender_action_to_graph
            .iter()
            .map(|id| self.id_to_index[id])
            .collect();
    }

    pub fn flag_to_index(&self) -> Vec<usize> {
        self.g
            .flags
            .iter()
            .map(|id| self.id_to_index[id])
            .collect::<Vec<usize>>()
    }

    pub fn enable_defense_step(
        &self,
        step_id: NodeID,
    ) -> SimResult<(HashSet<NodeID>, HashMap<NodeID, i32>)> {
        let mut ttc_change = HashMap::new();

        let affected_attacks = self.g.graph.children(&step_id);

        match affected_attacks {
            Some(affected_attacks) => {
                affected_attacks.iter().for_each(|x| {
                    ttc_change.insert(*x, 1000);
                });
            }
            None => {}
        }

        return Ok((HashSet::from([step_id]), ttc_change));
    }

    pub fn defense_action(&self, action: usize) -> SimResult<ActionResult> {
        let state = self.state.borrow();
        let defense_step_id = self.defender_action_to_graph[action];

        if state.enabled_defenses.contains(&defense_step_id) {
            // Already enabled
            // Do nothing
            return Ok(ActionResult {
                enabled_defenses: HashSet::new(),
                ttc_diff: HashMap::new(),
            });
        }

        let (enabled_defenses, ttc_diff) = self.enable_defense_step(defense_step_id)?;

        let result = ActionResult {
            enabled_defenses,
            ttc_diff,
        };

        return Ok(result);
    }

    pub fn work_on_attack_step(attack_step_id: NodeID) -> HashMap<NodeID, i32> {
        return HashMap::from([(attack_step_id, -1)]);
    }

    pub fn attack_action(&self, attacker_action: usize) -> SimResult<ActionResult> {
        // Have the attacker perform an action.

        let attack_step_id = self.attacker_action_to_graph[attacker_action];

        let state = self.state.borrow();
        let attack_surface_empty = state.attack_surface.is_empty();

        if attack_surface_empty {
            return Err(SimError {
                error: "Attack surface is empty.".to_string(),
            });
        }

        let result = ActionResult {
            enabled_defenses: HashSet::new(),
            ttc_diff: SimulatorRuntime::work_on_attack_step(attack_step_id),
        };

        return Ok(result);
    }

    fn map_state_to_observation(&self, state: &SimulatorState) -> Observation {
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

        let mut step_state = vec![false; self.id_to_index.len()];

        let disabled_defenses: HashSet<&u64> = self
            .g
            .defense_steps
            .difference(&state.enabled_defenses)
            .collect();
        disabled_defenses
            .iter()
            .map(|&node_id| self.id_to_index[node_id])
            .for_each(|index| {
                step_state[index] = true;
            });

        state
            .compromised_steps
            .iter()
            .map(|node_id| self.id_to_index[&node_id])
            .for_each(|index| {
                step_state[index] = true;
            });

        let ids_observed = state
            .compromised_steps // true alerts
            .union(&state.false_alerts) // add false alerts
            .filter_map(|x| match state.missed_alerts.contains(x) {
                true => None, // remove missed alerts
                false => Some(x),
            })
            .collect::<HashSet<&u64>>();

        let mut ids_observed_vec = vec![false; self.id_to_index.len()];
        ids_observed.iter().for_each(|node_id| {
            ids_observed_vec[self.id_to_index[node_id]] = true;
        });

        let mut action_mask = vec![false; SPECIAL_ACTIONS.len()];

        action_mask[ACTION_NOP] = true;
        //action_mask[ACTION_TERMINATE] = false;

        let defense_state = self
            .defender_action_to_graph
            .iter()
            .map(|f| self.id_to_index[&f])
            .map(|x| step_state[x])
            .collect::<Vec<bool>>();

        assert!(defense_state.len() == self.defender_action_to_graph.len());

        let defender_action_mask = action_mask
            .iter()
            .chain(defense_state.iter())
            .cloned()
            .collect::<Vec<bool>>();
        let attacker_action_mask = action_mask
            .iter()
            .chain(attack_surface_vec.iter())
            .cloned()
            .collect::<Vec<bool>>();

        assert!(
            defender_action_mask.len()
                == self.defender_action_to_graph.len() + SPECIAL_ACTIONS.len()
        );

        let edges = &self.g.graph.edges;

        // map edge indices to the vector indices
        let vector_edges = edges
            .iter()
            .map(|(from, to)| (self.id_to_index[from], self.id_to_index[to]))
            .collect::<Vec<(usize, usize)>>();

        Observation {
            attack_surface: attack_surface_vec,
            defense_surface: defense_state,
            defender_action_mask,
            attacker_action_mask,
            ttc_remaining,
            ids_observation: ids_observed_vec,
            state: step_state,
            edges: vector_edges,
            defense_indices: self.defender_action_to_state(),
            flags: self.flag_to_index(),
        }
    }

    pub fn step(&mut self, action_dict: HashMap<String, usize>) -> SimResult<(Observation, Info)> {
        let new_state = self.calculate_next_state(action_dict)?;

        let flag_status = get_flag_status(&self.g.flags, &new_state.compromised_steps);

        let result = Ok((
            self.map_state_to_observation(&new_state),
            new_state.to_info(
                self.g.attack_steps.len(),
                self.g.defense_steps.len(),
                flag_status,
            ),
        ));

        self.history.push(self.state.replace(new_state));

        return result;
    }

    fn calculate_next_state(
        &self,
        action_dict: HashMap<String, usize>,
    ) -> SimResult<SimulatorState> {
        let old_state = self.state.borrow();

        // Attacker selects and attack step from the attack surface
        // Defender selects a defense step from the defense surface, which is the vector of all defense steps that are not disabled

        let actor_funcs: Vec<fn(&SimulatorRuntime, usize) -> SimResult<ActionResult>> = vec![
            SimulatorRuntime::attack_action,
            SimulatorRuntime::defense_action,
        ];

        let total_result: ActionResult = action_dict.iter().fold(
            ActionResult {
                ttc_diff: HashMap::new(),
                enabled_defenses: old_state.enabled_defenses.clone(),
            },
            |mut total_result, (actor, action)| {
                if *action < SPECIAL_ACTIONS.len() {
                    return total_result;
                }

                let step_idx = *action - SPECIAL_ACTIONS.len();
                let actor_id = self.actors[actor];
                let result = actor_funcs[actor_id](&self, step_idx).unwrap();

                for (step_id, ttc) in result.ttc_diff.iter() {
                    let current_ttc = total_result.ttc_diff.entry(*step_id).or_insert(0);
                    *current_ttc += *ttc;
                }

                total_result
                    .enabled_defenses
                    .extend(result.enabled_defenses);

                total_result
            },
        );

        let mut remaining_ttc: HashMap<NodeID, u64> = old_state.remaining_ttc.clone();
        total_result.ttc_diff.iter().for_each(|(step_id, ttc)| {
            remaining_ttc.entry(*step_id).and_modify(|current_ttc| {
                *current_ttc = max(0, *current_ttc as i64 + *ttc as i64) as u64;
            });
        });

        let enabled_defenses = total_result.enabled_defenses;

        let compromised_steps = calculate_compromised_steps(&self.g, &remaining_ttc);

        let uncompromised_steps: HashSet<&u64> =
            self.g.attack_steps.difference(&compromised_steps).collect();

        let attack_surface =
            calculate_attack_surface(&self.g, &compromised_steps, &enabled_defenses)?;

        let mut rng = old_state.rng.clone();

        let missed_alerts = compromised_steps
            .iter()
            .filter_map(
                |id| match rng.gen::<f64>() < self.confusion_per_step[id].0 {
                    true => Some(*id), // if p < fnr, then we missed an alert
                    false => None,
                },
            )
            .collect::<HashSet<u64>>();

        let false_alerts = uncompromised_steps
            .iter()
            .filter_map(
                |id| match rng.gen::<f64>() < self.confusion_per_step[id].1 {
                    true => Some(**id), // if p < fpr, then we got a false alert
                    false => None,
                },
            )
            .collect::<HashSet<u64>>();

        Ok(SimulatorState {
            attack_surface,
            enabled_defenses,
            remaining_ttc,
            compromised_steps,
            time: old_state.time + 1,
            rng,
            num_observed_alerts: 0,
            //actions: action_dict,
            missed_alerts,
            false_alerts,
        })
    }
}

fn get_flag_status(flags: &HashSet<u64>, compromised_steps: &HashSet<u64>) -> HashMap<u64, bool> {
    return flags
        .iter()
        .map(|flag_id| (*flag_id, compromised_steps.contains(flag_id)))
        .collect();
}

fn calculate_compromised_steps(
    graph: &AttackGraph,
    remaining_ttc: &HashMap<u64, u64>,
) -> HashSet<u64> {
    let steps_with_zero_ttc: HashSet<u64> = remaining_ttc
        .iter()
        .filter_map(|(step, ttc)| match ttc {
            0 => Some(*step),
            _ => None,
        })
        .collect();

    steps_with_zero_ttc
        .iter()
        .filter_map(|step| match graph.graph.nodes.get(step) {
            Some(step) => Some(step),
            None => None,
        })
        .filter(|step| step.is_traversible(&steps_with_zero_ttc))
        .map(|step| step.id)
        .collect()
}

fn calculate_attack_surface(
    graph: &AttackGraph,
    compromised_steps: &HashSet<u64>,
    defense_state: &HashSet<u64>,
) -> SimResult<HashSet<u64>> {
    let attack_surface: HashSet<u64> =
        compromised_steps
            .iter()
            .fold(HashSet::new(), |mut acc, step| {
                let vulnerable_children =
                    match graph.get_vulnerable_children(step, &compromised_steps, &defense_state) {
                        Ok(v) => v,
                        Err(e) => {
                            panic!("Error in get_vulnerable_children: {} for step {}", e, step)
                        }
                    };
                acc.extend(vulnerable_children);
                acc
            });

    return Ok(attack_surface);
}

#[cfg(test)]
mod tests {
    use crate::{
        attackgraph, config,
        observation::{Info, Observation},
        runtime::{SimulatorRuntime, SPECIAL_ACTIONS},
    };

    #[test]
    fn test_sim_init() {
        let filename = "graphs/four_ways.yaml";
        let graph = attackgraph::load_graph_from_yaml(filename);
        let config = config::SimulatorConfig::default();
        let sim = SimulatorRuntime::new(graph, config).unwrap();

        let initial_state = sim.state.borrow();

        assert_eq!(initial_state.enabled_defenses.len(), 0);
        assert_eq!(initial_state.compromised_steps.len(), 1);
        assert_eq!(initial_state.remaining_ttc.len(), sim.g.graph.nodes.len());
        assert_eq!(
            initial_state
                .remaining_ttc
                .iter()
                .filter(|(_, &ttc)| ttc == 0)
                .count(),
            4 + 1 // 4 defense steps + 1 entrypoint
        );
        assert_eq!(initial_state.attack_surface.len(), 4);
        assert_eq!(
            sim.defender_action_to_graph.len(),
            sim.g.defense_steps.len()
        );
        assert_eq!(sim.attacker_action_to_graph.len(), sim.g.graph.nodes.len());
    }

    #[test]
    fn test_sim_obs() {
        let filename = "graphs/four_ways.yaml";
        let graph = attackgraph::load_graph_from_yaml(filename);
        let config = config::SimulatorConfig::default();
        let mut sim = SimulatorRuntime::new(graph, config).unwrap();

        let observation: Observation;
        let _info: Info;

        (observation, _info) = sim.reset(None).unwrap();

        assert_eq!(
            observation.defense_surface.iter().filter(|&x| *x).count(),
            4
        );
        assert_eq!(observation.attack_surface.iter().filter(|&x| *x).count(), 4);
        assert_eq!(
            observation.defender_action_mask.len(),
            4 + SPECIAL_ACTIONS.len()
        ); // count defenses + special actions
        assert_eq!(
            observation.attacker_action_mask.len(),
            sim.g.graph.nodes.len() + SPECIAL_ACTIONS.len()
        );
        assert_eq!(
            observation
                .defender_action_mask
                .iter()
                .filter(|&x| *x)
                .count(),
            4 + 1
        ); // count available defenses + available special actions
        assert_eq!(
            observation
                .attacker_action_mask
                .iter()
                .filter(|&x| *x)
                .count(),
            4 + 1
        );
        assert_eq!(observation.state.iter().filter(|&x| *x).count(), 4 + 1); // 4 available defenses + 1 compromised attack step
        assert_eq!(observation.state.len(), sim.g.graph.nodes.len());

        //check that all defense steps are disabled
        for i in observation.defense_indices.iter() {
            assert_eq!(observation.state[*i], true);
        }

        assert!(observation.ttc_remaining.iter().sum::<u64>() > 0);

        let edges = observation.edges;
        let entrypoint = sim.g.entry_points;
        let entrypoint_index = sim
            .id_to_index
            .iter()
            .find_map(|(id, index)| match entrypoint.contains(&id) {
                true => Some(index),
                false => None,
            })
            .unwrap();

        let indices_from_entrypoint = edges
            .iter()
            .filter_map(|(from, to)| match from == entrypoint_index {
                true => Some(to),
                false => None,
            })
            .collect::<Vec<&usize>>();

        // There should be four edges from the entrypoint
        assert_eq!(indices_from_entrypoint.len(), 4);

        // All the outgoing edges should be in the attack surface
        for index in indices_from_entrypoint {
            assert_eq!(observation.attack_surface[*index], true);
        }
    }
}
