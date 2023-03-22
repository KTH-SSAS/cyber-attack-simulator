use pyo3::pyclass;
use rand_chacha::ChaChaRng;
use rand_distr::Distribution;
use std::cmp::max;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::{fmt, vec};

use crate::attackgraph::{AttackGraph, TTCType};
use crate::config::SimulatorConfig;
use crate::graph::NodeID;

use rand::SeedableRng;
use rand_distr::Exp;

pub const ACTION_NOP: usize = 0; // No action
pub const ACTION_TERMINATE: usize = 1; // Terminate the simulation
pub const SPECIAL_ACTIONS: [usize; 2] = [ACTION_NOP, ACTION_TERMINATE];

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

#[pyclass]
pub(crate) struct Observation {
    #[pyo3(get)]
    pub attack_surface: Vec<bool>,
    #[pyo3(get)]
    pub defense_surface: Vec<bool>,
    #[pyo3(get)]
    pub attacker_action_mask: Vec<bool>,
    #[pyo3(get)]
    pub defender_action_mask: Vec<bool>,
    #[pyo3(get)]
    pub defense_state: Vec<bool>,
    #[pyo3(get)]
    pub attack_state: Vec<bool>,
    #[pyo3(get)]
    pub ids_observation: Vec<bool>,
    #[pyo3(get)]
    pub ttc_remaining: Vec<u64>,
}

#[pyclass]
pub(crate) struct Info {
    #[pyo3(get)]
    pub time: u64,
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

#[pyclass]
pub(crate) struct SimulatorState {
    time: u64,
    rng: ChaChaRng,
    enabled_defenses: HashSet<NodeID>,
    compromised_steps: HashSet<NodeID>,
    attack_surface: HashSet<NodeID>,
    remaining_ttc: HashMap<u64, TTCType>,
    num_observed_alerts: usize,
    actions: HashMap<String, usize>,
}

impl SimulatorState {
    fn new(
        graph: &AttackGraph,
        seed: u64,
        randomize_ttc: bool,
    ) -> SimResult<SimulatorState> {
        
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
            actions: HashMap::new(),
        })
    }

    fn to_info(&self, num_attacks: usize, num_defenses: usize, flag_status: HashMap<u64, bool>) -> Info {
        let num_compromised_steps = self.compromised_steps.len();
        let num_enabled_defenses = self.enabled_defenses.len();
        let num_compromised_flags = flag_status.iter().filter(|(_, v)| **v).count();


        Info {
            time: self.time,
            num_compromised_steps,
            num_observed_alerts: self.num_observed_alerts,
            perc_compromised_steps: num_compromised_steps as f64 / num_attacks as f64,
            perc_defenses_activated: num_enabled_defenses as f64 / num_defenses as f64,
            num_compromised_flags,
            perc_compromised_flags: num_compromised_flags as f64 / flag_status.len() as f64,
        }
    }
}

type ActionIndex = usize;
type ActorIndex = usize;

pub(crate) struct SimulatorRuntime {
    g: AttackGraph,
    state: RefCell<SimulatorState>,
    history: Vec<SimulatorState>,
    pub config: SimulatorConfig,
    false_negative_rate: f64,
    false_positive_rate: f64,
    pub ttc_sum: TTCType,
    
    pub actors: HashMap<String, ActorIndex>,

    pub graph_to_attack_state: HashMap<NodeID, usize>,
    pub graph_to_defense_state: HashMap<NodeID, usize>,

    pub defender_action_to_graph: Vec<NodeID>,
    pub attacker_action_to_graph: Vec<NodeID>,
}

pub fn get_initial_ttc_vals(
    rng : &mut ChaChaRng,
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
        let mut attack_indices = graph.attack_steps.iter().map(|x| *x).collect::<Vec<u64>>();
        attack_indices.sort();
        let mut defense_indices = graph.defense_steps.iter().map(|x| *x).collect::<Vec<u64>>();
        defense_indices.sort();

        let graph_to_attack_state = attack_indices
            .iter()
            .enumerate()
            .map(|(i, n)| (*n, i))
            .collect();

        let graph_to_defense_state = defense_indices
            .iter()
            .enumerate()
            .map(|(i, n)| (*n, i))
            .collect();

        let attacker_action_to_graph = attack_indices.iter().enumerate().map(|(_, n)| *n).collect();

        let defender_action_to_graph = defense_indices
            .iter()
            .enumerate()
            .map(|(_, n)| *n)
            .collect();

        let initial_state = SimulatorState::new(&graph, config.seed, config.randomize_ttc)?;

        let sim = SimulatorRuntime {
            state: RefCell::new(initial_state),
            g: graph,
            false_negative_rate: config.false_negative_rate,
            false_positive_rate: config.false_positive_rate,
            config,
            ttc_sum: 0,
            graph_to_attack_state,
            graph_to_defense_state,
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

        let new_state =
            SimulatorState::new(&self.g, self.config.seed, self.config.randomize_ttc)?;

        let flag_status = get_flag_status(&self.g.flags, &new_state.compromised_steps);
        
        let result = Ok((
            self.map_state_to_observation(&new_state),
            new_state.to_info(self.g.attack_steps.len(), self.g.defense_steps.len(), flag_status),
        ));
        self.ttc_sum = new_state.remaining_ttc.iter().map(|(_, &ttc)| ttc).sum();
        self.history.clear();
        self.state.replace(new_state);
        return result;
    }

    pub fn total_ttc_remaining(&self) -> TTCType {
        return self.state.borrow().remaining_ttc.iter().map(|(_, &ttc)| ttc).sum();
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

    pub fn defense_action(
        &self,
        action: usize,
    ) -> SimResult<(HashSet<NodeID>, HashMap<NodeID, i32>)> {
        let state = self.state.borrow();
        let defense_step_id = self.defender_action_to_graph[action];

        if state.enabled_defenses.contains(&defense_step_id) {
            return Err(SimError {
                error: "Defense step not in defense surface.".to_string(),
            });
        }

        return self.enable_defense_step(defense_step_id);
    }

    pub fn work_on_attack_step(attack_step_id: NodeID) -> HashMap<NodeID, i32> {
        return HashMap::from([(attack_step_id, -1)]);
    }

    pub fn attack_action(&self, attack_surface_idx: usize) -> SimResult<HashMap<NodeID, i32>> {
        // Have the attacker perform an action.
                
        let attack_step_id = self.attacker_action_to_graph[attack_surface_idx];
        
        let state = self.state.borrow();
        let attack_surface_empty = state.attack_surface.is_empty();

        if attack_surface_empty {
            return Err(SimError {
                error: "Attack surface is empty.".to_string(),
            });
        }

        return Ok(SimulatorRuntime::work_on_attack_step(attack_step_id));
    }

    fn map_state_to_observation(&self, state: &SimulatorState) -> Observation {
        let mut ttc_remaining = vec![0; self.g.attack_steps.len()];
        state
            .remaining_ttc
            .iter()
            .map(|(node_id, &ttc)| (self.graph_to_attack_state[&node_id], ttc))
            .for_each(|(index, ttc)| {
                ttc_remaining[index] = ttc;
            });

        let mut attack_surface_vec = vec![false; self.g.attack_steps.len()];
        state
            .attack_surface
            .iter()
            .map(|node_id| self.graph_to_attack_state[&node_id])
            .for_each(|index| {
                attack_surface_vec[index] = true;
            });

        let mut defense_state = vec![true; self.g.defense_steps.len()];
        state
            .enabled_defenses
            .iter()
            .map(|node_id| self.graph_to_defense_state[&node_id])
            .for_each(|index| {
                defense_state[index] = false;
            });

        let mut attack_state = vec![false; self.g.attack_steps.len()];
        state
            .compromised_steps
            .iter()
            .map(|node_id| self.graph_to_attack_state[&node_id])
            .for_each(|index| {
                attack_state[index] = true;
            });

        let mut ids_observation = defense_state.clone();
        ids_observation.extend(attack_state.clone());

        let mut action_mask = vec![false; SPECIAL_ACTIONS.len()];
        
        action_mask[ACTION_NOP] = true;

        let mut defender_action_mask = action_mask.clone();
        let mut attacker_action_mask = action_mask.clone();
        
        defender_action_mask.extend(defense_state.clone());
        attacker_action_mask.extend(attack_surface_vec.clone());

        Observation {
            attack_surface: attack_surface_vec,
            defense_state: defense_state.clone(),
            defense_surface: defense_state,
            defender_action_mask,
            attacker_action_mask,
            ttc_remaining,
            ids_observation,
            attack_state,
        }
    }

    pub fn step(&mut self, action_dict: HashMap<String, usize>) -> SimResult<(Observation, Info)> {
        let new_state = self.calculate_next_state(action_dict)?;

        let flag_status = get_flag_status(&self.g.flags, &new_state.compromised_steps);

        let result = Ok((
            self.map_state_to_observation(&new_state),
            new_state.to_info(self.g.attack_steps.len(), self.g.defense_steps.len(), flag_status),
        ));

        self.history.push(self.state.replace(new_state));

        return result;
    }

    fn calculate_next_state(
        &self,
        action_dict: HashMap<String, usize>,
    ) -> SimResult<SimulatorState> {
        let old_state = self.state.borrow();

        let mut total_ttc_diff: HashMap<NodeID, i32> = HashMap::new();
        let mut ttc_diff: HashMap<NodeID, i32>;
        let mut new_defenses: HashSet<u64> = HashSet::new();

        // Attacker selects and attack step from the attack surface
        // Defender selects a defense step from the defense surface, which is the vector of all defense steps that are not disabled

        for (actor, action) in action_dict.iter() {
            
            if *action < SPECIAL_ACTIONS.len()  {
                continue;
            }

            let step_idx = *action - SPECIAL_ACTIONS.len();
            let actor_id = self.actors[actor];

            if actor_id == ATTACKER {
                ttc_diff = self.attack_action(step_idx)?;
            } else if actor_id == DEFENDER {
                (new_defenses, ttc_diff) = self.defense_action(step_idx)?;
            } else {
                return Err(SimError {
                    error: format!("Invalid actor: {}", actor),
                });
            }

            for (step_id, ttc) in ttc_diff.iter() {
                let current_ttc = total_ttc_diff.entry(*step_id).or_insert(0);
                *current_ttc += *ttc;
            }
        }

        let mut remaining_ttc: HashMap<NodeID, u64> = old_state.remaining_ttc.clone();
        for (step_id, ttc) in total_ttc_diff.iter() {
            let current_ttc = remaining_ttc.entry(*step_id).or_insert(0);
            *current_ttc = max(0, *current_ttc as i64 + *ttc as i64) as u64;
        }

        let mut enabled_defenses = old_state.enabled_defenses.clone();
        enabled_defenses.extend(new_defenses);

        let compromised_steps = calculate_compromised_steps(&self.g, &remaining_ttc);
        let attack_surface =
            calculate_attack_surface(&self.g, &compromised_steps, &enabled_defenses)?;

        Ok(SimulatorState {
            attack_surface,
            enabled_defenses,
            remaining_ttc,
            compromised_steps,
            time: old_state.time + 1,
            rng: old_state.rng.clone(),
            num_observed_alerts: 0,
            actions: action_dict,
        })
    }
}

fn get_flag_status(flags: &HashSet<u64>, compromised_steps: &HashSet<u64>) -> HashMap<u64, bool> {
    return flags.iter().map(|flag_id| {
        (*flag_id, compromised_steps.contains(flag_id))
    })
    .collect()
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
    use crate::{attackgraph, config, runtime::SimulatorRuntime};

    #[test]
    fn test_sim_init() {
        let filename = "four_ways.yaml";
        let graph = attackgraph::load_graph_from_yaml(filename);
        let config = config::SimulatorConfig::default();
        let sim = SimulatorRuntime::new(graph, config).unwrap();

        let initial_state = sim.state.borrow();

        assert_eq!(initial_state.enabled_defenses.len(), 0);
        assert_eq!(initial_state.compromised_steps.len(), 1);
        assert_eq!(
            initial_state
                .remaining_ttc
                .iter()
                .filter(|(_, &ttc)| ttc == 0)
                .count(),
            1
        );
        assert_eq!(initial_state.attack_surface.len(), 4);
    }

}
