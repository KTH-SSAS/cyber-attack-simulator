use crate::attackgraph::{AttackGraph, TTCType};
use crate::observation::Info;
use crate::runtime::SimResult;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rand_distr::Distribution;
use rand_distr::Exp;
use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;
#[derive(Debug, Clone)]
pub(crate) struct StateError {
    message: String,
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StateError: {}", self.message)
    }
}

impl std::error::Error for StateError {
    fn description(&self) -> &str {
        &self.message
    }
}

impl<I> AttackGraph<I>
where
    I: Eq + Hash + Ord + Copy + Debug,
{
    pub(crate) fn is_defended(&self, node: &I, enabled_defenses: &HashSet<I>) -> bool {
        self.get_defense_parents(node)
            .any(|d| enabled_defenses.contains(d))
    }

    fn parent_conditions_fulfilled(&self, compromised_steps: &HashSet<I>, node_id: &I) -> bool {
        let mut parent_states = self
            .get_attack_parents(node_id)
            .map(|p| compromised_steps.contains(p));
        return self
            .get_step_err(node_id)
            .can_be_compromised(&mut parent_states)
            || self.is_entry(node_id);
    }

    pub(crate) fn step_will_be_compromised(
        &self,
        compromised_steps: &HashSet<I>,
        enabled_defenses: &HashSet<I>,
        ttc_remaining: &HashMap<I, TTCType>,
        step: &I,
        attack_step: Option<&I>,
        defense_step: Option<&I>,
    ) -> bool
    where
        I: Eq + Hash + Ord + Copy + Debug,
    {
        compromised_steps.contains(step)
            || match attack_step {
                Some(a) => {
                    a == step                                                  // Attacker selects this step
                    && ttc_remaining[a] == 0                                      // TTC is 0
                    && !self.step_will_be_defended(a, enabled_defenses, defense_step)
                    && self.parent_conditions_fulfilled(compromised_steps, step)
                    // Parent step(s) are compromised
                    // It is not defended by a defense
                }
                None => false,
            }
    }
}

pub(crate) type StateResult<T> = std::result::Result<T, StateError>;

#[derive(Clone, Eq)]
pub(crate) struct SimulatorState<I>
where
    I: Hash,
{
    // Decomposed State
    pub time: u64,
    pub compromised_steps: HashSet<I>,
    pub enabled_defenses: HashSet<I>,
    pub remaining_ttc: HashMap<I, TTCType>,
    pub rng: ChaChaRng,
    pub _defender_action: Option<I>, // Action that the defender took in previous state
    pub _attacker_action: Option<I>, // Action that the attacker took in previous state
}

impl<I> PartialEq for SimulatorState<I>
where
    I: Eq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        self.compromised_steps == other.compromised_steps
            && self.enabled_defenses == other.enabled_defenses
            && self.remaining_ttc == other.remaining_ttc
    }
}

#[derive(Clone)]
pub struct SimulatorObs<I> {
    pub time: u64,
    pub compromised_steps: HashSet<I>,
    pub enabled_defenses: HashSet<I>,
    pub remaining_ttc: HashMap<I, TTCType>,
}

impl<I> From<&SimulatorState<I>> for SimulatorObs<I>
where
    I: Eq + Hash + Ord + Copy + Debug,
{
    fn from(state: &SimulatorState<I>) -> Self {
        SimulatorObs {
            time: state.time,
            compromised_steps: state.compromised_steps.clone(),
            enabled_defenses: state.enabled_defenses.clone(),
            remaining_ttc: state.remaining_ttc.clone(),
        }
    }
}

impl<I> Debug for SimulatorState<I>
where
    I: Debug + Hash,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = format!("Time: {}\n", self.time);
        s += &format!("Enabled defenses: {:?}\n", self.enabled_defenses);
        s += &format!("Compromised steps: {:?}\n", self.compromised_steps);
        s += &format!("Remaining TTC: {:?}\n", self.remaining_ttc);
        write!(f, "{}", s)
    }
}

impl<I> SimulatorState<I>
where
    I: Eq + Hash + Ord + Copy + Debug,
{
    pub fn new(
        graph: &AttackGraph<I>,
        seed: Option<u64>,
        randomize_ttc: bool,
    ) -> SimResult<SimulatorState<I>> {
        let mut rng = match seed {
            Some(s) => ChaChaRng::seed_from_u64(s),
            None => ChaChaRng::from_entropy(),
        };
        let ttc_params = graph.ttc_params();
        let remaining_ttc = match randomize_ttc {
            true => Self::get_initial_ttc_vals(&mut rng, &ttc_params),
            false => HashMap::from_iter(ttc_params),
        };

        let enabled_defenses = HashSet::new();
        let compromised_steps = graph.entry_points();

        Ok(SimulatorState {
            time: 0,
            enabled_defenses: Self::_enabled_defenses(graph, &enabled_defenses, None),
            remaining_ttc: Self::_remaining_ttc(
                graph,
                &remaining_ttc,
                &enabled_defenses,
                None,
                None,
            ),
            compromised_steps: Self::_compromised_steps(
                graph,
                &remaining_ttc,
                &compromised_steps,
                &enabled_defenses,
                None,
                None,
            ),
            _attacker_action: None,
            _defender_action: None,
            rng,
        })
    }

    pub fn attacker_reward(&self, graph: &AttackGraph<I>, valid_steps: &HashSet<I>, selected_step: Option<&I>) -> i64 {
        let wait_penalty = -1;

        if selected_step.is_none() {
            return wait_penalty;
        };

        if let Some(step) = selected_step {
            if !valid_steps.contains(step) {
                return wait_penalty;
            }
        }

        let flag_value = 1;
        let flag_reward = self.compromised_steps
            .iter()
            .filter_map(|x| match graph.flags.contains(&x) {
                true => Some(flag_value),
                false => None,
            })
            .sum();

        match flag_reward {
            0 => wait_penalty,
            _ => flag_reward,
        }
    }

    fn enabled_defenses(&self, graph: &AttackGraph<I>, selected_defense: Option<&I>) -> HashSet<I> {
        Self::_enabled_defenses(graph, &self.enabled_defenses, selected_defense)
    }

    fn defense_will_be_enabled(
        enabled_defenses: &HashSet<I>,
        node: &I,
        selected_node: Option<&I>,
    ) -> bool {
        enabled_defenses.contains(node)
            || match selected_node {
                Some(selected_node) => node == selected_node,
                None => false,
            }
    }

    fn _enabled_defenses(
        graph: &AttackGraph<I>,
        enabled_defenses: &HashSet<I>,
        selected_defense: Option<&I>,
    ) -> HashSet<I> {
        graph
            .defense_steps
            .iter()
            .filter(|x| Self::defense_will_be_enabled(enabled_defenses, x, selected_defense))
            .map(|x| *x)
            .collect::<HashSet<I>>()
    }

    pub fn total_ttc_remaining(&self) -> TTCType {
        return self.remaining_ttc.iter().map(|(_, &ttc)| ttc).sum();
    }

    pub(crate) fn get_new_state(
        &self,
        graph: &AttackGraph<I>,
        attacker_step: Option<&I>,
        defender_step: Option<&I>,
    ) -> StateResult<SimulatorState<I>> {
        let rng = self.rng.clone();
        Ok(SimulatorState {
            enabled_defenses: self.enabled_defenses(graph, defender_step),
            remaining_ttc: self.remaining_ttc(graph, attacker_step, defender_step),
            compromised_steps: self.compromised_steps(graph, attacker_step, defender_step),
            time: self.time + 1,
            rng,
            _attacker_action: attacker_step.copied(),
            _defender_action: defender_step.copied(),
        })
    }

    pub fn defender_reward(&self, graph: &AttackGraph<I>, defense_step: Option<&I>) -> i64 {
        let downtime_value = -1;
        //let restoration_cost = -2;
        let flag_value = -2;

        // If a flag is compromised, it costs to keep it compromised
        let r1: i64 = self
            .compromised_steps
            .iter()
            .filter_map(|x| match graph.flags.contains(&x) {
                true => Some(flag_value),
                false => None,
            })
            .sum();

        // If a defense is enabled, it costs to keep it enabled
        let r2: i64 = self
            .enabled_defenses
            .iter()
            .map(|_| {
                return downtime_value;
            })
            .sum();

        // If a step is compromised, it costs more to enable a defense for it
        /*         let r3: i64 = match defense_step {
            Some(step) => graph
                .children(step)
                .iter()
                .filter_map(|x| match self.compromised_steps.contains(&x.id) {
                    true => Some(restoration_cost),
                    false => None,
                })
                .sum(),
            None => 0,
        }; */

        return r1 + r2;
    }

    fn remaining_ttc(
        &self,
        graph: &AttackGraph<I>,
        selected_attack: Option<&I>,
        selected_defense: Option<&I>,
    ) -> HashMap<I, TTCType> {
        Self::_remaining_ttc(
            graph,
            &self.remaining_ttc,
            &self.enabled_defenses,
            selected_attack,
            selected_defense,
        )
    }

    fn ttc_will_be_decreased(
        graph: &AttackGraph<I>,
        node: &I,
        ttc: u64,
        enabled_defenses: &HashSet<I>,
        selected_attack: Option<&I>,
        selected_defense: Option<&I>,
    ) -> bool {
        match selected_attack {
            Some(a) => {
                a == node
                    && ttc > 0
                    && !graph.step_will_be_defended(node, enabled_defenses, selected_defense)
            }
            None => false,
        }
    }

    fn compromised_steps(
        &self,
        graph: &AttackGraph<I>,
        attacker_step: Option<&I>,
        defender_step: Option<&I>,
    ) -> HashSet<I> {
        Self::_compromised_steps(
            graph,
            &self.remaining_ttc,
            &self.compromised_steps,
            &self.enabled_defenses,
            attacker_step,
            defender_step,
        )
    }

    fn _remaining_ttc(
        graph: &AttackGraph<I>,
        remaining_ttc: &HashMap<I, TTCType>,
        enabled_defenses: &HashSet<I>,
        selected_attack: Option<&I>,
        selected_defense: Option<&I>,
    ) -> HashMap<I, TTCType> {
        remaining_ttc
            .iter()
            .map(|(id, ttc)| {
                let ttc = match Self::ttc_will_be_decreased(
                    graph,
                    id,
                    *ttc,
                    enabled_defenses,
                    selected_attack,
                    selected_defense,
                ) {
                    true => *ttc - 1,
                    false => *ttc,
                };
                (*id, ttc)
            })
            .collect::<HashMap<I, TTCType>>()
    }

    fn _compromised_steps(
        graph: &AttackGraph<I>,
        remaining_ttc: &HashMap<I, TTCType>,
        compromised_steps: &HashSet<I>,
        enabled_defenses: &HashSet<I>,
        attacker_step: Option<&I>,
        defender_step: Option<&I>,
    ) -> HashSet<I> {
        graph
            .attack_steps
            .iter()
            .filter(|step| {
                !graph.step_will_be_defended(step, enabled_defenses, defender_step)
                    && graph.step_will_be_compromised(
                        compromised_steps,
                        enabled_defenses,
                        remaining_ttc,
                        step,
                        attacker_step,
                        defender_step,
                    )
            })
            .map(|s| *s)
            .collect()
    }

    pub fn to_info(
        &self,
        num_attacks: usize,
        num_defenses: usize,
        flag_status: HashMap<I, bool>,
    ) -> Info {
        let num_compromised_steps = self.compromised_steps.len();
        let num_enabled_defenses = self.enabled_defenses.len();
        let num_compromised_flags = flag_status.iter().filter(|(_, v)| **v).count();

        Info {
            time: self.time,
            sum_ttc: self.total_ttc_remaining(),
            num_compromised_steps,
            perc_compromised_steps: num_compromised_steps as f64 / num_attacks as f64,
            perc_defenses_activated: num_enabled_defenses as f64 / num_defenses as f64,
            num_compromised_flags,
            perc_compromised_flags: num_compromised_flags as f64 / flag_status.len() as f64,
        }
    }

    pub fn get_initial_ttc_vals(
        rng: &mut ChaChaRng,
        ttc_params: &Vec<(I, TTCType)>,
    ) -> HashMap<I, TTCType> {
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
}
