use crate::attackgraph::{AttackGraph, TTCType};
use crate::observation::Info;
use crate::runtime::{Confusion, SimResult};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use rand_distr::Distribution;
use rand_distr::Exp;
use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::{Debug, Display};
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

pub(crate) struct SimulatorState<I> {
    pub time: u64,
    pub rng: ChaChaRng,
    pub enabled_defenses: HashSet<I>,
    pub compromised_steps: HashSet<I>,
    pub attack_surface: HashSet<I>,
    pub defense_surface: HashSet<I>,
    pub remaining_ttc: HashMap<I, TTCType>,
    pub defender_observed_steps: HashSet<I>,
    //pub false_alerts: HashSet<I>,
    //pub missed_alerts: HashSet<I>,
    //actions: HashMap<String, usize>,
}

impl<I> Debug for SimulatorState<I>
where
    I: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = format!("Time: {}\n", self.time);
        s += &format!("Enabled defenses: {:?}\n", self.enabled_defenses);
        s += &format!("Compromised steps: {:?}\n", self.compromised_steps);
        s += &format!("Attack surface: {:?}\n", self.attack_surface);
        s += &format!("Remaining TTC: {:?}\n", self.remaining_ttc);
        s += &format!(
            "Steps observed as compromised by defender: {:?}\n",
            self.defender_observed_steps
        );
        write!(f, "{}", s)
    }
}

impl<I> SimulatorState<I>
where
    I: Eq + Hash + Ord + Display + Copy + Debug,
{
    pub fn new(
        graph: &AttackGraph<I>,
        confusion_per_step: &HashMap<I, Confusion>,
        seed: u64,
        randomize_ttc: bool,
    ) -> SimResult<SimulatorState<I>> {
        let mut rng = ChaChaRng::seed_from_u64(seed);
        let ttc_params = graph.ttc_params();
        let remaining_ttc = match randomize_ttc {
            true => Self::get_initial_ttc_vals(&mut rng, &ttc_params),
            false => HashMap::from_iter(ttc_params),
        };

        let enabled_defenses = HashSet::new();
        let compromised_steps = graph.entry_points();

        let p = rng.gen();

        Ok(SimulatorState {
            time: 0,
            rng,
            defense_surface: Self::_defense_surface(graph, &enabled_defenses, None),
            attack_surface: Self::_attack_surface(
                graph,
                &compromised_steps,
                &remaining_ttc,
                &enabled_defenses,
                None,
                None,
            ),
            enabled_defenses: Self::_enabled_defenses(graph, &enabled_defenses, None),
            remaining_ttc: Self::_remaining_ttc(graph, &remaining_ttc, None, None),
            compromised_steps: Self::_compromised_steps(
                graph,
                &remaining_ttc,
                &compromised_steps,
                &enabled_defenses,
                None,
                None,
            ),
            defender_observed_steps: Self::_defender_steps_observered(
                graph,
                &compromised_steps,
                confusion_per_step,
                p,
            ),
        })
    }

    pub fn total_ttc_remaining(&self) -> TTCType {
        return self.remaining_ttc.iter().map(|(_, &ttc)| ttc).sum();
    }

    pub fn export_rng(&self) -> ChaChaRng {
        self.rng.clone()
    }

    fn should_defense_be_enabled(
        enabled_defenses: HashSet<&I>,
        node: &I,
        selected_node: Option<&I>,
    ) -> bool {
        enabled_defenses.contains(node)
            || match selected_node {
                Some(selected_node) => node == selected_node,
                None => false,
            }
    }

    pub(crate) fn defender_steps_observered(
        &self,
        graph: &AttackGraph<I>,
        confusion_per_step: &HashMap<I, Confusion>,
        p: f64,
    ) -> HashSet<I> {
        Self::_defender_steps_observered(graph, &self.compromised_steps, confusion_per_step, p)
    }

    fn _defender_steps_observered(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        confusion_per_step: &HashMap<I, Confusion>,
        p: f64,
    ) -> HashSet<I> {
        graph
            .nodes()
            .iter()
            .map(|(i, _)| (i, compromised_steps.contains(i)))
            .map(|(i, compromised)| match compromised {
                true => (i, p < confusion_per_step[i].fnr),
                false => (i, p < confusion_per_step[i].fpr),
            })
            .filter_map(|(i, x)| match x {
                true => Some(*i),
                false => None,
            })
            .collect()
    }

    pub(crate) fn enabled_defenses(
        &self,
        graph: &AttackGraph<I>,
        selected_defense: Option<&I>,
    ) -> HashSet<I> {
        Self::_enabled_defenses(graph, &self.enabled_defenses, selected_defense)
    }

    fn _enabled_defenses(
        graph: &AttackGraph<I>,
        enabled_defenses: &HashSet<I>,
        selected_defense: Option<&I>,
    ) -> HashSet<I> {
        graph
            .defense_steps
            .iter()
            .filter_map(|x| {
                match Self::should_defense_be_enabled(
                    enabled_defenses.iter().collect(),
                    x,
                    selected_defense,
                ) {
                    true => Some(x.clone()),
                    false => None,
                }
            })
            .collect::<HashSet<I>>()
    }

    fn should_ttc_be_decreased(
        graph: &AttackGraph<I>,
        node: &I,
        ttc: u64,
        selected_attack: Option<&I>,
        selected_defense: Option<&I>,
    ) -> bool {
        match (selected_attack, selected_defense) {
            (Some(a), Some(d)) => a == node && ttc > 0 && !graph.step_is_defended_by(node, d),
            (Some(a), None) => a == node && ttc > 0,
            (None, Some(_)) => false,
            (None, None) => false,
        }
    }

    pub(crate) fn remaining_ttc(
        &self,
        graph: &AttackGraph<I>,
        selected_attack: Option<&I>,
        selected_defense: Option<&I>,
    ) -> HashMap<I, TTCType> {
        Self::_remaining_ttc(
            graph,
            &self.remaining_ttc,
            selected_attack,
            selected_defense,
        )
    }

    fn _remaining_ttc(
        graph: &AttackGraph<I>,
        remaining_ttc: &HashMap<I, TTCType>,
        selected_attack: Option<&I>,
        selected_defense: Option<&I>,
    ) -> HashMap<I, TTCType> {
        remaining_ttc
            .iter()
            .map(|(id, ttc)| {
                let ttc = match Self::should_ttc_be_decreased(
                    graph,
                    id,
                    *ttc,
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

    pub(crate) fn uncompromised_steps(&self, graph: &AttackGraph<I>) -> HashSet<I> {
        graph
            .attack_steps
            .difference(&self.compromised_steps)
            .map(|&x| x)
            .collect()
    }

    pub(crate) fn attack_surface(
        &self,
        graph: &AttackGraph<I>,
        defender_action: Option<&I>,
        attacker_action: Option<&I>,
    ) -> HashSet<I> {
        return Self::_attack_surface(
            graph,
            &self.compromised_steps,
            &self.remaining_ttc,
            &self.enabled_defenses,
            defender_action,
            attacker_action,
        );
    }

    fn _attack_surface(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        ttc_remaining: &HashMap<I, TTCType>,
        enabled_defenses: &HashSet<I>,
        defender_action: Option<&I>,
        attacker_action: Option<&I>,
    ) -> HashSet<I> {
        let attack_surface: HashSet<I> = graph
            .attack_steps
            .iter()
            .filter_map(|n| {
                match Self::is_in_attack_surface(
                    graph,
                    compromised_steps,
                    ttc_remaining,
                    enabled_defenses,
                    n,
                    defender_action,
                    attacker_action,
                ) {
                    true => Some(n),
                    false => None,
                }
            })
            .map(|n| *n)
            .collect();

        return attack_surface;
    }

    pub(crate) fn can_step_be_compromised(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        ttc_remaining: &HashMap<I, TTCType>,
        step: &I,
        attack_step: Option<&I>,
        defense_step: Option<&I>,
    ) -> bool {
        let parent_conditions_fulfilled =
            Self::parent_conditions_fulfilled(graph, compromised_steps, step);

        match (attack_step, defense_step) {
            (Some(a), Some(d)) => {
                parent_conditions_fulfilled
                    && a == step
                    && ttc_remaining[a] == 0
                    && !graph.step_is_defended_by(a, d)
            }
            (Some(a), None) => parent_conditions_fulfilled && a == step && ttc_remaining[a] == 0,
            (None, _) => false,
        }
    }

    pub(crate) fn defense_surface(
        &self,
        graph: &AttackGraph<I>,
        defender_action: Option<&I>,
    ) -> HashSet<I> {
        Self::_defense_surface(graph, &self.enabled_defenses, defender_action)
    }

    fn defense_available(
        step: &I,
        enabled_defenses: &HashSet<I>,
        defender_action: Option<&I>,
    ) -> bool {
        !enabled_defenses.contains(step)
            && match defender_action {
                Some(d) => d != step,
                None => true,
            }
    }

    fn _defense_surface(
        graph: &AttackGraph<I>,
        enabled_defenses: &HashSet<I>,
        defender_action: Option<&I>,
    ) -> HashSet<I> {
        graph
            .defense_steps
            .iter()
            .filter_map(
                |d| match Self::defense_available(d, enabled_defenses, defender_action) {
                    true => Some(*d),
                    false => None,
                },
            )
            .collect()
    }

    pub(crate) fn compromised_steps(
        &self,
        graph: &AttackGraph<I>,
        attacker_action: Option<&I>,
        defender_action: Option<&I>,
    ) -> HashSet<I> {
        Self::_compromised_steps(
            graph,
            &self.remaining_ttc,
            &self.compromised_steps,
            &self.enabled_defenses,
            attacker_action,
            defender_action,
        )
    }

    fn is_defended(graph: &AttackGraph<I>, node: &I, enabled_defenses: &HashSet<I>) -> bool {
        graph
            .get_defense_parents(node)
            .iter()
            .any(|d| enabled_defenses.contains(d))
    }

    fn _compromised_steps(
        graph: &AttackGraph<I>,
        remaining_ttc: &HashMap<I, TTCType>,
        compromised_steps: &HashSet<I>,
        enabled_defenses: &HashSet<I>,
        attacker_action: Option<&I>,
        defender_action: Option<&I>,
    ) -> HashSet<I> {
        graph
            .attack_steps
            .iter()
            .filter_map(|step| {
                let already_compromised = compromised_steps.contains(step);
                let defended = Self::is_defended(graph, step, enabled_defenses);
                let will_be_defended = match defender_action {
                    Some(d) => graph.step_is_defended_by(step, d),
                    None => false,
                };
                match !(defended || will_be_defended) && already_compromised
                    || Self::can_step_be_compromised(
                        graph,
                        compromised_steps,
                        remaining_ttc,
                        step,
                        attacker_action,
                        defender_action,
                    ) {
                    true => Some(*step),
                    false => None,
                }
            })
            .collect()
    }

    fn is_in_attack_surface(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        ttc_remaining: &HashMap<I, TTCType>,
        enabled_defenses: &HashSet<I>,
        node_id: &I,
        defender_action: Option<&I>,
        attacker_action: Option<&I>,
    ) -> bool {
        // Returns true if a node can be attacked given the current state of the
        // graph meaning that the
        let parent_states: Vec<bool> = graph
            .get_attack_parents(node_id)
            .iter()
            .map(|&p| {
                compromised_steps.contains(p)
                    || match attacker_action {
                        Some(a) => {
                            a == p
                                && Self::can_step_be_compromised(
                                    graph,
                                    compromised_steps,
                                    ttc_remaining,
                                    a,
                                    attacker_action,
                                    defender_action,
                                )
                        }
                        None => false,
                    }
            })
            .collect();

        let parent_conditions_fulfilled = graph
            .get_step(node_id)
            .unwrap()
            .can_be_compromised(&parent_states);

        let compromised = compromised_steps.contains(node_id);

        let defended = Self::is_defended(graph, node_id, enabled_defenses);

        let will_be_defended = match defender_action {
            Some(d) => graph.step_is_defended_by(node_id, d),
            None => false,
        };

        let will_be_attacked = match attacker_action {
            Some(a) => a == node_id,
            None => false,
        };

        return parent_conditions_fulfilled
            && !(compromised || defended || will_be_defended || will_be_attacked);
    }

    fn parent_conditions_fulfilled(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        node_id: &I,
    ) -> bool {
        let attack_parents: HashSet<&I> = graph.get_attack_parents(node_id);

        if attack_parents.is_empty() {
            return graph.is_entry(node_id);
        }

        let parent_states: Vec<bool> = attack_parents
            .iter()
            .map(|&p| compromised_steps.contains(p))
            .collect();

        return graph
            .get_step(node_id)
            .unwrap()
            .can_be_compromised(&parent_states);
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

    pub fn attacker_reward(&self, graph: &AttackGraph<I>) -> i64 {
        let flag_value = 1;
        self.compromised_steps
            .iter()
            .filter_map(|x| match graph.flags.contains(&x) {
                true => Some(flag_value),
                false => None,
            })
            .sum()
    }

    pub fn defender_reward(&self, graph: &AttackGraph<I>, defense_step: Option<&I>) -> i64 {
        let downtime_value = -1;
        let restoration_cost = -2;
        let flag_value = -3;

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
        let r3: i64 = match defense_step {
            Some(step) => graph
                .children(step)
                .iter()
                .filter_map(|x| match self.compromised_steps.contains(&x.id) {
                    true => Some(restoration_cost),
                    false => None,
                })
                .sum(),
            None => 0,
        };

        return r1 + r2 + r3;
    }
}
