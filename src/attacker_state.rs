use crate::attackgraph::{AttackGraph, TTCType};
use crate::observation::Info;
use crate::runtime::{Confusion, SimResult};
use crate::state::SimulatorState;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use rand_distr::Exp;
use rand_distr::{Distribution, Standard};
use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;

impl<I> SimulatorState<I>
where
    I: Eq + Hash + Ord + Display + Copy + Debug,
{

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

	pub(crate) fn attack_surface(
        &self,
        graph: &AttackGraph<I>,
        defender_step: Option<&I>,
        attacker_step: Option<&I>,
    ) -> HashSet<I> {
        return Self::_attack_surface(
            graph,
            &self.compromised_steps,
            &self.remaining_ttc,
            &self.enabled_defenses,
            defender_step,
            attacker_step,
        );
    }

	fn is_in_attack_surface(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        ttc_remaining: &HashMap<I, TTCType>,
        enabled_defenses: &HashSet<I>,
        node_id: &I,
        defender_step: Option<&I>,
        attacker_step: Option<&I>,
    ) -> bool {
        // Returns true if a node can be attacked given the current state of the
        // graph meaning that the
        let parent_states: Vec<bool> = graph
            .get_attack_parents(node_id)
            .iter()
            .map(|&p| {
                compromised_steps.contains(p)
                    || match attacker_step {
                        Some(a) => {
                            a == p
                                && Self::can_step_be_compromised(
                                    graph,
                                    compromised_steps,
                                    ttc_remaining,
                                    a,
                                    attacker_step,
                                    defender_step,
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

        let will_be_defended = match defender_step {
            Some(d) => graph.step_is_defended_by(node_id, d),
            None => false,
        };

        let will_be_attacked = match attacker_step {
            Some(a) => a == node_id,
            None => false,
        };

        return parent_conditions_fulfilled
            && !(compromised || defended || will_be_defended || will_be_attacked);
    }

	pub(crate) fn _attack_surface(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        ttc_remaining: &HashMap<I, TTCType>,
        enabled_defenses: &HashSet<I>,
        defender_step: Option<&I>,
        attacker_step: Option<&I>,
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
                    defender_step,
                    attacker_step,
                ) {
                    true => Some(n),
                    false => None,
                }
            })
            .map(|n| *n)
            .collect();

        return attack_surface;
    }

	pub(crate) fn _attacker_steps_observed(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        remaining_ttc: &HashMap<I, u64>,
        enabled_defenses: &HashSet<I>,
        defender_step: Option<&I>,
        attacker_step: Option<&I>,
    ) -> HashSet<I> {
        return Self::_attack_surface(
            graph,
            compromised_steps,
            remaining_ttc,
            enabled_defenses,
            defender_step,
            attacker_step,
        )
        .union(&compromised_steps)
        .cloned()
        .collect();
    }

    pub(crate) fn _attacker_possible_assets(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        remaining_ttc: &HashMap<I, u64>,
        enabled_defenses: &HashSet<I>,
        defender_step: Option<&I>,
        attacker_step: Option<&I>,
    ) -> HashSet<String> {
        Self::_attack_surface(
            graph,
            compromised_steps,
            remaining_ttc,
            enabled_defenses,
            defender_step,
            attacker_step,
        )
        .iter()
        .filter_map(|x| match graph.get_step(x) {
            Ok(step) => Some(step.asset()),
            Err(e) => None,
        })
        .collect::<HashSet<String>>()
    }

    pub(crate) fn _attacker_possible_actions(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        remaining_ttc: &HashMap<I, u64>,
        enabled_defenses: &HashSet<I>,
        defender_step: Option<&I>,
        attacker_step: Option<&I>,
    ) -> HashSet<String> {
        Self::_attack_surface(
            graph,
            compromised_steps,
            remaining_ttc,
            enabled_defenses,
            defender_step,
            attacker_step,
        )
        .iter()
        .filter_map(|x| match graph.get_step(x) {
            Ok(step) => Some(step.name.clone()),
            Err(e) => None,
        })
        .collect::<HashSet<String>>()
        .union(&HashSet::from_iter(vec!["wait".to_string()]))
        .cloned()
        .collect()
    }
}