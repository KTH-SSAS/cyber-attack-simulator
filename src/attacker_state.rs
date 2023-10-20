use crate::attackgraph::{AttackGraph, TTCType};

use crate::state::SimulatorState;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;

pub struct AttackerObs<I> {
    // Possible actions in the given state
    pub possible_objects: HashSet<I>,
    pub possible_actions: HashSet<String>,
    // MAL stuff
    pub possible_steps: HashSet<String>,
    pub possible_assets: HashSet<String>,
    // Observations
    pub observed_steps: HashSet<I>,
    pub reward: i64,
}

impl<I> AttackerObs<I>
where
    I: Eq + Hash + Ord + Display + Copy + Debug,
{
    pub(crate) fn new(s: &SimulatorState<I>, graph: &AttackGraph<I>) -> Self {
        let attack_surface = Self::_attack_surface(
            &graph,
            &s.compromised_steps,
            &s.remaining_ttc,
            &s.enabled_defenses,
            None,
            None,
        );

        let all_actions = HashMap::from([
            ("wait".to_string(), false),
            ("use".to_string(), attack_surface.len() > 0),
        ]);

        Self {
            possible_objects: attack_surface,
            possible_actions: all_actions
                .iter()
                .filter_map(|(k, v)| match v {
                    true => Some(k.clone()),
                    false => None,
                })
                .collect(),
            observed_steps: Self::_attacker_steps_observed(
                &graph,
                &s.compromised_steps,
                &s.remaining_ttc,
                &s.enabled_defenses,
                None,
                None,
            ),

            possible_assets: Self::_attacker_possible_assets(
                &graph,
                &s.compromised_steps,
                &s.remaining_ttc,
                &s.enabled_defenses,
                None,
                None,
            ),

            possible_steps: Self::_attacker_possible_actions(
                &graph,
                &s.compromised_steps,
                &s.remaining_ttc,
                &s.enabled_defenses,
                None,
                None,
            ),
            reward: s.attacker_reward(&graph),
        }
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
                                && graph.can_step_be_compromised(
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

        let defended = graph.is_defended(node_id, enabled_defenses);

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
            Err(_e) => None,
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
            Err(_e) => None,
        })
        .collect::<HashSet<String>>()
        .union(&HashSet::from_iter(vec!["wait".to_string()]))
        .cloned()
        .collect()
    }
}
