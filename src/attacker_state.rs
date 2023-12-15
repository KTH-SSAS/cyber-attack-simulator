use crate::attackgraph::{AttackGraph, TTCType};

use crate::state::SimulatorState;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

pub struct AttackerObs<I> {
    // Possible actions in the given state
    pub possible_objects: HashSet<I>,
    pub possible_actions: HashSet<String>,
    // MAL stuff
    pub possible_steps: HashSet<String>,
    pub possible_assets: HashSet<String>,
    // Observations
    pub observed_steps: HashMap<I, bool>,
    pub reward: i64,
}

impl<I> AttackerObs<I>
where
    I: Eq + Hash + Ord + Copy + Debug,
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

            possible_assets: Self::_attacker_possible_assets(&graph, &attack_surface),

            possible_steps: Self::_attacker_possible_actions(&graph, &attack_surface),
            reward: s.attacker_reward(&graph, None),
            possible_objects: attack_surface,
        }
    }

    fn step_will_be_in_attack_surface(
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
        let parent_states = graph.get_attack_parents(node_id).map(|p| {
            graph.step_will_be_compromised(
                compromised_steps,
                enabled_defenses,
                ttc_remaining,
                p,
                attacker_step,
                defender_step,
            )
        });

        let parent_conditions_fulfilled = graph
            .get_step(node_id)
            .unwrap()
            .can_be_compromised(parent_states);


        return parent_conditions_fulfilled
            && !(graph.step_will_be_compromised(
                compromised_steps,
                enabled_defenses,
                ttc_remaining,
                node_id,
                attacker_step,
                defender_step,
            ));
    }

    pub(crate) fn _step_will_be_observed(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        enabled_defenses: &HashSet<I>,
        node_id: &I,
        ttc_remaining: &HashMap<I, TTCType>,
        attacker_step: Option<&I>,
        defender_step: Option<&I>,
    ) -> bool {
        let parent_states = graph.get_attack_parents(node_id).map(|p| {
            graph.step_will_be_compromised(
                compromised_steps,
                enabled_defenses,
                ttc_remaining,
                p,
                attacker_step,
                defender_step,
            )
        });

        let parent_conditions_fulfilled = graph
            .get_step(node_id)
            .unwrap()
            .can_be_compromised(parent_states);

        let compromised = compromised_steps.contains(node_id);

        return parent_conditions_fulfilled || compromised;
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
                match Self::step_will_be_in_attack_surface(
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
    ) -> HashMap<I, bool> {
        graph
            .attack_steps
            .iter()
            .filter_map(|s| {
                match Self::_step_will_be_observed(
                    graph,
                    compromised_steps,
                    enabled_defenses,
                    s,
                    remaining_ttc,
                    attacker_step,
                    defender_step,
                ) {
                    false => None,
                    true => Some((*s, compromised_steps.contains(s))),
                }
            })
            .collect()
    }

    pub(crate) fn _attacker_possible_assets(
        graph: &AttackGraph<I>,
        attack_surface: &HashSet<I>,
    ) -> HashSet<String> {
        attack_surface
            .iter()
            .filter_map(|x| match graph.get_step(x) {
                Ok(step) => Some(step.asset()),
                Err(_e) => None,
            })
            .collect::<HashSet<String>>()
    }

    pub(crate) fn _attacker_possible_actions(
        graph: &AttackGraph<I>,
        attack_surface: &HashSet<I>,
    ) -> HashSet<String> {
        attack_surface
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
