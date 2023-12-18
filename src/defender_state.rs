use crate::attackgraph::AttackGraph;

use crate::state::SimulatorState;
use rand::Rng;
use rand_chacha::ChaChaRng;

use rand_distr::Standard;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

pub struct DefenderObs<I> {
    // Possible actions in the given state
    pub possible_objects: HashSet<I>,
    pub possible_actions: HashSet<String>,
    // MAL stuff
    pub possible_steps: HashSet<String>,
    pub possible_assets: HashSet<String>,
    // Observations
    pub observed_steps: HashMap<I, bool>,
}

impl<I> DefenderObs<I>
where
    I: Eq + Hash + Ord + Copy + Debug,
{
    pub(crate) fn steps_observed_as_compromised(&self, graph: &AttackGraph<I>) -> HashSet<I> {
        self.observed_steps
            .iter()
            .filter(|(k, _)| !graph.is_defense(*k)) // Do not consider enabled defenses as compromised
            .filter_map(|(k, v)| if *v { Some(*k) } else { None })
            .collect()
    }

    pub(crate) fn new(s: &SimulatorState<I>, graph: &AttackGraph<I>) -> Self {
        let all_actions = HashMap::from([
            ("wait".to_string(), true), // wait
            (
                "use".to_string(),
                graph.disabled_defenses(&s.enabled_defenses).len() > 0,
            ), // can use as long as there are disabled defenses
        ]);

        Self {
            possible_objects: Self::_defense_surface(&graph, &s.enabled_defenses, None),
            possible_actions: all_actions
                .iter()
                .filter_map(|(k, v)| if *v { Some(k.clone()) } else { None })
                .collect(),
            observed_steps: Self::_defender_steps_observered(
                &graph,
                &s.compromised_steps,
                &s.enabled_defenses,
                &mut s.rng.clone(),
            ),
            possible_assets: Self::_defender_possible_assets(&graph, &s.enabled_defenses, None),
            possible_steps: Self::_defender_possible_actions(&graph, &s.enabled_defenses, None),
        }
    }

    pub(crate) fn _defender_steps_observered(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        enabled_defenses: &HashSet<I>,
        rng: &mut ChaChaRng,
    ) -> HashMap<I, bool> {
        let is_observable = { |s| true }; // Defender observes all steps, but the states may be false
        let is_false_obs = |(p, rate)| p < rate;

        graph
            .nodes()
            .iter()
            .map(|(i, _)| {
                (
                    i,
                    compromised_steps.contains(i) || enabled_defenses.contains(i),
                )
            })
            .zip(rng.sample_iter::<f64, Standard>(Standard))
            .filter_map(|s| if is_observable(s) { Some(s) } else { None }) // Step is observed
            .map(|((i, step_state), p)| {
                let cause_false_negative = is_false_obs((p, graph.confusion_for_step(i).fnr));
                let cause_false_positive = is_false_obs((p, graph.confusion_for_step(i).fpr));
                match (step_state, cause_false_negative, cause_false_positive) {
                    (true, true, _) => (*i, false),   // False negative
                    (true, false, _) => (*i, true),   // True positive
                    (false, _, true) => (*i, true),   // False positive
                    (false, _, false) => (*i, false), // True negative
                }
            })
            .collect()
    }

    fn defense_will_be_available(
        step: &I,
        enabled_defenses: &HashSet<I>,
        defender_step: Option<&I>,
    ) -> bool {
        !enabled_defenses.contains(step)
            && if let Some(d) = defender_step {
                d != step
            } else {
                true
            }
    }

    pub(crate) fn _defense_surface(
        graph: &AttackGraph<I>,
        enabled_defenses: &HashSet<I>,
        defender_step: Option<&I>,
    ) -> HashSet<I> {
        graph
            .defense_steps
            .iter()
            .filter(|d| Self::defense_will_be_available(d, enabled_defenses, defender_step))
            .map(|d| *d)
            .collect()
    }

    pub(crate) fn _defender_possible_assets(
        graph: &AttackGraph<I>,
        enabled_defenses: &HashSet<I>,
        defender_step: Option<&I>,
    ) -> HashSet<String> {
        Self::_defense_surface(graph, enabled_defenses, defender_step)
            .iter()
            .filter_map(|x| match graph.get_step(x) {
                Some(step) => Some(step.asset()),
                None => None,
            })
            .collect::<HashSet<String>>()
    }

    pub(crate) fn _defender_possible_actions(
        graph: &AttackGraph<I>,
        enabled_defenses: &HashSet<I>,
        defender_step: Option<&I>,
    ) -> HashSet<String> {
        Self::_defense_surface(graph, enabled_defenses, defender_step)
            .iter()
            .filter_map(|x| match graph.get_step(x) {
                Some(step) => Some(step.name.clone()),
                None => None,
            })
            .collect::<HashSet<String>>()
            .union(&HashSet::from_iter(vec!["wait".to_string()]))
            .cloned()
            .collect()
    }
}
