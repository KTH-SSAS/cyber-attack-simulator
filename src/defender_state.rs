use crate::attackgraph::AttackGraph;

use crate::state::SimulatorState;
use rand::Rng;
use rand_chacha::ChaChaRng;

use rand_distr::Standard;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
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
    I: Eq + Hash + Ord + Display + Copy + Debug,
{

    pub(crate) fn steps_observed_as_compromised(&self) -> HashSet<I> {
        self.observed_steps
            .iter()
            .filter_map(|(k, v)| match v {
                true => Some(*k),
                false => None,
            })
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
                .filter_map(|(k, v)| match v {
                    true => Some(k.clone()),
                    false => None,
                })
                .collect(),
            observed_steps: Self::_defender_steps_observered(
                &graph,
                &s.compromised_steps,
                &mut s.rng.clone(),
            ),
            possible_assets: Self::_defender_possible_assets(&graph, &s.enabled_defenses, None),
            possible_steps: Self::_defender_possible_actions(&graph, &s.enabled_defenses, None),
        }
    }

    pub(crate) fn _defender_steps_observered(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        rng: &mut ChaChaRng,
    ) -> HashMap<I, bool> {
        // Defender observes all steps with potential false positives
        
        let is_obs = true;
        let is_false_obs = |(p, rate)| p < rate;
        
        graph
            .nodes()
            .iter()
            .map(|(i, _)| (i, compromised_steps.contains(i)))
            .zip(rng.sample_iter::<f64, Standard>(Standard))
            .filter_map(|s| match is_obs {true => Some(s), false => None}) // Step is observed
            .map(|((i, compromised), p)| match compromised {
                true => match is_false_obs((p, graph.confusion_for_step(i).fnr)) {
                    // Sample Bernoulli(fnr_prob)
                    true => (*i, false), // We observe the step, but it is reported as a false negative
                    false => (*i, compromised),
                },
                false => match is_false_obs((p, graph.confusion_for_step(i).fpr)) {
                    // Sample Bernoulli(fpr_prob)
                    true => (*i, compromised),
                    false => (*i, true), // We observe the step, but it is reported as a false positive
                },
            })
            .collect()
    }

    fn defense_available(
        step: &I,
        enabled_defenses: &HashSet<I>,
        defender_step: Option<&I>,
    ) -> bool {
        !enabled_defenses.contains(step)
            && match defender_step {
                Some(d) => d != step,
                None => true,
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
            .filter_map(
                |d| match Self::defense_available(d, enabled_defenses, defender_step) {
                    true => Some(*d),
                    false => None,
                },
            )
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
                Ok(step) => Some(step.asset()),
                Err(_e) => None,
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
                Ok(step) => Some(step.name.clone()),
                Err(_e) => None,
            })
            .collect::<HashSet<String>>()
            .union(&HashSet::from_iter(vec!["wait".to_string()]))
            .cloned()
            .collect()
    }
}
