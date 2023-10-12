use crate::attackgraph::AttackGraph;

use crate::state::SimulatorState;
use rand::Rng;
use rand_chacha::ChaChaRng;

use rand_distr::Standard;
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::hash::Hash;

pub(crate) struct DefenderObs<I> {
    // Possible actions in the given state
    pub possible_objects: HashSet<I>,
    // MAL stuff
    pub possible_steps: HashSet<String>,
    pub possible_assets: HashSet<String>,
    // Observations
    pub observed_steps: HashSet<I>,
}

impl<I> DefenderObs<I>
where
    I: Eq + Hash + Ord + Display + Copy + Debug,
{
    pub fn new(graph: &AttackGraph<I>, s: &SimulatorState<I>) -> Self {
        Self {
            possible_objects: Self::_defense_surface(graph, &s.enabled_defenses, None),
            observed_steps: Self::_defender_steps_observered(
                graph,
                &s.compromised_steps,
                &mut s.rng.clone(),
            ),
            possible_assets: Self::_defender_possible_assets(
                graph,
                &s.enabled_defenses,
                None,
            ),
            possible_steps: Self::_defender_possible_actions(
                graph,
                &s.enabled_defenses,
                None,
            ),
        }
    }

    pub fn _defender_steps_observered(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        rng: &mut ChaChaRng,
    ) -> HashSet<I> {
        graph
            .nodes()
            .iter()
            .map(|(i, _)| (i, compromised_steps.contains(i)))
            .zip(rng.sample_iter::<f64, Standard>(Standard))
            .filter_map(|((i, compromised), p)| match compromised {
                true => match p < graph.confusion_for_step(i).fnr {
                    // Sample Bernoulli(fnr_prob)
                    true => None,
                    false => Some(*i),
                },
                false => match p < graph.confusion_for_step(i).fpr {
                    // Sample Bernoulli(fpr_prob)
                    true => Some(*i),
                    false => None,
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
