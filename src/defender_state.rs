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
    
    pub(crate) fn defender_steps_observered(
        &self,
        graph: &AttackGraph<I>,
        confusion_per_step: &HashMap<I, Confusion>,
        rng: &mut ChaChaRng,
    ) -> HashSet<I> {
        Self::_defender_steps_observered(graph, &self.compromised_steps, confusion_per_step, rng)
    }


    pub(crate) fn _defender_steps_observered(
        graph: &AttackGraph<I>,
        compromised_steps: &HashSet<I>,
        confusion_per_step: &HashMap<I, Confusion>,
        rng: &mut ChaChaRng,
    ) -> HashSet<I> {
        graph
            .nodes()
            .iter()
            .map(|(i, _)| (i, compromised_steps.contains(i)))
            .zip(rng.sample_iter::<f64, Standard>(Standard))
            .filter_map(|((i, compromised), p)| match compromised {
                true => match p < confusion_per_step[i].fnr {
                    // Sample Bernoulli(fnr_prob)
                    true => None,
                    false => Some(*i),
                },
                false => match p < confusion_per_step[i].fpr {
                    // Sample Bernoulli(fpr_prob)
                    true => Some(*i),
                    false => None,
                },
            })
            .collect()
    }

    pub(crate) fn defense_surface(&self, graph: &AttackGraph<I>, defender_step: Option<&I>) -> HashSet<I> {
        Self::_defense_surface(graph, &self.enabled_defenses, defender_step)
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

    pub(crate) fn _defender_possible_assets(
        graph: &AttackGraph<I>,
        enabled_defenses: &HashSet<I>,
        defender_step: Option<&I>,
        attacker_step: Option<&I>,
    ) -> HashSet<String> {
        Self::_defense_surface(graph, enabled_defenses, defender_step)
            .iter()
            .filter_map(|x| match graph.get_step(x) {
                Ok(step) => Some(step.asset()),
                Err(e) => None,
            })
            .collect::<HashSet<String>>()
    }

    pub(crate) fn _defender_possible_actions(
        graph: &AttackGraph<I>,
        enabled_defenses: &HashSet<I>,
        defender_step: Option<&I>,
        attacker_step: Option<&I>,
    ) -> HashSet<String> {
        Self::_defense_surface(graph, enabled_defenses, defender_step)
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
