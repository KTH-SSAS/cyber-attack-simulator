use crate::attackgraph::{AttackGraph, TTCType};
use crate::observation::Info;
use crate::runtime::{ActionResult, SimError, SimResult};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rand_distr::Distribution;
use rand_distr::Exp;
use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Debug};
use std::hash::Hash;

pub(crate) struct SimulatorState<I> {
    pub time: u64,
    pub rng: ChaChaRng,
    pub enabled_defenses: HashSet<I>,
    pub compromised_steps: HashSet<I>,
    pub attack_surface: HashSet<I>,
    pub remaining_ttc: HashMap<I, TTCType>,
    pub num_observed_alerts: usize,
    pub false_alerts: HashSet<I>,
    pub missed_alerts: HashSet<I>,
    //actions: HashMap<String, usize>,
}

impl<I> Debug for SimulatorState<I> where I: Debug {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let mut s = format!("Time: {}\n", self.time);
		s += &format!("Enabled defenses: {:?}\n", self.enabled_defenses);
		s += &format!("Compromised steps: {:?}\n", self.compromised_steps);
		s += &format!("Attack surface: {:?}\n", self.attack_surface);
		s += &format!("Remaining TTC: {:?}\n", self.remaining_ttc);
		s += &format!("Num observed alerts: {}\n", self.num_observed_alerts);
		s += &format!("False alerts: {:?}\n", self.false_alerts);
		s += &format!("Missed alerts: {:?}\n", self.missed_alerts);
		write!(f, "{}", s)
	}
}

impl<I> SimulatorState<I>
where
    I: Eq + Hash + Ord + Display + Copy + Debug,
{
    pub fn new(
        graph: &AttackGraph<I>,
        seed: u64,
        randomize_ttc: bool,
    ) -> SimResult<SimulatorState<I>> {
        let mut rng = ChaChaRng::seed_from_u64(seed);
        let enabled_defenses = HashSet::new();
        let ttc_params = graph.ttc_params();
        let remaining_ttc = match randomize_ttc {
            true => Self::get_initial_ttc_vals(&mut rng, &ttc_params),
            false => HashMap::from_iter(ttc_params),
        };
        let compromised_steps = graph.entry_points();
        let attack_surface =
            match graph.calculate_attack_surface(&compromised_steps, &HashSet::new()) {
                Ok(attack_surface) => attack_surface,
                Err(e) => {
                    return Err(SimError {
                        error: e.to_string(),
                    })
                }
            };

        Ok(SimulatorState {
            time: 0,
            rng,
            enabled_defenses,
            attack_surface,
            remaining_ttc,
            num_observed_alerts: 0,
            compromised_steps,
            //actions: HashMap::new(),
            false_alerts: HashSet::new(),
            missed_alerts: HashSet::new(),
        })
    }

    pub fn total_ttc_remaining(&self) -> TTCType {
        return self.remaining_ttc.iter().map(|(_, &ttc)| ttc).sum();
    }

    pub fn get_ids_obs(&self) -> HashSet<&I> {
        self.compromised_steps // true alerts
            .union(&self.false_alerts) // add false alerts
            .filter_map(|x| match self.missed_alerts.contains(x) {
                true => None, // remove missed alerts
                false => Some(x),
            })
            .collect::<HashSet<&I>>()
    }

    pub fn attack_action(&self, step_id: &I) -> SimResult<ActionResult<I>> {
        let attack_surface_empty = self.attack_surface.is_empty();

        if attack_surface_empty {
            if cfg!(debug_assertions) {
                panic!("Attack surface is empty.");
            } else {
                log::warn!("Attack surface is empty.");
            }

            return Err(SimError {
                error: "Attack surface is empty.".to_string(),
            });
        }

        // If the selected attack step is not in the attack surface, do nothing
        if !self.attack_surface.contains(&step_id) {
            if cfg!(debug_assertions) {
                panic!("Attack step {} is not in the attack surface. Attack surface is: {:?}", step_id, self.attack_surface);
            }
            else {
                log::warn!("Attack step {} is not in the attack surface. Attack surface is: {:?}", step_id, self.attack_surface);
            }

            return Ok(ActionResult::default());
        }

        let result = ActionResult {
            enabled_defenses: HashSet::new(),
            ttc_diff: HashMap::from([(*step_id, -1)]),
            valid_action: true,
        };
        return Ok(result);
    }

    pub fn export_rng(&self) -> ChaChaRng {
        self.rng.clone()
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
            num_observed_alerts: self.num_observed_alerts,
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
