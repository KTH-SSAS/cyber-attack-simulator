use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_yaml::{self, Mapping};
use std::fmt::{self, Display};
use std::hash::Hash;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufReader,
};

type GraphResult<T> = std::result::Result<T, GraphError>;
use crate::graph::{Graph, Node};
use crate::loading::MALAttackStep;

#[derive(Debug, Clone)]
pub struct GraphError {
    pub message: String,
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GraphError: {}", self.message)
    }
}

impl std::error::Error for GraphError {
    fn description(&self) -> &str {
        &self.message
    }
}

// impl GraphError {
//     fn new(message: String) -> GraphError {
//         GraphError { message }
//     }

//     fn NoSuchAttackStep(id: u64) -> GraphError {
//         GraphError::new(format!("No such attack step: {}", id))
//     }
// }

#[derive(Clone, PartialEq)]
pub(crate) enum NodeType {
    And,
    Or,
    Defense,
}

impl From<&str> for NodeType {
    fn from(s: &str) -> Self {
        match s {
            "and" => NodeType::And,
            "or" => NodeType::Or,
            "defense" => NodeType::Defense,
            _ => panic!("Unknown node type: {}", s),
        }
    }
}

#[derive(Clone)]
pub(crate) enum Logic {
    And,
    Or,
}

impl From<&NodeType> for Logic {
    fn from(node_type: &NodeType) -> Self {
        match node_type {
            NodeType::And => Logic::And,
            NodeType::Or => Logic::Or,
            NodeType::Defense => Logic::Or,
        }
    }
}

pub type TTCType = u64; // Time to compromise

pub(crate) struct AttackStep {
    pub name: String,
    pub ttc: TTCType,
    pub logic: Logic,
    pub step_type: NodeType,
    //pub compromised: bool,
}

impl Display for AttackStep {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl From<&MALAttackStep> for AttackStep {
    fn from(s: &MALAttackStep) -> AttackStep {
        let node_type = NodeType::from(s.node_type.as_str());
        AttackStep {
            name: s.id.clone(),
            ttc: 1,
            logic: Logic::from(&node_type),
            step_type: node_type,
        }
    }
}

impl<I> AttackGraph<I>
where
    I: Eq + Hash + Ord + Display + Copy,
{
    pub fn new(
        nodes: Vec<MALAttackStep>,
        edges: HashSet<(String, String)>,
        flags: Vec<String>,
        entry_points: Vec<String>,
    ) -> AttackGraph<usize> {
        let numerical_indexes = nodes
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id.clone(), i))
            .collect::<HashMap<String, usize>>();

        let nodes = nodes
            .iter()
            .map(|s| {
                let id = numerical_indexes[&s.id];
                (
                    id,
                    Node {
                        id,
                        data: AttackStep::from(s),
                    },
                )
            })
            .collect();

        let edges = edges
            .iter()
            .map(|(parent, child)| {
                let parent = numerical_indexes[parent];
                let child = numerical_indexes[child];
                (parent, child)
            })
            .collect();

        let graph = Graph { nodes, edges };

        let attack_steps = graph
            .nodes
            .values()
            .filter_map(|n| match &n.data.step_type {
                NodeType::Defense => None,
                _ => Some(numerical_indexes[&n.data.name]),
            })
            .collect();

        let defenses = graph
            .nodes
            .values()
            .filter_map(|n| match &n.data.step_type {
                NodeType::Defense => Some(numerical_indexes[&n.data.name]),
                _ => None,
            })
            .collect();

        let flags = flags.iter().map(|x| numerical_indexes[x]).collect();

        // let defense_indices = steps
        //     .iter()
        //     .filter(|n| defense_steps.contains(&n.id))
        //     .enumerate()
        //     .map(|(i, s)| (s.id, i))
        //     .collect::<HashMap<I, usize>>();

        // let names = steps
        //     .iter()
        //     .map(|s| s.name.clone())
        //     .collect::<Vec<String>>();

        let graph = AttackGraph {
            graph,
            attack_steps,
            flags,
            defense_steps: defenses,
            entry_points: entry_points.iter().map(|x| numerical_indexes[x]).collect(),
        };

        return graph;
    }
}

impl PartialEq for AttackStep {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

pub(crate) struct AttackGraph<I> {
    graph: Graph<AttackStep, I>,
    attack_steps: HashSet<I>,
    pub defense_steps: HashSet<I>,
    pub flags: HashSet<I>,
    entry_points: HashSet<I>,
}

/*
fn get_parents<I>(id: &I, edges: &Vec<(I, I)>) -> HashSet<I> where I: Eq + Hash {
    let parents = edges
        .iter()
        .filter(|(_, child)| child == id)
        .map(|(parent, _)| *parent)
        .collect::<HashSet<I>>();
    return parents;
}

fn get_children<I>(id: &I, edges: &Vec<(I, I)>) -> HashSet<I> where I: Eq + Hash {
    let children = edges
        .iter()
        .filter(|(parent, _)| parent == id)
        .map(|(_, child)| *child)
        .collect::<HashSet<I>>();
    return children;
}
*/

impl<I> AttackGraph<I>
where
    I: Eq + Hash + Ord + Display + Copy,
{
    pub fn nodes(&self) -> &HashMap<I, Node<AttackStep, I>> {
        return &self.graph.nodes;
    }

    pub fn edges(&self) -> &HashSet<(I, I)> {
        return &self.graph.edges;
    }

    pub fn has_defense(&self, id: &I) -> bool {
        self.defense_steps.contains(id)
    }

    pub fn has_attack(&self, id: &I) -> bool {
        self.attack_steps.contains(id)
    }

    pub fn entry_points(&self) -> HashSet<I> {
        return self.entry_points.iter().map(|&i| i).collect();
    }



    pub fn uncompromised_steps<'a>(&self, compromised_steps: &'a HashSet<I>) -> HashSet<I> {
        self.attack_steps.difference(&compromised_steps).map(|x| *x).collect()
    }

    pub fn disabled_defenses<'a>(&self, enabled_defenses: &'a HashSet<I>) -> HashSet<I> {
        self.defense_steps.difference(enabled_defenses).map(|x| *x).collect()
    }

    pub fn flag_to_index(&self, id_to_index: &HashMap<I, usize>) -> Vec<usize> {
        self.flags
            .iter()
            .map(|id| id_to_index[id])
            .collect::<Vec<usize>>()
    }

    pub fn get_flag_status(&self, compromised_steps: &HashSet<I>) -> HashMap<I, bool> {
        return self
            .flags
            .iter()
            .map(|flag_id| (*flag_id, compromised_steps.contains(flag_id)))
            .collect();
    }

    pub fn defender_impact(&self, id_to_index: &HashMap<I, usize>) -> Vec<i64> {
        let mut impact = vec![0; id_to_index.len()];

        self.flags
            .iter()
            .map(|id| id_to_index[id])
            .for_each(|index| {
                impact[index] = -2 //-(self.ttc_sum as i64);
            });

        self.defense_steps
            .iter()
            .map(|id| id_to_index[id])
            .for_each(|index| {
                impact[index] = -1;
            });

        return impact;
    }

    pub fn attacker_impact(&self, id_to_index: &HashMap<I, usize>) -> Vec<i64> {
        let mut impact = vec![0; id_to_index.len()];

        self.flags
            .iter()
            .map(|id| id_to_index[id])
            .for_each(|index| {
                impact[index] = 1;
            });

        return impact;
    }

    pub fn number_of_defenses(&self) -> usize {
        self.defense_steps.len()
    }

    pub fn number_of_attacks(&self) -> usize {
        self.attack_steps.len()
    }

    pub fn children(&self, id: &I) -> Vec<&Node<AttackStep, I>> {
        return self.graph.children(id);
    }

    pub fn calculate_compromised_steps(&self, remaining_ttc: &HashMap<I, u64>) -> HashSet<I> {
        let steps_with_zero_ttc: HashSet<I> = remaining_ttc
            .iter()
            .filter_map(|(step, ttc)| match ttc {
                0 => Some(*step),
                _ => None,
            })
            .collect();

        steps_with_zero_ttc
            .iter()
            .filter_map(|step| match self.graph.nodes.get(step) {
                Some(step) => Some(step),
                None => None,
            })
            .filter(|step| self.is_traversible(&step.id, &steps_with_zero_ttc))
            .map(|step| step.id)
            .collect()
    }

    pub fn to_graphviz(&self, attributes: Option<&HashMap<I, Vec<(String, String)>>>) -> String {
        return self.graph.to_graphviz(attributes);
    }

    pub fn calculate_attack_surface(
        &self,
        compromised_steps: &HashSet<I>,
        defense_state: &HashSet<I>,
    ) -> GraphResult<HashSet<I>> {
        let attack_surface: HashSet<I> =
            compromised_steps
                .iter()
                .fold(HashSet::new(), |mut acc, step| {
                    let vulnerable_children = match self.get_vulnerable_children(
                        step,
                        &compromised_steps,
                        &defense_state,
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            panic!("Error in get_vulnerable_children: {} for step {}", e, step)
                        }
                    };
                    acc.extend(vulnerable_children);
                    acc
                });

        return Ok(attack_surface);
    }

    pub fn ttc_params(&self) -> Vec<(I, TTCType)> {
        let ttc_params: Vec<(I, TTCType)> = self
            .graph
            .nodes
            .values()
            .map(|x| (x.id, x.data.ttc))
            .map(|(id, ttc)| match self.entry_points.contains(&id) {
                true => (id, 0),
                false => (id, ttc),
            })
            .sorted()
            .collect();
        return ttc_params;
    }

    pub fn is_entry(&self, id: &I) -> bool {
        return self.entry_points.contains(id);
    }

    pub fn is_traversible(&self, node_id: &I, compromised_steps: &HashSet<I>) -> bool {
        let node = self.graph.nodes.get(node_id).unwrap();
        let parents = self.graph.parents(node_id);

        let attack_parents: HashSet<&I> = parents
            .iter()
            .filter_map(|&p| match (&p.data.step_type, NodeType::Defense) {
                (NodeType::Defense, NodeType::Defense) => None,
                _ => Some(&p.id),
            }) // Exclude defense parents
            .collect();

        if attack_parents.is_empty() {
            return self.is_entry(node_id);
        }

        let parent_states: Vec<bool> = attack_parents
            .iter()
            .map(|&p| compromised_steps.contains(p))
            .collect();

        return match node.data.logic {
            Logic::And => parent_states.iter().all(|x| *x),
            Logic::Or => parent_states.iter().any(|x| *x),
        };
    }

    pub fn is_vulnerable(
        &self,
        node_id: &I,
        compromised_steps: &HashSet<I>,
        enabled_defenses: &HashSet<I>,
    ) -> GraphResult<bool> {
        // Returns true if a node can be attacked given the current state of the
        // graph meaning that the
        let traversable = self.is_traversible(node_id, compromised_steps);
        let compromised = compromised_steps.contains(node_id);
        let parents = self.graph.parents(node_id);
        let defended = parents.iter().any(|d| enabled_defenses.contains(&d.id));
        return Ok(!compromised && traversable && !defended);
    }

    pub fn get_vulnerable_children(
        &self,
        step_id: &I,
        compromised_steps: &HashSet<I>,
        enabled_defenses: &HashSet<I>,
    ) -> GraphResult<HashSet<I>> {
        // let step = match self.get_step(attack_step) {
        //     Ok(c) => c,
        //     Err(e) => return Err(e),
        // };

        let step = self.graph.nodes.get(step_id).unwrap();
        let children: Vec<&Node<AttackStep, I>> = self.graph.children(step_id);

        let vulnerables: Vec<bool> = match children
            .iter()
            .map(|c| self.is_vulnerable(&c.id, compromised_steps, enabled_defenses))
            .collect()
        {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let vulnerable_children: HashSet<I> = children
            .iter()
            .zip(vulnerables.iter())
            .filter(|(_, c)| **c)
            .map(|(c, _)| c.id)
            .collect();
        return Ok(vulnerable_children);
    }
}

// fn traverse<'a>(entrypoint: &'a AttackStep) -> Vec<&'a AttackStep> {
//     // Perform a breadth-first search to find all attack steps
//     let mut queue = Vec::new();
//     let mut visited: Vec<&AttackStep> = Vec::new();
//     queue.push(entrypoint);

//     while !queue.is_empty() {
//         let current = queue.pop().unwrap();
//         for child in current.children.borrow().iter() {
//             if !visited.contains(&child) {
//                 queue.push(child);
//             }
//         }
//         visited.push(current);
//     }
//     return visited;
// }

#[derive(Serialize, Deserialize)]
pub struct TTC {
    #[serde(rename = "type")]
    ttc_type: String,
    name: String,
    arguments: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use crate::loading;

    #[test]
    fn load_graph_from_file() {
        let filename = "graphs/four_ways.yaml";
        let attackgraph = loading::load_graph_from_yaml(filename);
        let entry_point = attackgraph.entry_points.iter().collect::<Vec<&u64>>();

        let entry_point = *entry_point.first().unwrap();
        assert_eq!(
            attackgraph.graph.nodes[entry_point].data.name,
            "attacker-13-enter-13"
        );

        assert_eq!(attackgraph.graph.children(entry_point).len(), 4);
        assert_eq!(attackgraph.graph.parents(entry_point).len(), 0);
        assert_eq!(attackgraph.entry_points.len(), 1);
        assert_eq!(attackgraph.attack_steps.len(), 15);
        assert_eq!(attackgraph.defense_steps.len(), 4);
        assert_eq!(attackgraph.graph.nodes.len(), 19);
        assert_eq!(attackgraph.flags.len(), 4);

        let graphviz = attackgraph.graph.to_graphviz(None);
        let mut file = std::fs::File::create("test.dot").unwrap();
        file.write_all(graphviz.as_bytes()).unwrap();
        file.flush().unwrap();
    }
}
