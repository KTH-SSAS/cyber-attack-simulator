use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display};
use std::hash::Hash;

type GraphResult<T> = std::result::Result<T, GraphError>;
use crate::graph::{Graph, Node};
use crate::loading::MALAttackStep;

#[derive(Debug, Clone)]
pub(crate) struct GraphError {
    message: String,
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
    Exists,
}

impl From<&str> for NodeType {
    fn from(s: &str) -> Self {
        match s {
            "and" => NodeType::And,
            "or" => NodeType::Or,
            "defense" => NodeType::Defense,
            "exist" => NodeType::Exists,
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
            NodeType::Exists => Logic::Or,
        }
    }
}

pub(crate) type TTCType = u64; // Time to compromise

pub(crate) struct AttackStep {
    id: String,
    asset: String,
    asset_id: usize,
    name: String,
    ttc: TTCType,
    logic: Logic,
    step_type: NodeType,
    //pub compromised: bool,
}

impl Display for AttackStep {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}/{}/{}", self.asset, self.asset_id, self.name)
    }
}

impl From<&MALAttackStep> for AttackStep {
    fn from(s: &MALAttackStep) -> AttackStep {
        let node_type = NodeType::from(s.node_type.as_str());
        let (asset, asset_id) = match s.asset.split_once(":") {
            Some((asset, asset_id)) => (asset.to_string(), asset_id.parse::<usize>().unwrap()),
            None => (s.asset.clone(), 0),
        };
        AttackStep {
            id: s.id.clone(),
            name: s.name.clone(),
            asset,
            asset_id,
            ttc: 0,
            logic: Logic::from(&node_type),
            step_type: node_type,
        }
    }
}

impl AttackStep {
    pub(crate) fn to_info_tuple(&self) -> (String, usize, String) {
        //split name by colon
        return (self.asset.clone(), self.asset_id, self.name.clone());
    }

    pub(crate) fn can_be_compromised(&self, parent_states: &Vec<bool>) -> bool {
        if parent_states.len() == 0 {
            return false;
        }
        match self.logic {
            Logic::And => parent_states.iter().all(|x| *x),
            Logic::Or => parent_states.iter().any(|x| *x),
        }
    }
}

impl<I> AttackGraph<I>
where
    I: Eq + Hash + Ord + Display + Copy,
{
    pub(crate) fn word2idx(&self, word: String) -> usize {
        let word = match self.vocab.get(&word) {
            Some(idx) => *idx,
            None => panic!("No index for word '{}'", word),
        };
        return word;
    }

    pub(crate) fn new(
        nodes: Vec<MALAttackStep>,
        edges: HashSet<(String, String)>,
        flags: Vec<String>,
        entry_points: Vec<String>,
        vocab: HashMap<String, usize>,
    ) -> AttackGraph<usize> {
        // Hash the node names to numerical indexes
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
                let child = match numerical_indexes.get(child) {
                    Some(child) => *child,
                    None => panic!("No such child node for {}: {}", parent, child),
                };
                (parent, child)
            })
            .collect();

        let graph = Graph { nodes, edges };

        let attack_steps = graph
            .nodes
            .values()
            .filter_map(|n| match &n.data.step_type {
                NodeType::Defense => None,
                _ => Some(numerical_indexes[&n.data.id]),
            })
            .collect();

        let defenses = graph
            .nodes
            .values()
            .filter_map(|n| match &n.data.step_type {
                NodeType::Defense => Some(numerical_indexes[&n.data.id]),
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
            vocab,
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
    pub(crate) attack_steps: HashSet<I>,
    pub(crate) defense_steps: HashSet<I>,
    pub(crate) flags: HashSet<I>,
    entry_points: HashSet<I>,
    pub(crate) vocab: HashMap<String, usize>,
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
    pub(crate) fn nodes(&self) -> &HashMap<I, Node<AttackStep, I>> {
        return &self.graph.nodes;
    }

    pub(crate) fn edges(&self) -> &HashSet<(I, I)> {
        return &self.graph.edges;
    }

    pub(crate) fn get_step(&self, id: &I) -> GraphResult<&AttackStep> {
        match self.graph.nodes.get(id) {
            Some(step) => Ok(&step.data),
            None => Err(GraphError {
                message: format!("No such step: {}", id),
            }),
        }
    }

    pub(crate) fn name_of_step(&self, id: &I) -> String {
        return format!("{}", self.graph.nodes[id].data);
    }

    /*
    pub(crate) fn has_defense(&self, id: &I) -> bool {
        self.defense_steps.contains(id)
    }
    */

    pub(crate) fn has_attack(&self, id: &I) -> bool {
        self.attack_steps.contains(id)
    }

    pub(crate) fn entry_points(&self) -> HashSet<I> {
        return self.entry_points.iter().map(|&i| i).collect();
    }

    pub(crate) fn disabled_defenses<'a>(&self, enabled_defenses: &'a HashSet<I>) -> HashSet<I> {
        self.defense_steps
            .difference(enabled_defenses)
            .map(|x| *x)
            .collect()
    }

    pub(crate) fn flag_to_index(&self, id_to_index: &HashMap<I, usize>) -> Vec<usize> {
        self.flags
            .iter()
            .map(|id| id_to_index[id])
            .collect::<Vec<usize>>()
    }

    pub(crate) fn get_flag_status(&self, compromised_steps: &HashSet<I>) -> HashMap<I, bool> {
        return self
            .flags
            .iter()
            .map(|flag_id| (*flag_id, compromised_steps.contains(flag_id)))
            .collect();
    }

    pub(crate) fn number_of_defenses(&self) -> usize {
        self.defense_steps.len()
    }

    pub(crate) fn number_of_attacks(&self) -> usize {
        self.attack_steps.len()
    }

    pub(crate) fn children(&self, id: &I) -> HashSet<&Node<AttackStep, I>> {
        return self.graph.children(id);
    }

    pub(crate) fn to_graphviz(
        &self,
        attributes: Option<&HashMap<&I, Vec<(String, String)>>>,
    ) -> String {
        return self.graph.to_graphviz(attributes);
    }

    pub(crate) fn get_attack_parents(&self, id: &I) -> HashSet<&I> {
        self.graph
            .parents(id)
            .iter()
            .filter_map(|&p| match (&p.data.step_type, NodeType::Defense) {
                (NodeType::Defense, NodeType::Defense) => None,
                _ => Some(&p.id),
            }) // Exclude defense parents
            .collect()
    }

    pub(crate) fn get_defense_parents(&self, id: &I) -> HashSet<&I> {
        self.graph
            .parents(id)
            .iter()
            .filter_map(|&p| match (&p.data.step_type, NodeType::Defense) {
                (NodeType::Defense, NodeType::Defense) => Some(&p.id),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn step_is_defended_by(&self, step_id: &I, defense_id: &I) -> bool {
        let parents = self.graph.parents(step_id);
        return parents
            .iter()
            .filter_map(|f| match f.data.step_type {
                NodeType::Defense => Some(f.id),
                _ => None,
            })
            .any(|p| p == *defense_id);
    }

    pub(crate) fn ttc_params(&self) -> Vec<(I, TTCType)> {
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

    pub(crate) fn is_entry(&self, id: &I) -> bool {
        return self.entry_points.contains(id);
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

    /*
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
    */
}
