use core::fmt::Debug;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{self, Display};
use std::hash::Hash;
type GraphResult<T> = std::result::Result<T, GraphError>;
use crate::graph::{Graph, Node};
use crate::loading::MALAttackStep;

#[derive(Debug, Clone)]
pub(crate) struct Confusion {
    pub fnr: f64,
    pub fpr: f64,
}

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
    pub name: String,
    ttc: TTCType,
    logic: Logic,
    step_type: NodeType,
    confusion: Confusion,
    //pub compromised: bool,
}

impl Display for AttackStep {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}/{}/{}", self.asset, self.asset_id, self.name)
    }
}

fn split_asset_name(asset_name: &str) -> (String, usize) {
    let (asset, asset_id) = match asset_name.split_once(":") {
        Some((asset, asset_id)) => (asset.to_string(), asset_id.parse::<usize>().unwrap()),
        None => (asset_name.to_string(), 0),
    };
    return (asset, asset_id);
}

fn split_id(mal_id: &str) -> (String, usize, String) {
    let (asset, asset_id, name) = match mal_id.split_once(":") {
        Some((asset, rest)) => match rest.split_once(":") {
            Some((asset_id, name)) => (
                asset.to_string(),
                asset_id.parse::<usize>().unwrap(),
                name.to_string(),
            ),
            None => panic!("Invalid part: {} in {}", rest, mal_id),
        },
        None => panic!("Invalid id: {}", mal_id),
    };
    return (asset, asset_id, name);
}

impl AttackStep {
    pub(crate) fn to_info_tuple(&self) -> (String, usize, String) {
        //split name by colon
        return (self.asset.clone(), self.asset_id, self.name.clone());
    }

    pub(crate) fn asset(&self) -> String {
        return format!("{}:{}", self.asset, self.asset_id);
    }

    pub(crate) fn can_be_compromised_vec(&self, parent_states: &Vec<bool>) -> bool {
        match (&self.logic, parent_states.iter().any(|&x| x)) {
            (Logic::And, p) => p && parent_states.iter().all(|&x| x),
            (Logic::Or, p) => p,
        }
    }

    pub(crate) fn can_be_compromised(&self, parent_states: impl Iterator<Item = bool>) -> bool {
        //let mut parent_states = parent_states;
        parent_states.fold(false, |acc, x| match (&self.logic, x) {
            (Logic::And, p) => acc && p,
            (Logic::Or, p) => acc || p,
        })
    }

    fn from(s: &MALAttackStep, fpr: f64, fnr: f64) -> AttackStep {
        let node_type = NodeType::from(s.node_type.as_str());
        let (asset, asset_id) = split_asset_name(&s.asset);
        AttackStep {
            id: s.id.clone(),
            name: s.name.clone(),
            asset,
            asset_id,
            ttc: 0,
            logic: Logic::from(&node_type),
            confusion: match node_type {
                NodeType::Defense => Confusion { fnr: 0.0, fpr: 0.0 },
                _ => Confusion { fnr, fpr },
            },
            step_type: node_type,
        }
    }
}

impl<I> AttackGraph<I>
where
    I: Eq + Hash + Ord + Debug + Copy,
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
        fpr: f64,
        fnr: f64,
    ) -> AttackGraph<(usize, usize, usize)> {
        // Hash the node names to numerical indexes

        let translate_id = |x: &String| {
            let (asset, asset_id, name) = split_id(&x);
            let id = (vocab[&asset], asset_id, vocab[&name]);
            return id;
        };

        let nodes = nodes
            .iter()
            .map(|s| {
                let id = translate_id(&s.id);
                (
                    id,
                    Node {
                        id,
                        data: AttackStep::from(s, fpr, fnr),
                    },
                )
            })
            .collect();

        let edges = edges
            .iter()
            .map(|(parent, child)| {
                let parent = translate_id(parent);
                let child = translate_id(child);
                (parent, child)
            })
            .collect();

        let graph = Graph { nodes, edges };

        let attack_steps = graph
            .nodes
            .iter()
            .filter_map(|(i, n)| match &n.data.step_type {
                NodeType::Defense => None,
                _ => Some(*i),
            })
            .collect();

        let defenses = graph
            .nodes
            .iter()
            .filter_map(|(i, n)| match &n.data.step_type {
                NodeType::Defense => Some(*i),
                _ => None,
            })
            .collect();

        let flags = flags.iter().map(|x| translate_id(x)).collect();

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

        let entry_points = entry_points.iter().map(|x| translate_id(x)).collect();
        let graph = AttackGraph {
            vocab,
            graph,
            attack_steps,
            flags,
            defense_steps: defenses,
            entry_points,
        };

        return graph;
    }

    pub(crate) fn confusion_for_step(&self, i: &I) -> Confusion {
        self.get_step(i).unwrap().confusion.clone()
    }
}

impl PartialEq for AttackStep {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

pub(crate) struct AttackGraph<I>
where
    I: Ord,
{
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
    I: Eq + Hash + Ord + Debug + Copy,
{
    pub(crate) fn nodes(&self) -> &BTreeMap<I, Node<AttackStep, I>> {
        return &self.graph.nodes;
    }

    pub(crate) fn edges(&self) -> &Vec<(I, I)> {
        return &self.graph.edges;
    }

    pub(crate) fn get_step(&self, id: &I) -> GraphResult<&AttackStep> {
        match self.graph.nodes.get(id) {
            Some(step) => Ok(&step.data),
            None => Err(GraphError {
                message: format!("No such step: {:?}", id),
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

    pub(crate) fn distinct_objects(&self) -> HashSet<String> {
        self.graph
            .nodes
            .values()
            .map(|n| format!("{}:{}", n.data.asset.clone(), n.data.asset_id.clone()))
            .collect()
    }

    pub(crate) fn distinct_actions(&self) -> HashSet<String> {
        self.graph
            .nodes
            .values()
            .map(|n| n.data.name.clone())
            .collect()
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

    pub(crate) fn children(&self, id: &I) -> Vec<&Node<AttackStep, I>> {
        return self.graph.children(id);
    }

    pub(crate) fn to_graphviz(
        &self,
        attributes: Option<&HashMap<&I, Vec<(String, String)>>>,
    ) -> String {
        return self.graph.to_graphviz(attributes);
    }

    pub(crate) fn get_attack_parents<'a>(&'a self, id: &'a I) -> impl Iterator<Item = &'a I> {
        self.graph
            .parents(id)
            .map(|p| (&p.id, &p.data.step_type))
            .filter_map(|(id, step_type)| match (step_type, NodeType::Defense) {
                (NodeType::Defense, NodeType::Defense) => None,
                _ => Some(id),
            }) // Exclude defense parents
    }

    pub(crate) fn get_defense_parents<'a>(&'a self, id: &'a I) -> impl Iterator<Item = &'a I> {
        self.graph
            .parents(id)
            .map(|p| (&p.id, &p.data.step_type))
            .filter_map(|(id, step_type)| match (step_type, NodeType::Defense) {
                (NodeType::Defense, NodeType::Defense) => Some(id),
                _ => None,
            })
    }

    pub(crate) fn step_is_defended_by(&self, step_id: &I, defense_id: &I) -> bool {
        let parents = self.graph.parents(step_id);
        return parents
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
    use std::io::Write;

    use crate::loading;

    #[test]
    fn load_graph_from_file() {
        let filename = "graphs/corelang.json";
        let attackgraph = loading::load_graph_from_json(filename, None, 0.0, 0.0).unwrap();

        println!("{:?}", attackgraph.distinct_objects());
        println!("{:?}", attackgraph.distinct_actions());

        let graphviz = attackgraph.graph.to_graphviz(None);
        let mut file = std::fs::File::create("test.dot").unwrap();
        file.write_all(graphviz.as_bytes()).unwrap();
        file.flush().unwrap();
    }
}
