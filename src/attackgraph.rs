use serde_yaml::{self, Mapping};
use std::fmt;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufReader,
};

type GraphResult<T> = std::result::Result<T, GraphError>;
use crate::graph::{Graph, Node, NodeID};

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

impl GraphError {
    fn new(message: String) -> GraphError {
        GraphError { message }
    }

    fn NoSuchAttackStep(id: u64) -> GraphError {
        GraphError::new(format!("No such attack step: {}", id))
    }
}

pub(crate) enum Logic {
    And,
    Or,
}

impl Clone for Logic {
    fn clone(&self) -> Self {
        match self {
            Logic::And => Logic::And,
            Logic::Or => Logic::Or,
        }
    }
}

pub(crate) struct SerializedAttackStep {
    pub id: u64,
    pub name: String,
    pub ttc: u64,
    pub logic: Logic,
}

impl SerializedAttackStep {
    fn new<'a>(id: u64, name: String, ttc: u64, logic: Logic) -> SerializedAttackStep {
        SerializedAttackStep {
            id,
            name,
            ttc,
            logic,
        }
    }
}

pub type TTCType = u64; // Time to compromise

pub(crate) struct AttackStep {
    pub name: String,
    pub ttc: TTCType,
    pub logic: Logic,
    pub defenses: HashSet<NodeID>,
    pub is_entry: bool,
    //pub compromised: bool,
}

impl Node<AttackStep> {
    pub fn is_traversible(&self, compromised_steps: &HashSet<NodeID>) -> bool {
        let defenses = &self.data.defenses;

        let attack_parents: HashSet<&u64> = self
            .parents
            .iter()
            .filter(|&p| !defenses.contains(p))
            .collect();

        if attack_parents.is_empty() {
            return self.data.is_entry;
        }

        let parent_states: Vec<bool> = attack_parents
            .iter()
            .map(|&p| compromised_steps.contains(p))
            .collect();

        return match self.data.logic {
            Logic::And => parent_states.iter().all(|x| *x),
            Logic::Or => parent_states.iter().any(|x| *x),
        };
    }

    pub fn is_vulnerable(
        &self,
        compromised_steps: &HashSet<NodeID>,
        enabled_defenses: &HashSet<NodeID>,
    ) -> GraphResult<bool> {
        let traversable = self.is_traversible(compromised_steps);
        let compromised = compromised_steps.contains(&self.id);
        let defended = self
            .data
            .defenses
            .iter()
            .any(|d| enabled_defenses.contains(d));
        return Ok(!compromised && traversable && !defended);
    }
}

impl PartialEq for AttackStep {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

pub(crate) struct AttackGraph {
    pub graph: Graph<AttackStep>,
    pub attack_steps: HashSet<u64>,
    pub defense_steps: HashSet<u64>,
    pub flags: HashSet<u64>,
    pub entry_points: HashSet<u64>,
}

fn get_parents(id: &NodeID, edges: &Vec<(NodeID, NodeID)>) -> HashSet<NodeID> {
    let parents = edges
        .iter()
        .filter(|(_, child)| child == id)
        .map(|(parent, _)| *parent)
        .collect::<HashSet<u64>>();
    return parents;
}

fn get_children(id: &NodeID, edges: &Vec<(NodeID, NodeID)>) -> HashSet<NodeID> {
    let children = edges
        .iter()
        .filter(|(parent, _)| parent == id)
        .map(|(_, child)| *child)
        .collect::<HashSet<u64>>();
    return children;
}

impl AttackGraph {
    pub fn new<'a>(
        nodes: Vec<SerializedAttackStep>,
        edges: Vec<(NodeID, NodeID)>,
        defense_steps: Vec<NodeID>,
        flags: Vec<NodeID>,
        entry_points: Vec<NodeID>,
    ) -> AttackGraph {
        let nodes: HashMap<NodeID, Node<AttackStep>> = nodes
            .iter()
            .map(|s| {
                let children = get_children(&s.id, &edges);
                let parents = get_parents(&s.id, &edges);
                let defenses = defense_steps
                    .iter()
                    .filter(|d| parents.contains(d))
                    .map(|d| *d)
                    .collect();

                let attack_step = AttackStep {
                    name: s.name.clone(),
                    ttc: if entry_points.contains(&s.id) {
                        0
                    } else {
                        s.ttc
                    },
                    logic: s.logic.clone(),
                    defenses,
                    is_entry: entry_points.contains(&s.id),
                };

                (
                    s.id,
                    Node {
                        id: s.id,
                        data: attack_step,
                        parents,
                        children,
                    },
                )
            })
            .collect();

        let graph = Graph { nodes };

        let attack_steps = graph
            .nodes
            .keys()
            .filter_map(|n| match defense_steps.contains(&n) {
                true => None,
                false => Some(*n),
            })
            .collect();

        let defenses = defense_steps.iter().map(|x| *x).collect();
        let flags = flags.iter().map(|x| *x).collect();

        // let defense_indices = steps
        //     .iter()
        //     .filter(|n| defense_steps.contains(&n.id))
        //     .enumerate()
        //     .map(|(i, s)| (s.id, i))
        //     .collect::<HashMap<NodeID, usize>>();

        // let names = steps
        //     .iter()
        //     .map(|s| s.name.clone())
        //     .collect::<Vec<String>>();

        let graph = AttackGraph {
            graph,
            attack_steps,
            flags,
            defense_steps: defenses,
            entry_points: entry_points.iter().map(|x| *x).collect(),
        };

        return graph;
    }

    pub fn ttc_params(&self) -> Vec<(NodeID, TTCType)> {
        let mut ttc_params: Vec<(NodeID, TTCType)> = self
            .attack_steps
            .iter()
            .map(|i| self.graph.nodes.get(i).unwrap())
            .map(|x| (x.id, x.data.ttc))
            .map(|(id, ttc)| match self.entry_points.contains(&id) {
                true => (id, 0),
                false => (id, ttc),
            })
            .collect();
        ttc_params.sort();
        return ttc_params;
    }

    pub fn get_vulnerable_children(
        &self,
        step_id: &NodeID,
        compromised_steps: &HashSet<NodeID>,
        enabled_defenses: &HashSet<NodeID>,
    ) -> GraphResult<HashSet<NodeID>> {
        // let step = match self.get_step(attack_step) {
        //     Ok(c) => c,
        //     Err(e) => return Err(e),
        // };

        let step = self.graph.nodes.get(step_id).unwrap();
        let children: Vec<&Node<AttackStep>> = match step
            .children
            .iter()
            .map(|id| self.graph.nodes.get(id))
            .collect()
        {
            Some(c) => c,
            None => {
                return Err(GraphError {
                    message: "Could not find children".to_string(),
                })
            }
        };

        let vulnerables: Vec<bool> = match children
            .iter()
            .map(|&c| c.is_vulnerable(compromised_steps, enabled_defenses))
            .collect()
        {
            Ok(c) => c,
            Err(e) => return Err(e),
        };

        let vulnerable_children: HashSet<u64> = children
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

pub(crate) fn load_graph_from_yaml<'a: 'b, 'b>(filename: &str) -> AttackGraph {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let mapping: Mapping = serde_yaml::from_reader(reader).unwrap();

    // mapping.into_keys().for_each(|key| {
    //  	println!("{:?}", key);
    //  });

    let mut defenses = mapping
        .get(String::from("defenses"))
        .unwrap()
        .as_sequence()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect::<Vec<String>>();

    let mut flags: Vec<String> = mapping
        .get(String::from("flags"))
        .unwrap()
        .as_mapping()
        .unwrap()
        .iter()
        .map(|(k, _)| {
 
                k.as_str().unwrap().to_string()
            
        })
        .collect();

    defenses.sort();

    let entry_points = mapping
        .get(String::from("entry_points"))
        .unwrap()
        .as_sequence()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect::<Vec<String>>();

    let attack_graph = mapping
        .get(String::from("attack_graph"))
        .unwrap()
        .as_sequence()
        .unwrap();

    let mut id = 0;

    let mut step_names = attack_graph
        .iter()
        .map(|s| s.get("id").unwrap().as_str().unwrap().to_string())
        .collect::<Vec<String>>();

    step_names.sort();

    let attack_steps = step_names
        .iter()
        .map(|name| {
            let attack_step = attack_graph
                .iter()
                .find(|s| s.get("id").unwrap().as_str().unwrap() == name)
                .unwrap();
            //let id = attack_step.get(String::from("id")).unwrap().as_str().unwrap().to_string();
            let name = attack_step.get("id").unwrap().as_str().unwrap().to_string();
            let ttc = attack_step.get("ttc").unwrap().as_u64().unwrap();
            let logic = match attack_step.get("step_type").unwrap().as_str().unwrap() {
                "and" => Logic::And,
                "or" => Logic::Or,
                _ => panic!("Invalid logic"),
            };
            // let children = attack_step.get(String::from("children")).unwrap().as_sequence().unwrap();
            // let children = children.iter().map(|child| {
            //     child.as_str().unwrap().to_string()
            // }).collect::<Vec<String>>();
            let i = id;
            id = id + 1;
            return SerializedAttackStep::new(i, name.clone(), ttc, logic);
        })
        .collect::<Vec<SerializedAttackStep>>();

    let name2index = attack_steps
        .iter()
        .map(|s| (s.name.clone(), s.id))
        .collect::<HashMap<String, u64>>();

    let mut edges: HashSet<(u64, u64)> = HashSet::new();

    let mut edge: (u64, u64);
    let mut step_id: u64;
    let mut child_id: u64;
    for attack_step in attack_graph.iter() {
        let name = attack_step.get("id").unwrap().as_str().unwrap().to_string();
        let children = match attack_step.get("children") {
            Some(c) => c.as_sequence().unwrap(),
            None => panic!("No 'children' field for {}", name),
        };

        step_id = name2index[&name];
        for child in children.iter() {
            child_id = *name2index
                .get(&child.as_str().unwrap().to_string())
                .unwrap();
            edge = (step_id, child_id);
            edges.insert(edge);
        }
    }

    // for step in attack_steps.iter() {
    //     let parents = attack_steps.iter().filter(|attack_step| {
    //         attack_step.children.contains(&step)
    //     }).collect::<Vec<&AttackStep>>();
    //     step.parents = parents;
    // }

    // attack_graph.clone().into_keys().for_each(|key| {
    //      println!("{:?}", key);
    // });

    let defense_ids = attack_steps
        .iter()
        .filter(|step| defenses.contains(&step.name))
        .map(|step| step.id)
        .collect::<Vec<u64>>();

    let flag_ids = attack_steps
        .iter()
        .filter(|step| flags.contains(&step.name))
        .map(|step| step.id)
        .collect::<Vec<u64>>();

    let entry_ids = attack_steps
        .iter()
        .filter(|step| entry_points.contains(&step.name))
        .map(|step| step.id)
        .collect::<Vec<u64>>();

    return AttackGraph::new(
        attack_steps,
        edges.into_iter().collect(),
        defense_ids,
        flag_ids,
        entry_ids,
    );
}

#[cfg(test)]
mod tests {
    use crate::attackgraph;
    #[test]
    fn load_graph_from_file() {
        let filename = "four_ways.yaml";
        let attackgraph = attackgraph::load_graph_from_yaml(filename);
        let entry_point = attackgraph.entry_points.iter().collect::<Vec<&u64>>();

        let entry_point = *entry_point.first().unwrap();
        assert_eq!(
            attackgraph.graph.nodes[entry_point].data.name,
            "attacker-13-enter-13"
        );
        assert_eq!(attackgraph.graph.nodes[entry_point].children.len(), 4);
        assert_eq!(attackgraph.graph.nodes[entry_point].parents.len(), 0);
        assert_eq!(attackgraph.entry_points.len(), 1);
        assert_eq!(attackgraph.attack_steps.len(), 15);
        assert_eq!(attackgraph.defense_steps.len(), 4);
        assert_eq!(attackgraph.graph.nodes.len(), 19);
        assert_eq!(attackgraph.flags.len(), 4);
    }
}
