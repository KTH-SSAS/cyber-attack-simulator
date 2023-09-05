use std::{io::BufReader, fs::File};

use itertools::Itertools;
use serde::{Serialize, Deserialize};
use serde_yaml::Mapping;

use std::{
    collections::{HashMap, HashSet},
};


use crate::attackgraph::AttackGraph;

#[derive(Serialize, Deserialize)]
pub struct TTC {
    #[serde(rename = "type")]
    ttc_type: String,
    name: String,
    arguments: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct MALAttackStep {
    id: String,
    #[serde(rename = "type")]
    node_type: String,
    objclass: String,
    objid: String,
    atkname: String,
    ttc: Option<TTC>,
    links: Vec<String>,
    is_reachable: bool,
    defense_status: Option<String>,
    graph_type: String,
    is_traversable: bool,
    required_steps: Option<Vec<String>>,
    extra: Option<serde_json::Value>,
}

pub(crate) fn load_graph_from_json(filename: &str) -> Vec<MALAttackStep> {
    let file = match File::open(filename) {
        Ok(f) => f,
        Err(e) => panic!("Could not open file: {}. {}", filename, e),
    };

    let reader = BufReader::new(file);
    let steps: Vec<MALAttackStep> = serde_json::from_reader(reader).unwrap();
    return steps;
}

pub(crate) fn load_graph_from_yaml(filename: &str) -> AttackGraph {
    let file = match File::open(filename) {
        Ok(f) => f,
        Err(e) => panic!("Could not open file: {}. {}", filename, e),
    };
    let reader = BufReader::new(file);
    let mapping: Mapping = serde_yaml::from_reader(reader).unwrap();

    // mapping.into_keys().for_each(|key| {
    //  	println!("{:?}", key);
    //  });

    let defenses = mapping
        .get(String::from("defenses"))
        .unwrap()
        .as_sequence()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .sorted()
        .collect::<Vec<String>>();

    let flags: Vec<String> = mapping
        .get(String::from("flags"))
        .unwrap()
        .as_mapping()
        .unwrap()
        .iter()
        .map(|(k, _)| k.as_str().unwrap().to_string())
        .collect();

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

    let step_names = attack_graph
        .iter()
        .map(|s| s.get("id").unwrap().as_str().unwrap().to_string())
        .sorted()
        .collect::<Vec<String>>();

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
            let step_type = match attack_step.get("step_type").unwrap().as_str().unwrap() {
                "and" => NodeType::And,
                "or" => NodeType::Or,
                "defense" => NodeType::Defense,
                _ => panic!("Invalid logic"),
            };
            // let children = attack_step.get(String::from("children")).unwrap().as_sequence().unwrap();
            // let children = children.iter().map(|child| {
            //     child.as_str().unwrap().to_string()
            // }).collect::<Vec<String>>();
            let i = id;
            id = id + 1;
            SerializedAttackStep {
                id: i, 
                name: name.clone(), 
                ttc,
                step_type
            }
        })
        .collect::<Vec<SerializedAttackStep>>();

    let name2index = attack_steps
        .iter()
        .map(|s| (s.name.clone(), s.id))
        .collect::<HashMap<String, u64>>();

    let edges = attack_graph.iter().fold(HashSet::new(), |mut edges, step| {
        let name = step.get("id").unwrap().as_str().unwrap().to_string();
        let children = match step.get("children") {
            Some(c) => c.as_sequence().unwrap(),
            None => panic!("No 'children' field for {}", name),
        };

        for child in children.iter() {
            let child_id = *name2index
                .get(&child.as_str().unwrap().to_string())
                .unwrap();
            let edge = (name2index[&name], child_id);
            edges.insert(edge);
        }
        edges
    });

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
        .filter_map(|step| match defenses.contains(&step.name) {
            true => Some(step.id),
            false => None,
        })
        .collect::<Vec<u64>>();

    let flag_ids = attack_steps
        .iter()
        .filter_map(|step| match flags.contains(&step.name) {
            true => Some(step.id),
            false => None,
        })
        .collect::<Vec<u64>>();

    let entry_ids = attack_steps
        .iter()
        .filter_map(|step| match entry_points.contains(&step.name) {
            true => Some(step.id),
            false => None,
        })
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
    use std::{collections::HashSet, io::Write};

    use crate::loading::load_graph_from_yaml;

    use super::load_graph_from_json;

    #[test]
    fn load_graph_from_file() {
        let filename = "graphs/four_ways.yaml";
        let attackgraph = load_graph_from_yaml(filename);
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

        let graphviz = attackgraph.graph.to_graphviz();
        let mut file = std::fs::File::create("test.dot").unwrap();
        file.write_all(graphviz.as_bytes()).unwrap();
        file.flush().unwrap();
    }

    #[test]
    fn load_mal_graph() {
        let filename = "mal/atkgraph_2app_2cr_1net_1swvuln.json";
        let attack_steps = load_graph_from_json(filename);

        let edges: HashSet<(&String, &String)> = attack_steps
            .iter()
            .map(|step| {
                step.links
                    .iter()
                    .map(|child| (&step.id, child))
                    .collect::<HashSet<(&String, &String)>>()
            })
            .flatten()
            .collect();

        let attack_graph = AttackGraph::new(attack_steps, edges, vec![], vec![], vec![]);

        // let entry_points = attack_steps
        //     .iter()
        //     .filter(|step| step.atkname == "physicalAccess")
        //     .collect::<Vec<&MALAttackStep>>();

        // let discovered_assets: HashSet<String> = entry_points
        //     .iter()
        //     .map(|step|format!("{}:{}", step.objclass, step.objid).to_string())
        //     .collect::<HashSet<String>>();

        // let compromised_steps: HashSet<&MALAttackStep> = HashSet::new();

        // let not_done = true;
        // let user_selection: i32;
        // while not_done {
        //     println!("Discovered assets: ");
        //     println!("{:?}", discovered_assets);

        //     println!("Select an asset to examine:");
        //     discovered_assets.iter().enumerate().for_each(|(i, asset)| {
        //         println!("{}: {}", i, asset);
        //     });
            
        //     let mut line = String::new();
        //     let selection = std::io::stdin().read_line(&mut line).unwrap();
        // }
    }
}