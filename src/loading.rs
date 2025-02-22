use std::{collections::HashMap, fs::File, io::BufReader};

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use std::collections::HashSet;

use crate::{attackgraph::AttackGraph, vocab::Vocab};

pub type IOResult<T> = std::result::Result<T, IOError>;

#[derive(Debug, Clone)]
pub struct IOError {
    error: String,
}

impl std::fmt::Display for IOError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.error)
    }
}

#[derive(Serialize, Deserialize)]
pub struct TTC {
    #[serde(rename = "type")]
    ttc_type: String,
    name: String,
    arguments: Vec<f64>,
}

/*
#[derive(Serialize, Deserialize)]
pub struct MALAttackStep {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
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
*/

#[derive(Serialize, Deserialize)]
pub struct MALAttackStep {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    pub name: String,
    ttc: Option<TTC>,
    children: Vec<String>,
    parents: Option<Vec<String>>,
    compromised_by: Vec<String>,
    pub asset: String,
    defense_status: Option<String>,
    mitre_info: Option<String>,
    existence_status: Option<String>,
}

fn load_vocab_from_json(filename: &str) -> Result<Vocab, IOError> {
    let contents = std::fs::read_to_string(filename);

    let contents = match contents {
        Ok(c) => c,
        Err(e) => {
            return Err(IOError {
                error: format!("Could not read vocab file: {}. {}", filename, e),
            })
        }
    };

    let vocab =
        serde_json::from_str(&contents).and_then(|words: Vec<String>| Ok(Vocab::from(words)));

    match vocab {
        Ok(v) => Ok(v),
        Err(e) => Err(IOError {
            error: format!("Could not parse json in vocab file: {}", e),
        }),
    }
}

fn create_vocab_from_steps(steps: &Vec<MALAttackStep>) -> Vocab {
    let mut unique_words = HashSet::new();
    for step in steps {
        let (asset, _) = match step.asset.split_once(":") {
            Some((asset, _)) => (asset, ""),
            None => (step.asset.as_str(), ""),
        };
        unique_words.insert(asset.to_string());
        unique_words.insert(step.name.clone());
    }

    let unique_words: Vec<String> = unique_words.iter().sorted().cloned().collect();

    return Vocab::from(unique_words);
}

fn save_vocab_to_file(vocab: &Vocab, vocab_filename: Option<&str>) -> IOResult<()> {
    // Save the vocab to a file
    let vocab_filename = "generated_vocab.json";
    let vocab_json = match serde_json::to_string_pretty(&vocab) {
        Ok(v) => v,
        Err(e) => {
            return Err(IOError {
                error: format!("Could not serialize vocab to json: {}", e),
            });
        }
    };

    match std::fs::write(vocab_filename, vocab_json) {
        Ok(_) => (),
        Err(e) => {
            return Err(IOError {
                error: format!("Could not write vocab to file: {}", e),
            });
        }
    };

    return Ok(());
}

fn create_and_save_vocab_from_steps(steps: &Vec<MALAttackStep>) -> IOResult<Vocab> {
    let vocab = create_vocab_from_steps(steps);
    save_vocab_to_file(&vocab, None)?;
    return Ok(vocab);
}

pub(crate) fn load_graph_from_json(
    filename: &str,
    vocab_filename: Option<&str>,
    fnr: f64,
    fpr: f64,
) -> IOResult<AttackGraph<(usize, usize, usize)>> {
    let file = match File::open(filename) {
        Ok(f) => f,
        Err(e) => {
            return Err(IOError {
                error: format!("Could not find file: {}. {}", filename, e),
            })
        }
    };

    let reader = BufReader::new(file);
    let attack_steps: Vec<MALAttackStep> = match serde_json::from_reader(reader) {
        Ok(steps) => steps,
        Err(e) => {
            return Err(IOError {
                error: format!("Failed to parse list of attack steps: {}", e),
            })
        }
    };

    let child_edges: HashSet<(String, String)> = attack_steps
        .iter()
        .map(|step| {
            step.children
                .iter()
                .map(|child| (step.id.clone(), child.clone()))
                .collect::<HashSet<(String, String)>>()
        })
        .flatten()
        .collect();

    let parent_edges: HashSet<(String, String)> = attack_steps
        .iter()
        .filter_map(|step| match &step.parents {
            Some(parents) => Some((step.id.clone(), parents.clone())),
            None => None,
        })
        .map(|(id, parents)| {
            parents
                .iter()
                .map(|parent| (parent.clone(), id.clone()))
                .collect::<HashSet<(String, String)>>()
        })
        .flatten()
        .collect();

    let edges = child_edges
        .union(&parent_edges)
        .map(|(parent, child)| (parent.clone(), child.clone()))
        .collect::<HashSet<(String, String)>>();



    let nonexistant_nodes = attack_steps
        .iter()
        .filter_map(|step| match &step.existence_status {
            Some(status) => match status.as_str() {
                "False" => Some(step.id.clone()),
                _ => None,
            },
            None => None,
        })
        .collect::<Vec<String>>();

    let attack_steps = attack_steps
        .into_iter()
        .filter(|step| !nonexistant_nodes.contains(&step.id))
        .collect::<Vec<MALAttackStep>>();

    let edges = edges
        .into_iter()
        .filter(|(parent, child)| {
            !nonexistant_nodes.contains(parent) && !nonexistant_nodes.contains(child)
        })
        .collect::<HashSet<(String, String)>>();

    let entry_points = attack_steps
        .iter()
        .filter_map(|step| match step.name.as_str() {
            "firstSteps" => Some(step.id.clone()),
            _ => None,
        })
        .collect::<Vec<String>>();

    let flags = attack_steps
        .iter()
        .filter_map(|step| match step.name.as_str() {
            "gather" => Some(step.id.clone()),
            "modify" => Some(step.id.clone()),
            "read" => Some(step.id.clone()),
            "deny" => Some(step.id.clone()),
            _ => None,
        })
        .collect::<Vec<String>>();

    if entry_points.len() == 0 {
        return Err(IOError {
            error: "The graph does not seem to have any entry points.".to_string(),
        });
    }

    let vocab = match vocab_filename {
        None => create_and_save_vocab_from_steps(&attack_steps),
        Some(f) => load_vocab_from_json(f)
    };

    let vocab = match vocab {
        Ok(v) => v,
        Err(e) => {
            return Err(IOError {
                error: format!("Could not load vocab: {}", e),
            });
        }
    };

    let attack_graph =
        AttackGraph::<u64>::new(attack_steps, edges, flags, entry_points, vocab, fpr, fnr);

    return Ok(attack_graph);
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::load_graph_from_json;

    #[test]
    fn load_mal_graph() {
        let filename = "attack_simulator/graphs/corelang.json";
        let vocab_filename = "mal/corelang_vocab_merged.json";
        let attack_graph = load_graph_from_json(filename, Some(vocab_filename), 0.0, 0.0).unwrap();

        let graphviz = attack_graph.to_graphviz(None);

        let mut file = std::fs::File::create("mal/attackgraph.dot").unwrap();
        file.write_all(graphviz.as_bytes()).unwrap();
        file.flush().unwrap();

        // run dot
        /*
        let output = std::process::Command::new("dot")
            .arg("-Tpng")
            .arg("mal/attackgraph.dot")
            .arg("-o")
            .arg("mal/attackgraph.png")
            .output()
            .expect("failed to execute process");
        */

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
