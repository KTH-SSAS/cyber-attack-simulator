use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

use cached::proc_macro::cached;
use cached::SizedCache;

pub struct Graph<T, I>
where
    I: Ord,
{
    pub nodes: BTreeMap<I, Node<T, I>>,
    pub edges: Vec<(I, I)>,
}

#[cached(
    type = "SizedCache<String, HashSet<usize>>",
    create = "{ SizedCache::with_size(100) }",
    convert = r#"{ format!("{}", id) }"#
)]
fn parents_cached<I>(edges: HashSet<(usize, usize)>, id: usize) -> HashSet<usize> {
    return edges
        .iter()
        .filter(|(_, child)| *child == id)
        .map(|(parent, _)| parent.clone())
        .collect();
}

fn format_attributes(attributes: Option<&Vec<(String, String)>>) -> String {
    match attributes {
        Some(attributes) => {
            return attributes
                .iter()
                .map(|(key, value)| format!("{}=\"{}\"", key, value))
                .collect::<Vec<String>>()
                .join(", ")
        }
        None => return "".to_string(),
    }
}

impl<T, I: Eq + Hash> Graph<T, I>
where
    I: Debug + Ord,
    T: Display,
{
    /*
    pub fn get_data(&self, id: &I) -> Option<&T> {
        return match self.nodes.get(id) {
            Some(node) => Some(&node.data),
            None => None,
        };
    }
    */

    pub fn children(&self, id: &I) -> Vec<&Node<T, I>> {
        return self
            .edges
            .iter()
            .filter_map(|(parent, child)| match parent == id {
                true => Some(self.nodes.get(child).unwrap()),
                false => None,
            })
            .collect();
    }

    pub fn parents(&self, id: &I) -> Vec<&Node<T, I>> {
        return self
            .edges
            .iter()
            .filter_map(|(parent, child)| match child == id {
                true => Some(self.nodes.get(parent).unwrap()),
                false => None,
            })
            .collect();
    }

    pub fn edges_to_graphviz(&self) -> String {
        self.edges
            .iter()
            .map(|(parent, child)| format!("\"{:?}\" -> \"{:?}\"", parent, child))
            .collect::<Vec<String>>()
            .join("\n")
    }

    pub fn nodes_to_graphviz(&self, attributes: &HashMap<&I, Vec<(String, String)>>) -> String {
        self.nodes
            .iter()
            .map(|(id, _node)| format!("\"{:?}\" [{}]", id, format_attributes(attributes.get(id))))
            .collect::<Vec<String>>()
            .join("\n")
    }

    pub fn to_graphviz(&self, attributes: Option<&HashMap<&I, Vec<(String, String)>>>) -> String {
        let binding = self
            .nodes
            .iter()
            .map(|(id, n)| (id, vec![("label".to_owned(), n.data.to_string())]))
            .collect();
        let attributes = match attributes {
            Some(x) => x,
            None => &binding,
        };
        return format!(
            "{}\n{}\n{}\n{}\n{}",
            "digraph {",
            "size=\"50,50\"",
            self.nodes_to_graphviz(attributes),
            self.edges_to_graphviz(),
            "}"
        );
    }
}

pub struct Node<T, I> {
    pub id: I,
    pub data: T,
}

impl<T, I> Hash for Node<T, I>
where
    I: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T, I> PartialEq for Node<T, I>
where
    I: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}

impl<T, I> Eq for Node<T, I> where I: Eq {}
