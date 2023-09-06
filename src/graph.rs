use std::collections::HashSet;
use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;
use std::hash::Hasher;

pub struct Graph<T, I> {
	pub nodes: HashMap<I, Node<T, I>>,
	pub edges: HashSet<(I, I)>,
}

impl<T, I:Eq + Hash> Graph<T, I> where I : Display, T: std::fmt::Display {
	
	pub fn get_data(&self, id: &I) -> Option<&T> {
		return match self.nodes.get(id) {
			Some(node) => Some(&node.data),
			None => None,
		}
	}

	pub fn children(&self, id: &I) -> Vec<&Node<T, I>> {
		return self.edges.iter().filter(|(parent, _)| parent == id).map(|(_, child)| self.nodes.get(child).unwrap()).collect();
	}

	pub fn parents(&self, id: &I) -> Vec<&Node<T, I>> {
		return self.edges.iter().filter(|(_, child)| child == id).map(|(parent, _)| self.nodes.get(parent).unwrap()).collect();
	}

	pub fn edges_to_graphviz(&self) -> String {
		self.edges.iter().map(|(parent, child)| format!("{} -> {}", parent, child)).collect::<Vec<String>>().join("\n")
	}
	
	pub fn nodes_to_graphviz(&self) -> String {
		self.nodes.iter().map(|(id, node)| format!("{} [label=\"{}\"]", id, node.data)).collect::<Vec<String>>().join("\n")
	}

	pub fn to_graphviz(&self) -> String {
		return format!("{}\n{}\n{}\n{}",
			"digraph {",
			self.nodes_to_graphviz(),
			self.edges_to_graphviz(),
			"}"
		);
	}
}

pub struct Node<T, I> {
	pub id: I,
	pub data: T,
}

impl<T, I> Hash for Node<T, I> where I: Hash {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.id.hash(state);
	}
}

impl <T, I> PartialEq for Node<T, I> where I: PartialEq {
	fn eq(&self, other: &Self) -> bool {
		self.id.eq(&other.id)
	}
}