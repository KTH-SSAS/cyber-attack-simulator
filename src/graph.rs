use std::collections::HashSet;
use std::collections::HashMap;
use std::hash::Hash;

pub struct Graph<T, I> {
	pub nodes: HashMap<I, Node<T, I>>,
	pub edges: HashSet<(I, I)>,
}

impl<T, I:Eq + Hash> Graph<T, I> {
	
	pub fn get_data(&self, id: &I) -> Option<&T> {
		return match self.nodes.get(id) {
			Some(node) => Some(&node.data),
			None => None,
		}
	}

	pub fn children(&self, id: &I) -> Option<&HashSet<I>> {
		return match self.nodes.get(id) {
			Some(node) => Some(&node.children),
			None => None,
		}
	}
}

pub struct Node<T, I> {
	pub id: I,
	pub data: T,
	pub parents: HashSet<I>,
    pub children: HashSet<I>,
}