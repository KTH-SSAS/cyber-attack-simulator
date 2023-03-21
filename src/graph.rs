use std::collections::HashSet;
use std::collections::HashMap;

pub type NodeID = u64; // Global ID of a node

pub struct Graph<T> {
	pub nodes: HashMap<NodeID, Node<T>>,
}

impl<T> Graph<T> {
	
	pub fn get_data(&self, id: &NodeID) -> Option<&T> {
		return match self.nodes.get(id) {
			Some(node) => Some(&node.data),
			None => None,
		}
	}

	pub fn children(&self, id: &NodeID) -> Option<&HashSet<NodeID>> {
		return match self.nodes.get(id) {
			Some(node) => Some(&node.children),
			None => None,
		}
	}
}

pub struct Node<T> {
	pub id: NodeID,
	pub data: T,
	pub parents: HashSet<NodeID>,
    pub children: HashSet<NodeID>,
}