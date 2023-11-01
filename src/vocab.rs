use std::collections::HashMap;



pub(crate) struct Vocab {
	word2idx: HashMap<String, usize>,
	idx2word: HashMap<usize, String>,
}

impl Vocab {
	pub(crate) fn get_word(&self, idx: usize) -> String {
		match self.idx2word.get(&idx) {
			Some(word) => word.clone(),
			None => panic!("Index {} not found in vocab", idx),
		}
	}

	pub(crate) fn get_idx(&self, word: &str) -> usize {
		match self.word2idx.get(word) {
			Some(idx) => *idx,
			None => panic!("Word {} not found in vocab", word),
		}
	}

	pub(crate) fn export(&self) -> HashMap<String, usize> {
		self.word2idx.clone()
	}
	
}

impl From<Vec<String>> for Vocab {
	fn from(words: Vec<String>) -> Self {
		let word2idx = words
        .iter()
        .enumerate()
        .map(|(i, x)| (x.clone(), i))
        .collect();
		let idx2word = words
		.iter()
		.enumerate()
		.map(|(i, x)| (i, x.clone()))
		.collect();
		return Vocab {
			word2idx,
			idx2word
		}
	}
}

impl serde::Serialize for Vocab {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		self.word2idx.serialize(serializer)
	}
}