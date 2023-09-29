import json

merged_vocab = []

with open("corelang_vocab.json") as f:
	vocab = json.load(f)


for key, subvocab in vocab.items():
	merged_vocab += subvocab

with open("corelang_vocab_merged.json", "w") as f:
	json.dump(merged_vocab, f, indent=2)
