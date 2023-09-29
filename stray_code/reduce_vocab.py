import json

with open("corelang_vocab.json") as f:
    vocab = json.load(f)

words_to_strip = ["successful", "attempt"]

new_vocab = {key: set() for key in vocab.keys()}
for key, subvocab in vocab.items():
    for word in subvocab:
        find_prefixes = lambda word: [
            prefix for prefix in words_to_strip if word.startswith(prefix)
        ]
        prefixes = find_prefixes(word)
        if prefixes:
            for p in prefixes:
                new_vocab[key].add(word[len(p) :])
        else:
            new_vocab[key].add(word)

for key, subvocab in new_vocab.items():
    new_vocab[key] = list(subvocab)


with open("corelang_vocab_reduced.json", "w") as f:
    json.dump(new_vocab, f, indent=2)
