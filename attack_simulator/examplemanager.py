import os
from pathlib import Path
from typing import Dict
import json
import os
from pathlib import Path
from typing import Dict, List


def available_graphs() -> List[str]:
    paths = get_paths_to_graphs()
    return list(paths.keys())


def show_graph(graph_name: str) -> None:
    graphviz = get_graphviz_for_graph(graph_name)
    import graphviz as gv

    graph = gv.Source(graphviz)
    graph.render(filename=graph_name, format="png", cleanup=True)
    graph.view()

def get_vocab_for_graphs() -> list:
    vocab = set()

    for name, path in get_paths_to_graphs().items():

        if name == "corelang":
            continue
        
        with open(path, "r") as f:
            graph = json.load(f)

            for n in graph:
                vocab.add(n["name"])
                vocab.add(n["asset"].split(":")[0])

    sorted_vocab = sorted(vocab)
    return sorted_vocab


def get_paths_to_graphs() -> Dict[str, str]:

    path = Path(__file__).parent.absolute() / "graphs"
    examples = {}

    for file in path.glob("*.json"):
        examples[file.stem] = str(file)

    return examples


def get_graphviz_for_graph(graph_name: str) -> str:
    paths = get_paths_to_graphs()
    if graph_name not in paths:
        raise ValueError(f"Graph {graph_name} not found.")

    with open(paths[graph_name], "r", encoding="utf8") as f:
        nodes = json.load(f)

    output = "digraph G {\n"

    for n in nodes:
        output += f"\"{n['id']}\" [label=\"{n['id']}\"]\n"

    for n in nodes:
        for c in n["children"]:
            output += f"\"{n['id']}\" -> \"{c}\"\n"

    output += "}"
    return output


if __name__ == "__main__":
    vocab = get_vocab_for_graphs()
