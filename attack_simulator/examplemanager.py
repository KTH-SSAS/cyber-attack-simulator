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

def get_paths_to_graphs() -> Dict[str, str]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    examples = {}
        
    # walk through current folder to find valid domains
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f.endswith('.json'):
                examples[Path(f).stem] = os.path.join(dirpath, f)
               
    return examples

def get_graphviz_for_graph(graph_name) -> str:
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