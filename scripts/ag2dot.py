#!/usr/bin/env python3
import json
import sys

nodes = json.loads(sys.stdin.read())

output = "digraph G {\n"

for n in nodes:
    output += f"\"{n['id']}\" [label=\"{n['id']}\"]\n"

for n in nodes:
    for c in n["children"]:
        output += f"\"{n['id']}\" -> \"{c}\"\n"

output += "}"
print(output)
