# read tikz data from stdin and write dot data to stdout
# usage: python tikz2dot.py < tikz.tex > dot.dot

import sys
import re

# read tikz data from stdin
tikz = sys.stdin.read()
# with open("/home/jakob/threeways.tikz", "r") as f:
# 	tikz = f.read()

# remove comments
tikz = re.sub(r"(?m)%.*\n", "", tikz)

output = "digraph G {\n"

for l in tikz.splitlines():
    match = re.search(r"\{.*?\}", l)
    label = match.group(0)[1:-1] if match else ""
    match = re.search(r"\(.*?\)", l)
    node_id = match.group(0)[1:-1] if match else ""

    if label and node_id:
        output += node_id + f' [label="{label}"]' + "\n"


for l in tikz.splitlines():
    match = re.search(r"\((\d\d?)\.center\) to", l)
    from_ = match.group(1) if match else ""
    match = re.search(r"to \((\d\d?)\.center\)", l)
    to = match.group(1) if match else ""

    if to and from_:
        output += f"{from_} -> {to}" + "\n"

output += "}"
print(output)
