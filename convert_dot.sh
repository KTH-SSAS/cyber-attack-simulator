#!/bin/bash
#convert all dot files to png
for i in *.dot; do
	dot -Tpng $i -o $i.png
done