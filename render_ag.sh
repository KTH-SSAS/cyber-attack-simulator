#!/bin/bash
for name in ./attack_simulator/graphs/*.json; do
	echo $name
	./scripts/ag2dot.py < $name > ${name%.json}.dot
	dot -Tpng ${name%.json}.dot > ${name%.json}.png
	rm ${name%.json}.dot
done