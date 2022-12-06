#!/usr/bin/env python3
from run import run
import sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit()
config_file = Path(sys.argv[1])
run(config_file, num_workers=0, debug=True)
