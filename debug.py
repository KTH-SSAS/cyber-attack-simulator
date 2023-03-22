#!/usr/bin/env python3
import sys
from pathlib import Path

from run import run

if len(sys.argv) < 2:
    print("Usage: debug.py <config_file>")
    sys.exit()
config_file = Path(sys.argv[1])
run(config_file, num_workers=0, debug=True)
