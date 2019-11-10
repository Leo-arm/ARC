# Main driver application for one case of the ARC test
# Selected files:
# ../data/training/d364b489.json pattern around hole
# ../data/training/dbc1a6ce.json draw line between holes
# ../data/training/b60334d2.json pattern around holes
# Additional test cases:
# ../data/training/ce22a75a.json pattern around holes
# ../data/training/6cdd2623.json draw lines between corner holes


import sys
import os
import arc

def run(filename):
    a = arc.Arc(filename)
    # TODO(Leo): solve should take a grid as param as per assignment
    if a.solve():
        print(filename)
    return 0

# Run with this parameter
# ../data/training/*.json
# and that gets you all observations with holes in them.
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <arc task name>. Stopping.")
        sys.exit(-1)

    import glob
    for filename in glob.glob(sys.argv[1]):
        print(f"\nFile: {filename}")
        run(filename)

    # sys.exit(run(sys.argv[1]))