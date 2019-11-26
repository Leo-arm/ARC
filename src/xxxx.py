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
import glob
import arc

def run(filename):
    a = arc.Arc(filename)
    # TODO(Leo): solve should take a grid as param as per assignment
    if a.solve():
        print(filename)
    return 0

# TODO(Leo): This file's filename needs to follow the naming guidelines.
# Three ways to run this:
# ../data/training/*.json
# ../data/training/d364b489.json ../data/training/dbc1a6ce.json ../data/training/b60334d2.json
# ../data/training/d364b489.json
if __name__ == '__main__':
    import doctest
    doctest.testmod()

    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <arc task name>. Stopping.")
        sys.exit(-1)

    # If this is a filename, then run this filename. If it is a glob
    # then run all files. If we have multiple filenames on the cli, then
    # do this for all of them.
    for args in sys.argv[1:]:
        for filename in glob.glob(args):
            run(filename)
    exit(0)
