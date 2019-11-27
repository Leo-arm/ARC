# Main driver application for one case of the ARC test
# Usage:
#
# ptyhon <scriptname> <sample name>
#
# This will run the script, train and print the training pairs.
# If a separate test needs to be run, then the script has to be run first to do
# the training, passing in the training file in json format. The training
# data will be stored in the Arc object. The test can then be run by
# calling the solve() function in this file, passing in a grid in json format.
# Upon reading the assignment again I realise that you could read the
# requirements in a couple of different ways. I took it as the training
# happens on all examples, and then the test can be run on a single input.
# This is how it is implemented.

import sys
import arc

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <arc task name>. Stopping.")
        sys.exit(-1)

    exit(arc.solve(sys.argv[1]))

