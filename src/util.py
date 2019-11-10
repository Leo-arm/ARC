# Utilities for loading, converting, printing of json ARC files.

import json

class Util(object):

    def load(filename):
        """
        Load an ARC Json file, convert to an array and return.
        """
        with open(filename) as fp:
            data = json.load(fp)
        return data
        
