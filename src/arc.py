# Main class for the ARC solver
# Command line: /data/training/dbc1a6ce.json ../data/training/b60334d2.json

import io
import operator
from util import Util
from collections import defaultdict
from functools import reduce
from itertools import combinations
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pprint

pp = pprint.PrettyPrinter(indent=4).pprint

class Cell(object):
    """
    Immutable cell object representing a grid cell and its colour
    """
    def __init__(self, g, x, y, c):
        self._grid = g
        self._cell = (x, y, c)

    @property
    def x(self):
        return self._cell[0]

    @property
    def y(self):
        return self._cell[1]

    @property
    def c(self):
        return self._cell[2]

    @property
    def g(self):
        return self._grid

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.c == other.c

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if self.y < other.y:
            return True
        return self.x < other.x

    def __hash__(self):
        """Assuming small grids < 256 in height or width here"""
        return self.x << 16 + self.y << 8 + self.c

    def __str__(self):
        return f"cell({self.x},{self.y} c={self.c})"

    def neighbours(self):
        """
        Iterate over the eight neighbours of this cell. At the edge of the
        grid we may have fewer than eight neighbours.
        """
        for y in range(self.y - 1, self.y + 2):
            for x in range(self.x - 1, self.x + 2):
                if self.x != x or self.y != y:
                    neighbour = self.g.at(x, y)
                    if neighbour:
                        yield neighbour

class GridShape(set):
    """
    A shape on the grid
    """

    def __repr__(self):
        size = self.size()
        g = Grid.from_cells(size[0], size[1], 0, self, True)
        return g.__str__()

    def contains(self, other):
        """Return True if other is entirely contained with self"""
        if isinstance(other, Cell):
            return other in self
        return (self & other) == other

    def size(self):
        """
        Return the size of an object as an x, y tuple
        """
        first = next(iter(self))
        minx = reduce(lambda v, c: v if v < c.x else c.x, self, first.x)
        maxx = reduce(lambda v, c: v if v > c.x else c.x, self, first.x)
        miny = reduce(lambda v, c: v if v < c.y else c.y, self, first.y)
        maxy = reduce(lambda v, c: v if v > c.y else c.y, self, first.y)
        return ((maxx - minx + 1), (maxy - miny + 1))

    def center(self):
        """
        Return the center of the shape as an x, y tuple
        """
        size = self.size()
        return (size[0] // 2, size[1] // 2)

class Grid(object):
    # TODO(Leo): change colour output to False before submitting
    def __init__(self, grid, colour=True):
        # Change whatever comes in into a 2d array
        self._grid = [[x if x else 0 for x in y] for y in grid]
        # Debug in full colour :-)
        self.colour = self.ansi if colour else self.nocolour

    @classmethod
    def from_json(cls, grid, colour=True):
        return cls(grid, colour)

    @classmethod
    def from_size(cls, x_size, y_size, bg, colour=True):
        g = [[bg for x in range(x_size)] for y in range(y_size)]
        return cls(g, colour)

    @classmethod
    def from_cells(cls, x_size, y_size, bg, shape, colour=True):
        """
        Create a grid of a certain size and add the shape to it.
        Mainly for debug.
        """
        assert x_size > 0
        assert y_size > 0
        g = Grid.from_size(x_size, y_size, bg, colour)
        g.add_shape((x_size + 1) // 2, (y_size + 1) // 2, shape)
        return g

    def __str__(self):
        sio = io.StringIO()
        print(" ", ''.join([f" {str(i % 10)} " for i in range(len(self._grid[0]))]), file=sio)
        for i, row in enumerate(self._grid):
            # TODO(Leo): put the space back in before submitting
            print(i % 10, ''.join([self.colour(s) for s in row]), file=sio)
        return sio.getvalue()

    def nocolour(self, i):
        return str(i)

    def ansi(self, i):
        """
        Print the grid in colour for easy debugging
        """
        return f"\033[1;30;{40+i}m {str(i)} \033[0m"

    def at(self, x, y):
        """
        Return the cell at the coordinates
        """
        if y >= 0 and y < len(self._grid):
            if x >=0 and x < len(self._grid[0]):
                return Cell(self, x, y, self._grid[y][x])
        return None

    def add_shape(self, x, y, shape):
        """
        Add a shape centered on the x, y coordinates
        """
        shape_x, shape_y = shape.center()
        for cell in shape:
            # Awkward as the shape cell coordinates have absolute coordinates
            xpos = x + (cell.x - shape_x)
            ypos = y + (cell.y - shape_y)
            if xpos >= 0 and xpos < len(self._grid[0]) and \
                ypos >= 0 and ypos < len(self._grid):
                self._grid[ypos][xpos] = cell.c

    def cells(self):
        """
        Iterate over cells from top left to bottom right
        """
        for y, row in enumerate(self._grid):
            for x, c in enumerate(row):
                yield(Cell(self, x, y, c))

    def segment_connected(self, objs, obj, cell, bg):
        """Segment an object from the background and connect the shape.
        Rather inefficient but works all the same"""
        def accept(cell):
            """Determine if this cell is not already in some object or the
            wrong background colour"""
            for obj in objs:
                if cell in obj:
                    return False
            return cell.c != bg

        if accept(cell):
            obj.add(cell)
            for neighbour in cell.neighbours():
                self.segment_connected(objs, obj, neighbour, bg)
        return obj

    def nof_objects(self):
        freq = defaultdict(int)
        for cell in self.cells():
            freq[cell.c] += 1
        return len(freq.keys())

    def holes(self):
        """
        Return a list of the holes
        """
        # TODO(Leo): larger holes
        holes = []
        for cell in self.cells():
            for neighbour in cell.neighbours():
                if cell.c == neighbour.c:
                    break
            else:
                # Neighbours are not the same as this cell
                hole = GridShape()
                hole.add(cell)
                holes.append(hole)
        return holes

    def object_shape(self):
        """
        Find the shape of the objects. It is assumed that
        the most common colour is the background.
        Return a list of objects.
        """
        freq = defaultdict(int)
        for cell in self.cells():
            freq[cell.c] += 1
        bg = max(freq.items(), key=operator.itemgetter(1))[0]
        # Now segment the objects out of the background. This fails for
        # connected objects.
        objs = []
        for cell in self.cells():
            new_obj = GridShape()
            # Add to known objects so check for inclusion also works for this
            # new object
            objs.append(new_obj)
            obj = self.segment_connected(objs, new_obj, cell, bg)
            if not len(obj):
                # Nothing in it so remove it.
                del objs[-1]
        # List of detected objects
        return objs

    def grid_size(self):
        """
        Return a tuple for the grid size
        """
        return (len(self._grid[0]), len(self._grid))

class Observation(object):
    def __init__(self, json):
        self.input = Grid.from_json(json['input'])
        self.output = Grid.from_json(json['output'])

    def __str__(self):
        return f"Input\n{str(self.input)}\nOutput:\n{str(self.output)}"

    def holes(self, from_input):
        fr = self.input if from_input else self.output
        return fr.holes()

    def nof_objects(self, from_input):
        fr = self.input if from_input else self.output
        return fr.nof_objects()

    def object_shape(self, from_input):
        fr = self.input if from_input else self.output
        return fr.object_shape()

    def grid_size(self, from_input):
        fr = self.input if from_input else self.output
        return fr.grid_size()

class Task(object):

    def __init__(self, json):
        # Construct objects from json
        self._obs = []
        for obs in json['train']:
            self._obs.append(Observation(obs))
        self._test = []
        for test in json['test']:
            self._test.append(Observation(test))
        # Input feature description
        self.input_feat = {}
        # Output feature description
        self.output_feat = {}
        # Common features between input and output
        self.common_feat = {}

    def __str__(self):
        s = "Observations\n"
        for obs in self._obs:
            s += str(obs)
        s += "Tests\n"
        for test in self._test:
            s += str(test)
        return s

    def _find_features(self):
        """Run each of the features and collect the results for each of the
        examples, for both the inputs and the test cases.
        This results in the dict with as key the name of the
        detector and as value a list with one entry per example.
        """
        features = {
            'holes'       : self.holes,
            'nof_objects' : self.nof_objects,
            'object_shape': self.object_shape,
            'grid_size'   : self.grid_size,
        }
        def collect_features(f, source, from_input):
            """Collect features from inputs or outputs,
            sourced from observations or test grids"""
            for name, feature in features.items():
                feature(f, source, from_input)

        self.input_feat = {}
        self.output_feat = {}
        self.test_feat = {}
        collect_features(self.input_feat, self._obs, True)
        collect_features(self.output_feat, self._obs, False)
        collect_features(self.test_feat, self._test, True)

    def _add_patterns(self):
        """
        Using the gathered information, add patterns deduced from the
        information for each of the input and the output.
        E.g. if there are one, then two, then three objects
        in the inputs, see if it makes sense to add a case for four."""
        pass

    def _find_common_patterns(self):
        """
        Compare the detected patterns in input and output and deduce what
        the rules are that may be applied to convert the input into the output
        """

        # The input_feat, output_feat are both a dictionary with as key and
        # attribute. The value is a list, one for each observation, with either
        # a number (e.g. number of objects) or a list of objects in the grid.

        def common_keys():
            """Return the common keys (attributes) between inputs and outputs"""
            return [key for key in self.input_feat.keys() if key in self.output_feat]

        def is_equal(a, b):
            def same_type(c, d):
                return type(c).__name__ == type(d).__name__

            if same_type(a, b):
                if isinstance(a, list):
                    if len(a) == len(b):
                        for d, e in zip(a, b):
                            if not is_equal(d, e):
                                return False
                    return True
                # Let Python figure it out, e.g. call rich comparision functions
                return a == b

        # Add the fact that the output grid size is equal to the input grid size
        match = []
        if 'grid_size' in self.input_feat and 'grid_size' in self.output_feat:
            for inputs, outputs in zip(self.input_feat['grid_size'], self.output_feat['grid_size']):
                if inputs != outputs:
                    break
            else:
                self.common_feat['same_grid_size'] = self.input_feat['grid_size'][0]


        # Add the fact that the number of objects is the same in input and output
        match = []
        if 'object_shape' in self.input_feat and 'object_shape' in self.output_feat:
            for inputs, outputs in zip(self.input_feat['object_shape'], self.output_feat['object_shape']):
                match.append(len(inputs) == len(outputs))
        if len(match):
            self.common_feat['same_nof_objects'] = match

        # Add the fact that holes in the input match with holes in the output
        match = []
        if 'holes' in self.input_feat and 'object_shape' in self.output_feat:
            # For each observation
            for inputs, outputs in zip(self.input_feat['holes'], self.output_feat['object_shape']):
                # For each hole in observation, and for each output shape
                for input in inputs:
                    # For each shape in output
                    found = False
                    for output in outputs:
                        if output.contains(input):
                            found = True
                            break
                    if found:
                        # Hole matches with shape
                        match.append(output)
                        continue
                if len(match):
                    self.common_feat['hole_inside_object'] = match

        # Add the fact that the output shapes are all the same
        if 'object_shape' in self.output_feat:
            # For each observation
            for obs in self.output_feat['object_shape']:
                for a, b in combinations(obs, 2):
                    if not a.contains(b) or not b.contains(a):
                        break
            else:
                # They are all the same so pick the first
                self.common_feat['same_output_shapes'] = self.output_feat['object_shape'][0][0]

        # Add the fact that the input shapes are all the same
        # TODO(Leo): refactor this with the above rule
        if 'object_shape' in self.input_feat:
            # For each observation
            for obs in self.input_feat['object_shape']:
                for a, b in combinations(obs, 2):
                    if not a.contains(b) or not b.contains(a):
                        break
            else:
                # They are all the same so pick the first
                self.common_feat['same_input_shapes'] = self.output_feat['object_shape'][0][0]

        # Add the fact that the input has some same attribute as the output
        for attr in common_keys():
            if is_equal(self.input_feat[attr], self.output_feat[attr]):
                self.common_feat[attr] = self.input_feat[attr]

        # TODO(Leo): remove
        print(self.common_feat)

    def _find_program(self):
        """
        Given common attributes, find an algorithm to go from input to output
        """
        program = []
        if 'same_output_shapes' in self.common_feat and \
           'same_input_shapes' in self.common_feat and \
            'same_nof_objects' in self.common_feat:
            # This should be broken down into smaller steps, e.g.
            # 1. take shape,
            # 2. for each input shape location
            # 3.     copy shape to equivalent location.


            # should have the shape attached to same_output_shapes
            # this should be modeled like the other features earlier on.

            program.append(self.Isa.copy_output_shape_to_input_shape_pos)

        return program

    class Isa(object):
        """
        Instruction set for executing a 'program' that creates an output from
        an input image.
        """
        @classmethod
        def copy_output_shape_to_input_shape_pos(self, answer, input_data, program_data):
            for obs in input_data['holes']:
                # TODO(Leo): assuming one test observation
                for input_shape in obs:
                    x, y = input_shape.center()
                    answer.add_shape(x, y, program_data['same_output_shapes'])

    def execute(self, program, input_data, program_data):
        print("Test image")
        print(repr(self._test))

        x, y = program_data['grid_size'][0]
        answer = Grid.from_size(x, y, 0, True)
        for instruction in program:
                instruction(answer, input_data, program_data)
        print("------------\n")
        print(answer)

    def solve(self):
        """
        Apply a number of recognition methods to find out what the salient
        features of this task is.
        """
        # TODO(Leo): remove print statement
        print(self)

        self._find_features()

        # self._pca(res)

        # Now we have basic information of the inputs and the test cases.
        # Now add more information by adding patterns. For example, if the
        # inputs each have one more hole in them, infer that the test case
        # might have one more again.
        self._add_patterns()

        # Compare the inputs and the test cases to see what patterns are
        # common. Throw out the ones that are not to reduce the search space.
        self._find_common_patterns()

        # Search for a set of transformations that map the inputs to the
        # outputs. Store these as a set of instructions to apply to the test.
        self._program = self._find_program()

        print(self._program)

        # test = self._find_features(from_test)

        self.execute(self._program, self.test_feat, self.common_feat)

    def nof_objects(self, res, source, from_input):
        """
        Count the number of objects
        Add a list with an entry per observation
        """
        l = []
        for obs in source:
            l.append(obs.nof_objects(from_input))
        res['nof_objects'] = l

    def holes(self, res, source, from_input):
        """
        Count the number of 'holes'
        Add a list with an entry per observation
        """
        l = []
        for obs in source:
            l.append(obs.holes(from_input))
        res['holes'] = l

    def object_shape(self, res, source, from_input):
        """
        Capture the object shapes
        Add a list with an entry per observation
        """
        l = []
        for obs in source:
            l.append(obs.object_shape(from_input))
        res['object_shape'] = l

    def grid_size(self, res, source, from_input):
        """
        Determine the grid size for inputs and outputs
        Grid size is a tuple x, y
        """
        l = []
        for obs in source:
            l.append(obs.grid_size(from_input))
        res['grid_size'] = l


class Arc(object):

    def __init__(self, filename):
        self._task = Task(Util.load(filename))

    def solve(self):
        return self._task.solve()
