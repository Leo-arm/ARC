# Main class for the ARC solver
# Student name(s): Leo Blonk
# Student ID(s): 19240143
# Source code available on github account Leo-arm.
# Command line: /data/training/dbc1a6ce.json ../data/training/b60334d2.json

import io
import sys
import glob
import json
import operator
from collections import defaultdict, Counter
from functools import reduce
from itertools import combinations
import pprint

# Debug in full colour if True, False for release as per requirements
# The printout is much clearer when this is switched on. This requires an
# ANSI compatible terminal.
g_fancy_output = False

# Global pretty printer for debug
pp = pprint.PrettyPrinter(indent=4).pprint

class Cell(object):
    """
    Immutable cell object representing a grid cell and its colour.
    """
    def __init__(self, g, x, y, c):
        self._grid = g
        self._cell = (x, y, c)

    @property
    def x(self):
        """Shorthand for x coordinate"""
        return self._cell[0]

    @property
    def y(self):
        """Shorthand for y coordinate"""
        return self._cell[1]

    @property
    def c(self):
        """Shorthand for colour"""
        return self._cell[2]

    @property
    def g(self):
        """The grid that this cell belongs to"""
        return self._grid

    def __eq__(self, other):
        if other:
            return self.x == other.x and self.y == other.y and self.c == other.c
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if self.y < other.y:
            return True
        return self.x < other.x

    def __hash__(self):
        """Assuming small grids < 256 in height or width"""
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

    def close_neighbours(self):
        """
        Iterate over the four neighbours of this cell, i.e. -1,0, +1,0
        0,-1 and 0,+1. At the edge of the grid we may have fewer than four
        neighbours.
        """
        for xx, yy in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
            neighbour = self.g.at(self.x + xx, self.y + yy)
            if neighbour:
                yield neighbour


    def hv_align_to(self, other):
        """
        Iterate over the cells that horizontally -or- vertically connect self to
        # other, excluding the center of self and other.
        """
        if self.x == other.x:
            start = self.y + 1 if self.y < other.y else other.y - 1
            end  = other.y if other.y > self.y else self.y
            for y in range(start, end):
                yield self.g.at(self.x, y)
        elif self.y == other.y:
            start = self.x + 1 if self.x < other.x else other.x - 1
            end  = other.x if other.x > self.x else self.x
            for x in range(start, end):
                yield self.g.at(x, self.y)


class GridShape(set):
    """
    A shape on a grid. Shapes are stored as a set of cells
    """

    def __repr__(self):
        if len(self):
            size = self.size()
            g = Grid.from_cells(size[0], size[1], 0, self, g_fancy_output)
            return g.__str__()
        return "GridShape(<empty>)"

    def normalise(self):
        """Adjust coordinates to be relative. That is to say, the first row
        and column of member cells are at 0."""
        minx, miny, maxx, maxy = self.extent()
        new_shape = GridShape()
        for c in self:
            new_shape.add(Cell(c.g, c.x - minx, c.y - miny, c.c))
        return new_shape

    def contains(self, other):
        """Return True if other is entirely contained with self.
        Made complicate by the fact that shape cell coordinates are absolute,
        that is abolute coordinates on the grid."""

        if isinstance(other, Cell):
            return other in self
        a = self.normalise()
        b = self.normalise()
        return (a & b) == b

    def extent(self):
        """Return the absolute extent of the shape"""
        first = next(iter(self))
        minx = reduce(lambda v, c: v if v < c.x else c.x, self, first.x)
        maxx = reduce(lambda v, c: v if v > c.x else c.x, self, first.x)
        miny = reduce(lambda v, c: v if v < c.y else c.y, self, first.y)
        maxy = reduce(lambda v, c: v if v > c.y else c.y, self, first.y)
        return (minx, miny, maxx, maxy)

    def pos(self):
        """Return the absolute position of the center of the shape on the grid"""
        minx, miny, maxx, maxy = self.extent()
        return ((maxx + minx) // 2, (maxy + miny) // 2)

    def size(self):
        """
        Return the size of an object as an x, y tuple
        """
        minx, miny, maxx, maxy = self.extent()
        return ((maxx - minx + 1), (maxy - miny + 1))

    def center(self):
        """
        Return the relative center of the shape as an x, y tuple
        """
        size = self.size()
        return (size[0] // 2, size[1] // 2)

    def on_edge(self, grid):
        """Answer if this object might be on an edge of the grid"""
        x, y = grid.grid_size()
        for cell in self:
            if cell.x <= 0 or cell.y <= 0:
                return True
            if cell.x >= x - 1 or cell.y >= y - 1:
                return True
        return False


class Grid(object):
    """
    A grid of cells.
    """

    # TODO(Leo): change colour output to False before submitting
    def __init__(self, grid, colour=False):
        # Change whatever comes in, into a 2d array
        self._grid = [[x if x else 0 for x in y] for y in grid]
        # Debug in full colour :-)
        self.colour = self.ansi if colour else self.nocolour

    @classmethod
    def from_json(cls, grid, colour=True):
        """
        Create a grid from a json specification
        """
        return cls(grid, colour)

    @classmethod
    def from_size(cls, x_size, y_size, bg, colour=False):
        """
        Create a grid of a specified size
        """
        g = [[bg for x in range(x_size)] for y in range(y_size)]
        return cls(g, colour)

    @classmethod
    def from_cells(cls, x_size, y_size, bg, shape, colour=False):
        """
        Create a grid of a certain size and add the shape to it.
        Mainly for debug.
        """
        assert x_size > 0
        assert y_size > 0
        g = Grid.from_size(x_size, y_size, bg, colour)
        shape = shape.normalise()
        g.add_shape((x_size + 1) // 2, (y_size + 1) // 2, shape)
        return g

    def __str__(self):
        sio = io.StringIO()
        if g_fancy_output:
            print(" ", ''.join([f" {str(i % 10)} " for i in range(len(self._grid[0]))]), file=sio)
        for i, row in enumerate(self._grid):
            # Use a space if not printing in colour, else no space. Makes it
            # easy on the eye for debug
            if g_fancy_output:
                sp = ''
                num = i % 10
            else:
                sp = ' '
                num = ""
            print(num , sp.join([self.colour(s) for s in row]), file=sio)
        return sio.getvalue()

    def nocolour(self, i):
        """Return a plain string for a cell"""
        return str(i)

    def ansi(self, i):
        """
        Print the cells in colour for easy debugging
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

    def put(self, cell):
        """
        Put a cell on the grid, with bounds checking
        """
        if cell.x >= 0 and cell.x < len(self._grid[0]) and \
                cell.y >= 0 and cell.y < len(self._grid):
            self._grid[cell.y][cell.x] = cell.c

    def add_shape(self, x, y, shape):
        """
        Add a shape centered on the x, y coordinates
        """
        # Awkward as the shape cell coordinates have absolute coordinates
        norm_shape = shape.normalise()
        adj_x, adj_y = norm_shape.center()
        for cell in norm_shape:
            xpos = x + cell.x - adj_x
            ypos = y + cell.y - adj_y
            self.put(Cell(self, xpos, ypos, cell.c))

    def cells(self):
        """
        Iterate over cells from top left to bottom right
        """
        for y, row in enumerate(self._grid):
            for x, c in enumerate(row):
                yield(Cell(self, x, y, c))

    def background(self):
        """
        Return the majority colour of the grid. This is assumed to be the
        background.
        """
        cnt = Counter();
        for cell in self.cells():
            cnt[cell.c] += 1
        return cnt.most_common(1)[0][0]

    def segment_connected(self, objs, obj, cell, bg):
        """
        Segment an object from the background and connect the shape.
        Rather inefficient but works all the same.
        """
        def accept(c):
            """Determine if this cell is not already in some object or the
            wrong background colour"""
            for obj in objs:
                if c in obj:
                    return False
            return c.c != bg

        if accept(cell):
            obj.add(cell)
            for neighbour in cell.close_neighbours():
                self.segment_connected(objs, obj, neighbour, bg)
        return obj

    def nof_objects(self):
        """Return the number of objects"""
        return len(self.object_shape())

    def holes(self):
        """Return a list of the holes"""
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

    def dots(self):
        """
        Return a list of the dots. A dot is a single cell of one colour that is
        not the background.
        """
        bg = self.background()
        return [c for c in self.cells() if c.c != bg]

    def object_shape(self):
        """
        Find the shape of the objects. It is assumed that
        the most common colour is the background.
        Return a list of objects.
        """
        bg = self.background()
        # Now segment the objects out of the background. This fails for
        # connected objects.
        objs = []
        for cell in self.cells():
            new_obj = GridShape()
            # Add to known objects so check for inclusion also works for this
            # new object
            objs.append(new_obj)
            obj = self.segment_connected(objs, new_obj, cell, bg)
            if len(obj) < 2:
                # Nothing in it so remove it. Dots don't count
                del objs[-1]
        # List of detected objects
        return objs

    def grid_size(self):
        """
        Return a tuple for the grid size
        """
        return (len(self._grid[0]), len(self._grid))

    def dots_aligned(self):
        """Return a list of dots that are aligned vertically or horizontally"""
        align = []
        for a, b in combinations(self.dots(), 2):
            # Get the one and only cell; they are dots
            if a.x == b.x or a.y == b.y:
                # Store both
                align.append((a, b))
        return align


class Observation(object):
    """
    An observation is a training input grid and an output grid
    """
    def __init__(self, json):
        self.input = Grid.from_json(json['input'], g_fancy_output)
        self.output = Grid.from_json(json['output'], g_fancy_output)

    def __str__(self):
        """Return a string representing both the input and the output"""
        return f"{self.input_as_str()}\n{self.output_as_str()}"

    def input_as_str(self):
        """Return string representation of input grid only"""
        return str(self.input)

    def output_as_str(self):
        """Return string representation of output grid only"""
        return str(self.output)

    def bg(self, from_input):
        """Return the background for input or output grid"""
        fr = self.input if from_input else self.output
        return fr.background()

    def holes(self, from_input):
        """Return the holes for input or output grid"""
        fr = self.input if from_input else self.output
        return fr.holes()

    def dots(self, from_input):
        """Return the dots for input or output grid"""
        fr = self.input if from_input else self.output
        return fr.dots()

    def nof_objects(self, from_input):
        """Return the number of objects for input or output grid"""
        fr = self.input if from_input else self.output
        return fr.nof_objects()

    def object_shape(self, from_input):
        """Return the object shapes for input or output grid"""
        fr = self.input if from_input else self.output
        shapes = fr.object_shape()
        # Fix up objects that appear to be cut off by edges
        centre_shapes = [shape for shape in shapes if not shape.on_edge(fr)]
        if len(shapes) and len(centre_shapes):
            for a, b in combinations(centre_shapes, 2):
                if not (a.contains(b) and b.contains(a)):
                    break
            else:
                # They are all the same. Assume the edge shapes are also the same
                shapes = [centre_shapes[0] for i in range(len(shapes))]
        return shapes

    def grid_size(self, from_input):
        """Return the grid size for input or output grid"""
        fr = self.input if from_input else self.output
        return fr.grid_size()

    def dots_aligned(self, from_input):
        """Return the aligned dots for input or output grid"""
        fr = self.input if from_input else self.output
        return fr.dots_aligned()


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
        for ob in self._obs:
            s += str(ob)
        s += "Tests\n"
        for test in self._test:
            s += str(test)
        return s

    def print_training_outputs(self):
        """Print the training inputs as per requirements"""
        for ob in self._obs:
            print(ob.output_as_str())

    def _find_features(self):
        """Run each of the feature detectors and collect the results for each
        of the examples, for both the inputs and the test cases.
        The detector detects all features that may be detected in input, output
        and test grids, without relating the features to one-another.
        The features that are detected should be sufficient to construct the
        test output from the test input, the "program" and common derived
        features (i.e. program constant data).
        This results in the dict with as key the name of the
        detector and as value a list with one entry per example.
        """
        features = {
            'bg'          : self.bg,
            'holes'       : self.holes,
            'dots'        : self.dots,
            'dots_aligned': self.dots_aligned,
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
        collect_features(self.input_feat, self._obs, True)
        collect_features(self.output_feat, self._obs, False)

        # Separately collect the features from the test example
        self.test_feat = {}
        collect_features(self.test_feat, self._test, True)

    def _add_patterns(self):
        """
        Using the gathered information, add patterns deduced from the
        information for each of the input and the output.
        E.g. if there are one, then two, then three objects
        in the inputs, see if it makes sense to add a case for four."""

        # No real need for patterns in these three samples.
        pass

    def _find_common_patterns(self):
        """
        Compare the detected patterns in input and output and deduce what
        the rules are that may be applied to convert the input into the output
        """

        # The input_feat, output_feat are both a dictionary with as key and
        # attribute. The value is a list, one for each observation, with either
        # a number (e.g. number of objects) or a list of objects in the grid
        # (e.g. shapes)

        # This should be refactored into a small language that would make it
        # much easier to add similar functionality.

        def common_keys():
            """Return the common keys (attributes) between inputs and outputs"""
            return [key for key in self.input_feat.keys() if key in self.output_feat]

        def is_equal(a, b):
            """Determine if the 'a' fact is the same as the 'b' fact"""
            def same_type(c, d):
                return type(c).__name__ == type(d).__name__

            if same_type(a, b):
                if isinstance(a, list):
                    # Can't compare lists in a meaningful way, so compare
                    # the elements ourselves
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

        # Add the fact that the output shapes are all the same.
        # Allow for partial shapes at the borders
        if 'object_shape' in self.output_feat:
            # For each observation
            for pairs in self.output_feat['object_shape']:
                same = True
                for a, b in combinations(pairs, 2):
                    if not a.contains(b) or not b.contains(a):
                        same = False
                        break
                if not same:
                    break
            else:
                # They are all the same so pick the first
                self.common_feat['same_output_shapes'] = self.output_feat['object_shape'][0][0]

        # Add the fact that the input dots are the same in the output.
        if 'dots' in self.input_feat and 'dots' in self.output_feat:
            match = []
            # For each observation
            for indots, outdots in zip(self.input_feat['dots'], self.output_feat['dots']):
                same_dots = []
                for indot in indots:
                    if not indot in outdots:
                        break
                    same_dots.append(indot)
                else:
                    match.append(same_dots)
            else:
                # List of matching output dots between input and output
                # Only add if all input dots matched output dots for all obs
                if len(match) == len(self.input_feat['dots']):
                    self.common_feat['dots_same_input_output'] = match

        # Add the fact that the input shapes are all the same
        if 'object_shape' in self.input_feat:
            # For each observation
            for pairs in self.input_feat['object_shape']:
                for a, b in combinations(pairs, 2):
                    if not a.contains(b) or not b.contains(a):
                        break
            else:
                # They are all the same so pick the first
                self.common_feat['same_input_shapes'] = self.output_feat['object_shape'][0][0]

        # Add the fact that holes are horizontally or vertically aligned
        # in the input and in the output
        if 'dots_aligned' in self.input_feat:
            if 'dots_aligned' in self.output_feat:
                self.common_feat['dots_aligned'] = self.input_feat['dots_aligned']

        # Add the fact that there is a connection between aligned dots in the
        # output, and it is not the background.
        # We use the input dots but check the output connections

        def find_output_match(ab, output_pairs):
            """Find a match in the output pairs"""
            if output_pairs:
                for pair in output_pairs:
                    if ab == pair:
                        return pair
            return (None, None)

        if 'dots_aligned' in self.common_feat:
            match = []
            for i, (pairs, output_pairs) in enumerate(zip(self.common_feat['dots_aligned'], self.output_feat['dots_aligned'])):
                # For all pairs in an observation
                bg = self.output_feat['bg'][i]
                fg = None
                pair_found = None
                for a, b in pairs:
                    ac, bc = find_output_match((a, b), output_pairs)
                    if ac != None and bc != None:
                        for c in ac.hv_align_to(bc):
                            # Several cases make us fail:
                            # - we found the first cell's foreground and the next
                            #   cell does not match
                            # - We found the background
                            if (fg != None and c.c != fg) or c.c == bg:
                                # Not the same colour, abort
                                fg = None
                                break
                            # fg is None, or it matched
                            fg = c.c
                        else:
                            pair_found = fg
                match.append(pair_found)

            for fg in match:
                if not fg:
                    break
            else:
                self.common_feat['hv_output_dots_aligned_bg'] = fg

        # For all facts that are the same in input and output, add the fact
        # that they are in fact the same
        for attr in common_keys():
            if is_equal(self.input_feat[attr], self.output_feat[attr]):
                self.common_feat[attr] = self.input_feat[attr]

    def _find_program(self):
        """
        Given common attributes, find an algorithm to go from input to output
        """
        program = [self.Isa.create_grid]
        if 'same_output_shapes' in self.common_feat and \
           'same_input_shapes' in self.common_feat and \
            'same_nof_objects' in self.common_feat and \
            'hv_output_dots_aligned_bg' not in self.common_feat:

            # This should be broken down into smaller steps, e.g.
            # 1. take shape,
            # 2. for each target location (where target locations are
            #                       determined by feature analysis)
            # 3.   copy shape to location.

            # should have the shape attached to same_output_shapes
            # this should be modeled like the other features earlier on.
            program.append(self.Isa.copy_output_shape_to_input_shape_pos)

        if 'dots_aligned' in self.test_feat and \
           'hv_output_dots_aligned_bg' in self.common_feat:
            program.append(self.Isa.connect_hv_alligned)

        if 'dots_same_input_output' in self.common_feat:
            program.append(self.Isa.copy_dots)

        return program

    class Isa(object):
        """
        Instruction set for executing a 'program' that creates an output from
        an input image.
        """
        @classmethod
        def create_grid(cls, answer, input_data, program_data):
            x, y = input_data['grid_size'][0]
            return Grid.from_size(x, y, 0, g_fancy_output)

        @classmethod
        def copy_output_shape_to_input_shape_pos(self, answer, input_data, program_data):
            for input_shapes in input_data['holes']:
                for input_shape in input_shapes:
                    cx, cy = input_shape.pos()
                    answer.add_shape(cx, cy, program_data['same_output_shapes'])
            return answer

        @classmethod
        def copy_dots(self, answer, input_data, program_data):
            for dots in input_data['dots']:
                # Multiple grids, but for tests really only one
                for dot in dots:
                    answer.put(dot)
            return answer

        @classmethod
        def connect_hv_alligned(cls, answer, input_data, program_data):
            if 'hv_output_dots_aligned_bg' in program_data:
                bg = program_data['hv_output_dots_aligned_bg']
                for pairs in input_data['dots_aligned']:
                    for a, b in pairs:
                        for c in a.hv_align_to(b):
                            answer.put(Cell(answer, c.x, c.y, 8))
            return answer

    def execute(self, program, input_data, program_data):
        """Execute the program on a grid."""

        # Print the test sample
        print(self._test[0].input_as_str())

        answer = None
        for instruction in program:
                answer = instruction(answer, input_data, program_data)
                if answer == None:
                    print("Seg fault")
                    return 255

        # Print the result
        print(answer)
        return 0

    def solve(self):
        """
        Apply a number of recognition methods to find out what the salient
        features of this task is.
        """

        # Find elementary features such as holes, shapes etc.
        self._find_features()

        # Now we have basic information of the inputs and the test cases.
        # Add more information by adding patterns. For example, if the input
        # has a number of shapes, and the output always has one more, infer that
        # a shape must be added.
        self._add_patterns()

        # Compare the inputs and the test cases to see what patterns are
        # common. Throw out the ones that are not to reduce the search space.
        # For example, if the inputs all have dots, and the output all have
        # shapes, infer that dots correspond to shapes.
        self._find_common_patterns()

        # Search for a set of transformations that map the inputs to the
        # outputs. Store these as a set of instructions to apply to the test
        # grid.
        self._program = self._find_program()

        # Execute the program on the test example. The features in the test grid
        # are already collected so apply the program, taking the features in
        # the test grid and the common features as input data.
        return self.execute(self._program, self.test_feat, self.common_feat)

    # The idea with the functions below is that they can be modified easily
    # to allow more facts to be added based on their specific findings. This
    # turns out to be unnecessary for the samples that were chosen so they
    # could be refactored out. There is duplication, not very DRY.

    def nof_objects(self, res, source, from_input):
        """
        Count the number of objects
        Add a list with an entry per observation
        """
        l = []
        for obs in source:
            l.append(obs.nof_objects(from_input))
        res['nof_objects'] = l

    def bg(self, res, source, from_input):
        """
        Determine the background (most common colour)
        """
        l = []
        for obs in source:
            l.append(obs.bg(from_input))
        res['bg'] = l

    def holes(self, res, source, from_input):
        """
        Count the number of 'holes'
        Add a list with an entry per observation
        """
        l = []
        for obs in source:
            l.append(obs.holes(from_input))
        res['holes'] = l

    def dots(self, res, source, from_input):
        """
        Count the number of 'dots'. A dot is a single cell of one colour
        Add a list with an entry per observation
        """
        l = []
        for obs in source:
            l.append(obs.dots(from_input))
        res['dots'] = l

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

    def dots_aligned(self, res, source, from_input):
        """
        Determine if the dots, if any, are aligned
        """
        l = []
        for obs in source:
            l.append(obs.dots_aligned(from_input))
        res['dots_aligned'] = l


class Arc(object):

    def __init__(self, filename):
        self._task = Task(self.load(filename))

    def load(self, filename):
        """
        Load an ARC Json file, convert to an array and return.
        """
        with open(filename) as fp:
            data = json.load(fp)
        return data

    def print_training_outputs(self):
        self._task.print_training_outputs()

    def solve(self):
        """Solve the puzzle and store the resulting program."""
        return self._task.solve()

def solve(filename):
    """
    Run a sample, do the analysis and store a program to apply to a test case
    """
    arc = Arc(filename)
    arc.print_training_outputs()
    return arc.solve()

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <arc task name>. Stopping.")
        sys.exit(-1)

    # If this is a filename, then run this filename. If it is a glob
    # then run all files. If we have multiple filenames then run all of them.
    for args in sys.argv[1:]:
        for filename in glob.glob(args):
            solve(filename)
    exit(0)

