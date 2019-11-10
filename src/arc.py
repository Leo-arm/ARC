# Main class for the ARC solver

import io
from util import Util
from collections import defaultdict
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


class Grid(object):
    # TODO(Leo): change colour output to False before submitting
    def __init__(self, grid, colour=True):
        # Fix the null/None values
        self._grid = [[x if x else 0 for x in y] for y in grid]
        # Debug in full colour :-)
        self.colour = self.ansi if colour else self.nocolour

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

    def cells(self):
        """
        Iterate over cells from top left to bottom right
        """
        for y, row in enumerate(self._grid):
            for x, c in enumerate(row):
                yield(Cell(self, x, y, c))

    def nof_objects(self):
        freq = defaultdict(int)
        for cell in self.cells():
            freq[cell.c] += 1
        return len(freq.keys())


    def nof_holes(self):
        # TODO(Leo): larger holes
        holes = 0
        for cell in self.cells():
            for neighbour in cell.neighbours():
                if cell.c == neighbour.c:
                    break
            else:
                # Neighbours are not the same as this cell
                holes += 1
        return holes


class Observation(object):
    def __init__(self, json):
        self.input = Grid(json['input'])
        self.output = Grid(json['output'])

    def __str__(self):
        return f"Input\n{str(self.input)}\nOutput:\n{str(self.output)}"

    def nof_holes(self):
        return self.input.nof_holes()

    def nof_objects(self):
        return self.input.nof_objects()


class Task(object):

    def __init__(self, json):
        self._obs = []
        for obs in json['train']:
            self._obs.append(Observation(obs))
        self._test = []
        for test in json['test']:
            self._test.append(Observation(test))

    def __str__(self):
        s = "Observations\n"
        for obs in self._obs:
            s += str(obs)
        s += "Tests\n"
        for test in self._test:
            s += str(test)
        return s

    def solve(self):
        """
        Apply a number of recognition methods to find out what the salient
        features of this task is.
        """
        features = {
            'nof_holes': self.nof_holes,
            'nof_objects': self.nof_objects,
        }

        # Run each of the features and collect the results for each of the
        # examples. This results in the dict with as key the name of the
        # detector and as value a list with one entry per example.
        res = {}
        for name, feature in features.items():
            feature(res)

        print(res)
        table = [res[k] for k in res.keys()]
        table = list(map(list, zip(*table)))
        pp(table)

        pca = PCA(n_components=2)
        pc = pca.fit_transform(table)
        pp(pc)

        # Another way to look at this is that we need a set of transformations
        # such that the difference between the input and the output is 0 for
        # that attribute. So just throw transformations at it until we have
        # a vector that is perfectly correlated.

        fig = plt.plot(figsize = (8,8))
        plt.scatter(pc[0], pc[1])
        plt.show()


    def nof_objects(self, res):
        l = []
        for obs in self._obs:
            l.append(obs.nof_objects())
        res['nof_objects'] = l

    def nof_holes(self, res):
        l = []
        for obs in self._obs:
            l.append(obs.nof_holes())
        res['nof_holes'] = l


class Arc(object):

    def __init__(self, filename):
        self._task = Task(Util.load(filename))

    def solve(self):
        return self._task.solve()
