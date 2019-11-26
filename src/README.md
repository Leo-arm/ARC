# The Abstraction and Reasoning Corpus (ARC)

This code implements a framework and a number of low-level detectors to solve
the Abstractions and Reasoning Corpus. Three hand-picked examples are used
to develop the code.

The approach is to first analyse the input and output grids for low level features
like holes, objects, background etc. Once the low level features are recovered
the features are analysed and a higher level set of facts is discovered, such as
that all shapes in the input are the same, that all dots in the input match all dots
in the output etc. Each fact that applies is added to a dict. This dict thus
collects all features that may be relevant.
Once all facts have been added, an (currently rather rough and ready) 
determination is made how to map the input to the output. This currently
generates a very rudimentary "program", really a set of function pointers, that
call functions do the actual mapping. Finally, this program is executed.

Testing
Testing with doctest is very awkward because the setup that is required is
substantial. Setting up a mock framework costs some time so to expedite the
process I resorted to my fallback, which is to single-step through the code
the first time it is run and manually verify results. Needless to say, this is
not sustainable in a complex and long-term project.

Some observations:
- subleties: initially "holes" were used as the commonality between the three
  samples. This did not work out as a hole was defined as a cell that has a
  different colour as the neighbouring cells. It turns out that in one example
  two cells that are obviously holes to a human are located at 0,0 and -1,-1
  (one to the lower right of the other) so are not really holes. Also, a cell
  near a corner is easily mistaken as a candidate because two out of three have
  the same colour and one neighbour is different.
- crossover between features
  What looks like a generic feature often is not. E.g. a shape in one problem
  is really a connector in another. The solution is to treat a connector as
  a shape, i.e. break down features to lower level features,
  but sadly a lack of time stops me from doing this
- Dealing with one case at the time is one thing, but the feature detectors
  are interacting so the features need to be fine-grained to avoid overlap
  where none is intended.
  
Corner case if two seemingly separate objects are connected over a corner
only

Future directions
The current features are too high-level and need to be broken down into more
elementary attributes.
Once elementary features are determined, then an (implicit or explicit)
dependency graph could be used to deduce higher level features. It seems
clear that a hierarchy of features can be used to solve these puzzles.
Alternatively, the program could be constructed from the input and output grids
and then the neccessary low level features could be detected as part of the
program, i.e. a more lazy implementation than the current.   


## Running
There are a few ways to run this aside from following the requirements.
You can use the following command-line:
../data/training/*.json
Run on all filenames that match the wildcard character
../data/training/d364b489.json ../data/training/dbc1a6ce.json ../data/training/b60334d2.json
Run on these files
../data/training/d364b489.json
Run on this file, as per requirement
There must be at least one parameter to the script.

## Implementation

