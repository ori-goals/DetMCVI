# Experimental Domains
In each domain there is a header file `<domain>.h`, and code which calls detMCVI and other benchmarks in `<domain>.cpp`. 
The file `<domain>_timeseries.cpp` runs the benchmarks with evaluations at regular intervals instead of just at the end.
See `include/Params.h` for the input parameters.

## Usage
```
<domain>_timeseries <problem_file> <output_file> [options]
```
`<problem_file>` should be a text file containing the problem specification, as detailed in each section below.
`<output_file>` will be the csv file for the program to write results to.
Use `--help` to find out more about the available options.

## Domains
### Canadian Traveller Problem

The Canadian Traveller Problem (CTP) was introduced in Papadimitriou and Yannakakis (1989).
An agent must navigate a weighted connected graph to a goal node, but some edges are blocked and can only be discovered to be blocked upon reaching an adjacent node.
We assume that the layout of the map and the probability of edge blockage are known a-priori.

To run the Canadian Traveller Problem example, run `build/experiments/CTP_experiment`.
The file `experiments/CTP_generator.py` is provided to generate graphs for the CTP experiment, but manual graphs can also be specified.

An example graph input looks like:
```
CTPNodes: 0 1 2 3
CTPEdges:
0 1 1 # start node, end node, edge weight
0 2 1
1 3 1
2 3 3
CTPStochEdges:
1 3 0.5 # start node, end node, probability of blockage
CTPOrigin: 0
CTPGoal: 3
```


### Wumpus World

Wumpus World is described in Russell and Norvig (2021).
The player must find the gold and climb out, avoiding breezy pits and the smelly Wumpus.
Actions are to turn left or right, move forward, shoot an arrow, pick up the gold, and climb out.

An example problem file looks like
```
4
```
This is just the grid size, the rest is defined in the problem.


### Maze

The Maze domain is a problem of reaching a goal in a maze where the shape of the maze is known but the initial position is not.
Only the orthogonally adjacent tiles are observable.
There are four actions, one to move in each direction.

An example problem file looks like
```
###########
#     #   #
##### # ###
#     #   #
##### # ###
#       # #
# # # ### #
# # # #   #
# ##### # #
#       #G#
###########
```
G marks the goal, spaces are possible starting states.


### SortGame

The sort game is a problem of sorting a list of numbers by swapping two elements.
The action space is the set of possible swaps.
The observation is the Spearman Footrule Distance between the current list and the sorted list.

An example problem file looks like
```
5
```
This is the number of items to be sorted

## References
Bai, H. et al. (2011) ‘Monte Carlo value iteration for continuous-state POMDPs’, in. Algorithmic Foundations of Robotics IX: Selected Contributions of the Ninth International Workshop on the Algorithmic Foundations of Robotics, Springer, pp. 175–191.

Papadimitriou, C.H. and Yannakakis, M. (1989) ‘Shortest paths without a map’, in Ausiello, G., Dezani-Ciancaglini, M., and Della Rocca, S.R. (eds) Automata, Languages and Programming. Berlin, Heidelberg: Springer Berlin Heidelberg (Lecture Notes in Computer Science), pp. 610–620. doi:10.1007/BFb0035787.

Russell, S. and Norvig, P. (2021) Artificial Intelligence: A Modern Approach, Global Edition. Pearson Education. Available at: https://books.google.co.uk/books?id=cb0qEAAAQBAJ.


