# Traveling Salesman Problem Research

Implementation of research related to different methods of solving the Traveling Salesman Problem. Performed for the Capstone Project as part of Temple University's Master's in Computer Science program.

## Setup

### With Python and Pip

* Install dependencies with `pip install -r requirements.txt`
* Install pyconcorde via the instructions on [GitHub](https://github.com/jvkersch/pyconcorde)

### With uv

* Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
* Install Python 3.13 via `uv python install 3.13.1`
* Install dependencies via `uv sync`
* Create a virtual environment with `uv venv --python 3.13.1`
* Activate the environment via `source .venv/bin/activate`
* Install dependencies via `uv sync`
* Install pyconcorde
  * Clone the repository on [GitHub](https://github.com/jvkersch/pyconcorde)
  * `cd` into the directory and run `uv pip install -e .`

## TSP Problem Data

Problems used are located in the `data` repo and can be called by name in the `main.py` script. Additional problems can 
be added by downloading a [tsplib](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)-compatible files and adding 
them to the `data` directory. Files can be decompressed with `gzip -d`.

## Algorithm Implementations

The algorithm implementations are included in `src/algorithms`. Algorithms have been implemented by using procedures described in research papers, with help from online resources as needed. None of the concepts are original at this point, but all code is my own, and any references used in created the code are listed.

Code follows object-oriented programming principles. All algorithms inherit from the base class `models/tsp::TSP`, and all except the Concorde wrapper inherit from `models/networkx_tsp::NetworkxTSP`, a subclass of `TSP`.

### External Source of Truth

* `concorde.py` - This is an external library that is used as the source of truth. For example, for any problems created without a known, exact solution, whatever Concorde gives will be considered the optimal solution. More information about the program is available here: https://www.math.uwaterloo.ca/tsp/concorde.html

### Exact Methods

* `brute_force.py` - A brute-force implementation. This was really just done to get a better understanding of the problem and to show why it is not a good solution.
* `held_karp.py` - A dynamic programming implementation. Similar to brute force, was done more to gain a better understanding of exact solutions to the problem. Implemented based on  Held-Karp's [A Dynamic Programming Approach to Sequencing Problems](https://www.jstor.org/stable/2098806).

### Ant Colony Optimization

* `aco` - Module for implementation of Ant Colony Optimization (ACO) algorithms. Algorithms were created by following Chatper 3 of [Ant Colony Optimization by Dorigo and Stutzle](https://web2.qatar.cmu.edu/~gdicaro/15382/additional/aco-book.pdf).
  * `base_aco.py` - Baseline implementation of ACO solutions to TSP. There are several types of this algorithm, and this file contains functions used by all of them. By default, runs the Ant System algorithm.
  * `ant_system.py` - The most basic ACO algorithm. Simply inherits everything from `BaseACO`.
  * `min_max_ant_system.py` - Improves upon Ant System. Longer runtime but generally better performance than Ant System

### Q-Learning

* `q_learning` - Module for implementation of Q-learning algorithms. Created following [Reinforcement learning for the traveling salesman problem: Performance comparison of three algorithms](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/tje2.12303).
  * `base_q_learning.py` - Base class containing functionality that all Q-learning algorithms will use. Requires inheriting classes to use implement a function to update the Q-table, and a function to exploit the knowledge gained from the Q-table during training
  * `q_learning.py` - Solves TSP using Q-learning algorithm
  * `double_q_learning.py` - Solves TSP using double Q-learning algorithm. Similar to Q-learning, but manages two Q-matrices that interact with each other throughout training to better balance information and prevent stagnation.

### Other Methods

* `nearest_neighbor_search.py` - Nearest-neighbor search algorithm adapted to generate routes for the Traveling-Salesman Problem. Not meant to be a solution, but used in setting up the Ant Colony Optimization algorithm

## Running the repo

Everything runs through the `main.py` script. The script takes 2 arguments. All available options can be seen by running `python main.py --help`.

* `-p` or `--problem` - Required. Specifies the problem to run by its filename without the suffix. For example, `-p fri26` will run the problem from the file at `data/tsplib/fri26.tsp`.
* `-a` or `--algorithm` - Optional. Algorithm(s) to run by the abbreviated name defined at `main.py:24`. If not specified, runs the problem using the Concorde solver, Min-Max Ant System, and Double Q-Learning. For example, running `python main.py -p p5` will run the problem from the file at `data/custom/p5.tsp` on these 3 algorithms.

## Runtime

Currently, excluding the exact methods, Min-Max Ant System takes the longest to run. Around half an hour should be allocated if running on one of the 50+ city problems. Exact runtimes for several methods can be found in `results.csv`. Generally `barachet10.tsp` is used frequently in testing, as it runs quickly and is slightly challenging for an algorithm to solve, so is a decent way to test new implementations. The problem was taken from L.L. Barachet's [Graphic Solution of the Traveling-Salesman Problem](https://www.jstor.org/stable/166872).

## Workload stack

### Current Week

* Documentation and strong typing for existing functions
* Write midterm paper
* Script that automatically writes results

### Part II

* Algorithm tuning, mainly for Q-learning
* Exploring unique problem shapes
* Neural network with Q-learning

## Future Work

* Dynamic Traveling-Salesman Problem
* Epsilon factor in ACO
