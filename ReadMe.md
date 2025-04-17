# Traveling Salesman Problem Research

Implementation of research related to different methods of solving the Traveling Salesman Problem. Performed for the Capstone Project as part of Temple University's Master's in Computer Science program.

## Setup

### With Python and Pip

* Install dependencies with `pip install -r requirements.txt`
* Install pyconcorde via `pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde'`
  * Note that this step can be skipped, as `pyconcorde` can only be installed on Linux. If not installed, a No-op class that simply returns the best known cost from a config will be used in place of Concorde

### With uv

* Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
* Install Python 3.13 via `uv python install 3.13.2`
* Create a virtual environment with `uv venv --python 3.13.2`
* Activate the environment via `source .venv/bin/activate`
* Install dependencies via `uv sync`
* Install pyconcorde via `uv pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde'`
  * Note that this step can be skipped, as `pyconcorde` can only be installed on Linux. If not installed, a No-op class that simply returns the best known cost from a config will be used in place of Concorde

## TSP Problem Data

Problems used are located in the `data` repo and can be called by name in the `main.py` script. Problems should be in a [tsplib](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)-compatible format. Additional problems can be added by creating them, or downloading them from tsplib and adding them to the `data` directory. Files from tsplib can be decompressed with `gzip -d`.

## Algorithm Implementations

The algorithm implementations are included in `src/algorithms`. Algorithms have been implemented by using procedures described in research papers, with help from online resources as needed. None of the concepts are original at this point, but all code is my own, and any references used in code are listed.

Code follows object-oriented design principles. All algorithms inherit from the base class `models/tsp::TSP`, and all except the Concorde wrapper inherit from `models/networkx_tsp::NetworkxTSP`, a subclass of `TSP`.

### External Source of Truth

* `concorde.py` - This is an external library that is used as the source of truth. For example, for any problems created without a known, exact solution, whatever Concorde gives will be considered the optimal solution. More information about the program is available here: https://www.math.uwaterloo.ca/tsp/concorde.html

### Exact Methods

* `brute_force.py` - A brute-force implementation. This was really just done to get a better understanding of the problem and to show why it is not a good solution.
* `held_karp.py` - A dynamic programming implementation. Similar to brute force, was done more to gain a better understanding of exact solutions to the problem. Implemented based on Held-Karp's [A Dynamic Programming Approach to Sequencing Problems](https://www.jstor.org/stable/2098806).

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
  * `deep_q_learning.py` - Solves TSP utilizing a feedforward neural network instead of a Q-table.

### Other Methods

* `nearest_neighbor_search.py` - Nearest-neighbor search algorithm adapted to generate routes for the Traveling-Salesman Problem. Not meant to be a solution, but used in setting up the Ant Colony Optimization algorithm

## Running the repo

Everything runs through the `main.py` script. The script takes 2 arguments. All available options can be seen by running `python main.py --help`.

* `-p` or `--problem` - Required. Specifies the problem to run by its filename without the suffix. For example, `-p fri26` will run the problem from the file at `data/tsplib/fri26.tsp`.
* `-a` or `--algorithm` - Optional. Algorithm(s) to run by the abbreviated name. If not specified, runs the problem using the Concorde solver, Min-Max Ant System, and Double Q-Learning. For example, running `python main.py -p p5` will run the problem from the file at `data/custom/p5.tsp` on these 3 algorithms. Other options include
  * `concorde` - Concorde solver
  * `nns` - Nearest Neighbor Search
  * `bf` - Brute force
  * `hk` - Held-Karp Dynamic Programming solution
  * `as` - Ant System
  * `mmas` - Max-Min Ant System
  * `q` - Q-Learning
  * `dq` - Double Q-Learning
  * `dqn` - Deep Q-Learning
  * `aco` - Ant System and Max-Min Ant System
  * `rl` - Q-Learning and Double Q-Learning
  * `proj` - Concorde, Max-Min Ant System, and Double Q-Learning
  * `all` - All available algorithms
* `m` or `--mlflow-project` - Optional. Logs results to MLflow as an experiment under the specified name

### View Results

Results for tracked experiments are stored in a local instance of MLflow. To view the results, do the following

* Run the MLflow server locally via `mlflow server`
* Go to the provided URL, likely http://127.0.0.1:5000
* Click on the desired experiment
* In the "Columns" dropdown, deactivate all the "Attributes" columns and activate the "Metrics" and "Parameters" columns

## Generate UML

Done with pyreverse.

```shell
cd src
uv run pyreverse -o png -p . .
```

## Future Work

* Dynamic Traveling-Salesman Problem
* 3D Instances
