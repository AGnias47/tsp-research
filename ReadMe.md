# Traveling Salesman Problem Research

Implementation of research related to different methods of solving the Traveling Salesman Problem. Performed for the Capstone Project as part of Temple University's Master's in Computer Science program.

## Setup

* Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
* Install Python 3.13 via `uv python install 3.13.1`
* Install dependencies via `uv sync`
* Create a virtual environment with `uv venv --python 3.13.1`
* Activate the environment via `source .venv/bin/activate`
* Install dependencies via `uv sync`
* Install pyconcorde via the instructions on [GitHub](https://github.com/jvkersch/pyconcorde)

### TSP problems

By default, problems are not provided with this repo. Problems can be downloaded from the 
[tsplib](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) website. Once downloaded, they can then be referenced 
by their absolute path, or can be placed in a local subdirectory and specified in the config file. `problems_parent_path` defines the parent path containing all problems. 
`problems` contains entries where the key is the problem name and the value is the 
problem subpath, i.e. `problems_parent_path`/`<problem subpath>`.

Custom problems can be created so long as they are in a `tsplib` compatible format.

#### Sample Problem

Taken from Barachet, ["Graphic Solution of the Traveling-Salesman Problem"](https://www.jstor.org/stable/166872). Optimal cost is `378` and optimal route is `[0 6 5 7 8 9 4 3 2 1 0]`.

```shell
NAME: barachet10
TYPE: TSP
COMMENT: Barachet's 10 City Example from Graphic Solution of the Traveling-Salesman Problem
DIMENSION: 10
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: LOWER_DIAG_ROW
EDGE_WEIGHT_SECTION
0
28 0
57 28 0
72 45 20 0
81 54 30 10 0
85 57 28 20 22 0
80 63 57 72 81 63 0
113 85 57 45 41 28 80 0
89 63 40 20 10 28 89 40 0
80 63 57 45 41 63 113 80 40 0
EOF
```

## Run

Run the main script via

```shell
python main.py -p <problem name or comma-separated list of names>
```

By default, the problem(s) will run on all available algorithms. Specific algorithms can be specified with the `-a` parameter, either as a single algorithm abbreviation or a comma-separated list of algorithm abbreviations (see `main.py::ALGORITHMS` for full list).

Tests can be run via

```shell
pytest
```

## Future Work

* Dynamic Traveling-Salesman Problem

