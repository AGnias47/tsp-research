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

## Run

Run the main script via

```shell
python main.py -p <problem name>
```

Tests can be run via

```shell
pytest
```

## Future Work

* Dynamic Traveling-Salesman Problem

